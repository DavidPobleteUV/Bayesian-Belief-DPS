# bootstrap_new_pc.ps1
# -------------------------------------------------------------------
# Configuración inicial del workflow DPS↔MLP↔WEAP en una PC nueva.
#
# Asume que ya hiciste git clone de los 3 repos y los pusiste en
# C:\Users\<user>\Documents\GitHub_DPL\ (o cambia $REPOS_ROOT abajo).
#
# Ejecuta en PowerShell desde la carpeta donde está este script:
#     .\bootstrap_new_pc.ps1
# -------------------------------------------------------------------

$ErrorActionPreference = "Stop"

# Si tu carpeta raíz es diferente, edita esta variable:
$REPOS_ROOT = "$HOME\Documents\GitHub_DPL"

$DPS_REPO   = "$REPOS_ROOT\Bayesian-Belief-DPS"
$MLP_REPO   = "$REPOS_ROOT\WEAP_HydroMLP_RecursiveGW"
$ZARR_REPO  = "$REPOS_ROOT\WEAP_2_ZARR"

function Section($msg) {
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host $msg -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan
}

function CheckRepo($path, $name) {
    if (-not (Test-Path $path)) {
        Write-Host "[ERROR] No existe el repo: $path" -ForegroundColor Red
        Write-Host "  Clónalo primero con: git clone https://github.com/<usuario>/$name.git $path" -ForegroundColor Yellow
        exit 1
    }
    Write-Host "[OK] $name encontrado: $path" -ForegroundColor Green
}

# ─── Paso 1: Verificar repos ─────────────────────────────────────────
Section "Paso 1 — Verificación de repos"
CheckRepo $DPS_REPO  "Bayesian-Belief-DPS"
CheckRepo $MLP_REPO  "WEAP_HydroMLP_RecursiveGW"
CheckRepo $ZARR_REPO "WEAP_2_ZARR"

# ─── Paso 2: Venv del MLP ────────────────────────────────────────────
Section "Paso 2 — Venv del MLP (WEAP_HydroMLP_RecursiveGW)"
Set-Location $MLP_REPO
if (-not (Test-Path "venv_HydroMLP\Scripts\python.exe")) {
    Write-Host "Creando venv_HydroMLP..." -ForegroundColor Yellow
    python -m venv venv_HydroMLP
    & ".\venv_HydroMLP\Scripts\python.exe" -m pip install --upgrade pip --quiet
    & ".\venv_HydroMLP\Scripts\python.exe" -m pip install -r requirements.txt --quiet
    & ".\venv_HydroMLP\Scripts\python.exe" -m pip install -e . --quiet
    Write-Host "[OK] venv_HydroMLP creado e instalado" -ForegroundColor Green
} else {
    Write-Host "[OK] venv_HydroMLP ya existe" -ForegroundColor Green
}

# ─── Paso 3: Venv del DPS ────────────────────────────────────────────
Section "Paso 3 — Venv del DPS (Bayesian-Belief-DPS)"
Set-Location $DPS_REPO
if (-not (Test-Path "venv_DPS\Scripts\python.exe")) {
    Write-Host "Creando venv_DPS..." -ForegroundColor Yellow
    python -m venv venv_DPS
    $py = ".\venv_DPS\Scripts\python.exe"
    & $py -m pip install --upgrade pip --quiet
    & $py -m pip install torch pytorch-lightning numpy pandas zarr scipy platypus-opt pyyaml scikit-learn matplotlib --quiet
    & $py -m pip install -e $MLP_REPO --quiet
    Write-Host "[OK] venv_DPS creado e instalado" -ForegroundColor Green
} else {
    Write-Host "[OK] venv_DPS ya existe" -ForegroundColor Green
}

# ─── Paso 4: Venv del WEAP_2_ZARR (opcional, solo si vas a correr WEAP) ──
Section "Paso 4 — Venv del WEAP_2_ZARR (opcional)"
Set-Location $ZARR_REPO
if (-not (Test-Path "venv_WEAP2Zarr\Scripts\python.exe")) {
    Write-Host "Creando venv_WEAP2Zarr..." -ForegroundColor Yellow
    python -m venv venv_WEAP2Zarr
    $py = ".\venv_WEAP2Zarr\Scripts\python.exe"
    & $py -m pip install --upgrade pip --quiet
    if (Test-Path "requirements.txt") {
        & $py -m pip install -r requirements.txt --quiet
    } else {
        & $py -m pip install numpy pandas pyyaml zarr pywin32 openpyxl scipy --quiet
    }
    Write-Host "[OK] venv_WEAP2Zarr creado" -ForegroundColor Green
} else {
    Write-Host "[OK] venv_WEAP2Zarr ya existe" -ForegroundColor Green
}

# ─── Paso 5: Verificación de import ──────────────────────────────────
Section "Paso 5 — Verificación de imports"
Set-Location $DPS_REPO
& ".\venv_DPS\Scripts\python.exe" -c "from rdm_mlp.models.lightning_module import WEAPHydroMLPLightning; print('[OK] rdm_mlp importable desde venv_DPS')"

# ─── Paso 6: Carpetas necesarias ─────────────────────────────────────
Section "Paso 6 — Carpetas del bridge"
$dirs = @(
    "$DPS_REPO\data_weap",
    "$DPS_REPO\data_weap\climate_base",
    "$DPS_REPO\data_weap\exports",
    "$DPS_REPO\runs_weap",
    "$DPS_REPO\tests"
)
foreach ($d in $dirs) {
    if (-not (Test-Path $d)) {
        New-Item -ItemType Directory -Force -Path $d | Out-Null
        Write-Host "  Creada: $d" -ForegroundColor Green
    }
}

# ─── Paso 7: Próximos pasos ──────────────────────────────────────────
Section "Próximos pasos manuales"
Write-Host @"
1. Asegúrate que el repo del modelo tenga el checkpoint entrenado:
     $MLP_REPO\runs\best_model.ckpt
   Si no existe, transfiere el .ckpt desde la PC original o reentrena.

2. Asegúrate que el zarr de training data exista:
     $MLP_REPO\data\weap_weekly.zarr
   Si no existe, transfiérelo desde la PC original o regenéralo desde WEAP_2_ZARR.

3. Para correr WEAP en esta PC, debes tener:
   - WEAP instalado con licencia
   - El área Quilimari_WEAP_MODFLOW_RDM en:
       C:\Users\<user>\Documents\WEAP Areas\Quilimari_WEAP_MODFLOW_RDM\
   - Todos los CSVs de GCMs en la subcarpeta GCMs\
   - La carpeta Policies\ (vacía, se llena en cada iteración del DPS)

4. Activa el venv del DPS y corre el primer extract:
     cd $DPS_REPO
     .\venv_DPS\Scripts\Activate.ps1
     python weap_dps/extract_data.py
     python tests/test_mlp_surrogate.py

5. Sigue el manual completo:
     $DPS_REPO\weap_dps\QUICKSTART_OTHER_PC.md
"@ -ForegroundColor Yellow

Write-Host ""
Write-Host "[BOOTSTRAP COMPLETADO]" -ForegroundColor Green
