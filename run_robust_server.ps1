# run_robust_server.ps1 - Robust DPS con el modelo iter1_clean_h128.
#
# NOTA: este archivo se mantiene en ASCII puro a proposito. Windows PowerShell
# 5.1 lee los .ps1 como ANSI si no tienen BOM, y cualquier acento o guion largo
# rompe el parser en el servidor.
#
# Lanza N semillas EN PARALELO (1 core por proceso, OMP=1). Varias semillas
# permiten evaluar CONVERGENCIA: si los frentes de distintas semillas coinciden,
# el resultado es robusto; si no, faltan evaluaciones.
#
# Preset por defecto:
#   3 semillas x 6000 evaluaciones x (5 climas x 3 demandas = 15 escenarios)
#   ~17.7 s por evaluacion  ->  ~29 h de reloj (semillas en paralelo)
#
# Reanudable: si un .dat ya existe, esa semilla se salta.
#
# Uso:
#   .\run_robust_server.ps1
#   .\run_robust_server.ps1 -Evaluations 4000
#   .\run_robust_server.ps1 -Seeds 42,123,456,789,1010,2020,3030

param(
    [int]   $Evaluations = 6000,
    [int]   $Population  = 100,
    [int]   $NClimate    = 5,
    [double]$Lambda      = 1.0,
    [int[]] $Seeds       = @(42, 123, 456),
    [string]$OutDir      = "runs_weap\robust_iter1_h128"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py = "python"
if (Test-Path "venv_DPS\Scripts\python.exe") { $py = "venv_DPS\Scripts\python.exe" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# --- Chequeos previos (fallar aqui es barato; a las 20 h no) ---
$need = @("data_weap\best_model.ckpt",
          "data_weap\X_template.npz",
          "data_weap\scalers_weap.npz",
          "data_weap\transform_params_weap.npz",
          "data_weap\manifest_inputs.csv",
          "data_weap\train_subset.zarr")
foreach ($f in $need) {
    if (-not (Test-Path $f)) {
        throw "Falta $f . Copia los artefactos del modelo desde la PC de entrenamiento."
    }
}

# Dependencias: un import que falla mata el proceso a los 2 s y deja el .err
# como unica traza. Mejor descubrirlo aqui, en una sola linea, que en N logs.
$mods = "numpy,torch,pytorch_lightning,zarr,pandas,platypus,rdm_mlp"
# importlib.util NO queda disponible con solo "import importlib": es un
# submodulo y hay que importarlo explicitamente.
$chk = & $py -c @"
import importlib.util as u
print(','.join([m for m in '$mods'.split(',') if u.find_spec(m) is None]))
"@
if ($LASTEXITCODE -ne 0) { throw "No se pudo ejecutar '$py'. Revisa el interprete." }
if ($chk.Trim()) {
    throw ("Faltan modulos: {0}`nInstala con:  pip install {1}" -f `
           $chk.Trim(), ($chk.Trim() -replace ',', ' ' -replace 'pytorch_lightning', 'pytorch-lightning'))
}

# NO setear DPS_J4_CAL: forzaria un factor ESCALAR y anularia la calibracion
# condicional al numero de acciones (config_weap.j4_calibration_factor).
Remove-Item Env:\DPS_J4_CAL     -ErrorAction SilentlyContinue
Remove-Item Env:\DPS_TRAIN_ZARR -ErrorAction SilentlyContinue
$env:OMP_NUM_THREADS = "1"      # 1 core por proceso; si no, se pisan entre si
$env:MKL_NUM_THREADS = "1"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"     # sin esto Python bufferea al redirigir a archivo
                                # y los logs (y check_progress) van muy atrasados
$env:DPS_WATERFALL = "0"        # cascada determinista OFF (probada: empeora J4)

$nScen = $NClimate * 3
# 1.92 s por rollout de escenario, MEDIDO con iter1_clean_h256 (28.8 s por
# evaluacion de 15 escenarios) tanto en el servidor como en la PC de trabajo.
# El valor anterior (1.18) venia de medir h128 con UN escenario y subestimaba
# el ETA a la mitad.
$eta   = [math]::Round($Evaluations * 1.92 * $nScen / 3600, 1)
$seedList = $Seeds -join ", "
$modelo = Split-Path $OutDir -Leaf     # el nombre estaba fijo como "h128"

# --- Aviso de RAM: cada proceso carga el modelo + los N escenarios en memoria.
# Con muchas semillas en paralelo se puede agotar la RAM y los procesos mueren
# sin dejar traza clara (el .err a veces queda vacio si el SO los mata).
$ramGB = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB
# ESTIMACION (no medida): ~0.9 GB de base por proceso (torch + modelo) + ~5 MB
# por escenario. Si los procesos mueren sin dejar nada en el .err, sospechar de
# RAM: el SO los mata sin traza. Si el .err tiene un traceback, es otra cosa.
$perProc  = 0.9 + 0.005 * $nScen
$needGB   = [math]::Round($Seeds.Count * $perProc, 1)
$maxSeeds = [math]::Max(1, [math]::Floor(($ramGB - 2.0) / $perProc))   # 2 GB para el SO
if ($Seeds.Count -gt $maxSeeds) {
    Write-Warning ("RAM insuficiente: {0} semillas x ~{1} GB = ~{2} GB, y el equipo tiene {3:N1} GB." -f `
                   $Seeds.Count, [math]::Round($perProc, 2), $needGB, $ramGB)
    Write-Warning ("Con esta RAM caben ~{0} semillas. Los procesos de mas moriran sin error claro." -f $maxSeeds)
    Write-Warning ("Alternativas: -Seeds con {0} valores, o -NClimate 3 (9 escenarios, menos memoria)." -f $maxSeeds)
    Write-Host ""
    $r = Read-Host "Continuar igual? (s/N)"
    if ($r -notmatch '^[sSyY]') { Write-Host "Cancelado."; return }
}

Write-Host ""
Write-Host ("Robust DPS - salida {0}" -f $modelo)
Write-Host ("  semillas     : {0}   (en paralelo)" -f $seedList)
Write-Host ("  evaluaciones : {0}  | poblacion: {1}" -f $Evaluations, $Population)
Write-Host ("  escenarios   : {0}  [{1} climas x 3 demandas]  | lambda={2}" -f $nScen, $NClimate, $Lambda)
Write-Host ("  ETA          : ~{0} h de reloj" -f $eta)
Write-Host ("  salida       : {0}" -f $OutDir)
Write-Host ""

$procs = @()
foreach ($s in $Seeds) {
    $out = Join-Path $OutDir ("pareto_seed{0}.dat" -f $s)
    if (Test-Path $out) {
        Write-Host ("  SKIP seed={0} (ya existe)" -f $s)
        continue
    }
    $log = Join-Path $OutDir ("seed{0}.log" -f $s)
    $arg = @("weap_dps\main_robust_weap.py",
             "--evaluations", $Evaluations,
             "--population",  $Population,
             "--seed",        $s,
             "--n_climate",   $NClimate,
             "--lam",         $Lambda,
             "--output",      $out)
    $p = Start-Process -FilePath $py -ArgumentList $arg -NoNewWindow -PassThru `
                       -RedirectStandardOutput $log -RedirectStandardError ($log + ".err")
    $procs += [PSCustomObject]@{ Seed = $s; ProcId = $p.Id; Log = $log }
    Write-Host ("  LANZADA seed={0}  PID {1}  -> {2}" -f $s, $p.Id, $log)
}

# Registro de lanzamiento. El .log NO sirve para medir tiempo: al reescribir un
# archivo con el mismo nombre Windows conserva la CreationTime original, asi que
# tras un intento fallido el ETA sale calculado desde horas antes.
if ($procs.Count -gt 0) {
    $procs | Select-Object Seed, ProcId, @{n = "StartUtc"; e = { (Get-Date).ToUniversalTime().ToString("o") } } |
        Export-Csv (Join-Path $OutDir "launched.csv") -NoTypeInformation -Encoding ASCII
}

if ($procs.Count -eq 0) {
    Write-Host ""
    Write-Host "Nada que lanzar (todas las semillas ya tienen resultado)."
    return
}

Write-Host ""
Write-Host "Corriendo. Para seguir una semilla:"
Write-Host ("  Get-Content '{0}' -Tail 5 -Wait" -f $procs[0].Log)
Write-Host ""
Write-Host "Para ver el avance de todas (usa check_progress.ps1):"
Write-Host "  .\check_progress.ps1"
Write-Host ""
$procs | Format-Table -AutoSize
