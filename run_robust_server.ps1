# run_robust_server.ps1 — Robust DPS con el modelo iter1_clean_h128.
#
# Lanza N semillas EN PARALELO (1 core por proceso, OMP=1). Varias semillas no
# son un lujo: son lo que permite evaluar CONVERGENCIA (si los frentes de
# distintas semillas coinciden, el resultado es robusto; si no, faltan
# evaluaciones). Es lo que el paper necesita reportar.
#
# Preset por defecto = Preset C ampliado:
#   3 semillas x 6000 evaluaciones x (5 climas x 3 demandas = 15 escenarios)
#   ~17.7 s por evaluacion  ->  ~29 h de reloj (las 3 semillas en paralelo)
#
# Reanudable: si un .dat ya existe, esa semilla se salta.
#
# Uso:
#   .\run_robust_server.ps1                      # preset por defecto
#   .\run_robust_server.ps1 -Evaluations 4000    # ~20 h
#   .\run_robust_server.ps1 -Seeds 42,123        # solo 2 semillas

param(
    [int]   $Evaluations = 6000,
    [int]   $Population  = 100,
    [int]   $NClimate    = 5,       # x3 esquinas de demanda = 15 escenarios
    [double]$Lambda      = 1.0,     # aversion al riesgo: mean + lambda*std
    [int[]] $Seeds       = @(42, 123, 456),
    [string]$OutDir      = "runs_weap\robust_iter1_h128"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# Interprete: venv_DPS si existe, si no el del sistema
$py = if (Test-Path "venv_DPS\Scripts\python.exe") { "venv_DPS\Scripts\python.exe" } else { "python" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# ── Chequeos previos (fallar aqui es barato; a las 20 h no) ──
foreach ($f in "data_weap\best_model.ckpt", "data_weap\X_template.npz",
                "data_weap\scalers_weap.npz", "data_weap\transform_params_weap.npz",
                "data_weap\manifest_inputs.csv") {
    if (-not (Test-Path $f)) { throw "Falta $f  — copia los artefactos del modelo." }
}
if (-not (Test-Path "data_weap\train_subset.zarr")) {
    throw "Falta data_weap\train_subset.zarr — sin el, no se puede armar el ensamble climatico."
}

# NO setear DPS_J4_CAL: eso forzaria un factor ESCALAR y anularia la calibracion
# condicional al numero de acciones (config_weap.j4_calibration_factor).
Remove-Item Env:\DPS_J4_CAL      -ErrorAction SilentlyContinue
Remove-Item Env:\DPS_TRAIN_ZARR  -ErrorAction SilentlyContinue
$env:OMP_NUM_THREADS = "1"       # 1 core por proceso: si no, se pisan entre si
$env:MKL_NUM_THREADS = "1"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONIOENCODING = "utf-8"
$env:DPS_WATERFALL = "0"         # cascada determinista OFF (probada: empeora J4)

$nScen = $NClimate * 3
$eta   = [math]::Round($Evaluations * 1.18 * $nScen / 3600, 1)
Write-Host ""
Write-Host "Robust DPS — modelo iter1_clean_h128"
Write-Host "  semillas     : $($Seeds -join ', ')   (en paralelo)"
Write-Host "  evaluaciones : $Evaluations  | poblacion: $Population"
Write-Host "  escenarios   : $nScen ($NClimate climas x 3 demandas)  | lambda=$Lambda"
Write-Host "  ETA          : ~$eta h de reloj"
Write-Host "  salida       : $OutDir"
Write-Host ""

$procs = @()
foreach ($s in $Seeds) {
    $out = Join-Path $OutDir "pareto_seed$s.dat"
    if (Test-Path $out) { Write-Host "  SKIP seed=$s (ya existe)"; continue }
    $log = Join-Path $OutDir "seed$s.log"
    $args = @("weap_dps\main_robust_weap.py",
              "--evaluations", $Evaluations, "--population", $Population,
              "--seed", $s, "--n_climate", $NClimate, "--lam", $Lambda,
              "--output", $out)
    $p = Start-Process -FilePath $py -ArgumentList $args -NoNewWindow -PassThru `
                       -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    $procs += [PSCustomObject]@{ Seed = $s; PID = $p.Id; Log = $log }
    Write-Host "  LANZADA seed=$s  (PID $($p.Id))  -> $log"
}

if (-not $procs) { Write-Host "`nNada que lanzar."; return }

Write-Host "`nCorriendo. Para seguir el avance:"
Write-Host "  Get-Content '$($procs[0].Log)' -Tail 5 -Wait"
Write-Host "Para ver cuantas evaluaciones lleva cada una:"
Write-Host "  Get-ChildItem '$OutDir\*.log' | ForEach-Object { `$n=(Select-String 'NPV TOTAL' `$_).Count; `"{0}: ~{1} evals`" -f `$_.Name, [int](`$n/$nScen) }"
Write-Host ""
$procs | Format-Table -AutoSize
