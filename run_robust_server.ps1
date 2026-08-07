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

# NO setear DPS_J4_CAL: forzaria un factor ESCALAR y anularia la calibracion
# condicional al numero de acciones (config_weap.j4_calibration_factor).
Remove-Item Env:\DPS_J4_CAL     -ErrorAction SilentlyContinue
Remove-Item Env:\DPS_TRAIN_ZARR -ErrorAction SilentlyContinue
$env:OMP_NUM_THREADS = "1"      # 1 core por proceso; si no, se pisan entre si
$env:MKL_NUM_THREADS = "1"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONIOENCODING = "utf-8"
$env:DPS_WATERFALL = "0"        # cascada determinista OFF (probada: empeora J4)

$nScen = $NClimate * 3
$eta   = [math]::Round($Evaluations * 1.18 * $nScen / 3600, 1)
$seedList = $Seeds -join ", "

Write-Host ""
Write-Host "Robust DPS - modelo iter1_clean_h128"
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
