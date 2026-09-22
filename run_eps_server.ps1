# run_eps_server.ps1 - Robust DPS con eps-NSGA-II (main_eps_robust_weap.py).
#
# NOTA: ASCII puro a proposito. Windows PowerShell 5.1 lee los .ps1 como ANSI si
# no tienen BOM, y cualquier acento o guion largo rompe el parser en el servidor.
#
# Hermano de run_robust_server.ps1, que se conserva intacto para poder repetir
# la corrida NSGA-II si hace falta. Las diferencias:
#
#   - llama a main_eps_robust_weap.py (archivo de eps-dominancia + reinicios)
#   - el presupuesto por defecto es 10000 y no 4000, para que la curva HV(nfe)
#     alcance a mostrar si 4000 bastaban
#   - salida por defecto a runs_weap\eps
#
# COSTO MEDIDO sobre la corrida iter02 (misma maquina, mismo emulador):
#   58.5 s por evaluacion de 27 escenarios  ->  2.17 s por escenario
#   Las semillas corren en paralelo con 1 hilo cada una, asi que el reloj lo fija
#   el presupuesto POR SEMILLA, no cuantas semillas haya.
#       4000  evaluaciones -> ~65 h
#       10000 evaluaciones -> ~163 h
#
# Reanudable: el .ckpt guarda el archivo eps cada --checkpoint_every
# evaluaciones. Si un .dat final ya existe, esa semilla se salta.
#
# Uso:
#   .\run_eps_server.ps1
#   .\run_eps_server.ps1 -Evaluations 4000
#   .\run_eps_server.ps1 -Seeds 42,123,456 -EpsProgress

param(
    [int]   $Evaluations    = 10000,
    [int]   $Population     = 100,
    [int]   $NClimate       = 5,
    [double]$Lambda         = 1.0,
    [int[]] $Seeds          = @(42, 123, 456, 789, 1010),
    [int]   $RestartWindow  = 10,
    [int]   $MaxPopulation  = 300,
    [double]$EpsScale       = 1.0,
    [switch]$EpsProgress,
    [string]$OutDir         = "runs_weap\eps_iter02"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py = "python"
if (Test-Path "venv_DPS\Scripts\python.exe") { $py = "venv_DPS\Scripts\python.exe" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# --- Chequeos previos (fallar aqui es barato; a las 100 h no) ---
$need = @("data_weap_iter02\best_model.ckpt",
          "data_weap_iter02\X_template.npz",
          "data_weap_iter02\scalers_weap.npz",
          "data_weap_iter02\transform_params_weap.npz",
          "data_weap_iter02\manifest_inputs.csv")
foreach ($f in $need) {
    if (-not (Test-Path $f)) {
        throw "Falta $f . Copia los artefactos del modelo iter02 desde la PC de entrenamiento."
    }
}

$mods = "numpy,torch,pytorch_lightning,zarr,pandas,platypus,rdm_mlp"
$chk = & $py -c @"
import importlib.util as u
print(','.join([m for m in '$mods'.split(',') if u.find_spec(m) is None]))
"@
if ($LASTEXITCODE -ne 0) { throw "No se pudo ejecutar '$py'. Revisa el interprete." }
if ($chk.Trim()) {
    throw ("Faltan modulos: {0}`nInstala con:  pip install {1}" -f `
           $chk.Trim(), ($chk.Trim() -replace ',', ' ' -replace 'pytorch_lightning', 'pytorch-lightning'))
}

# Artefactos de iter02. Si no se apunta DPS_DATA_DIR, config_weap cae a data_weap
# (el modelo de iter01) y la corrida saldria con el emulador equivocado SIN error.
$env:DPS_DATA_DIR = (Resolve-Path "data_weap_iter02").Path

Remove-Item Env:\DPS_J4_CAL     -ErrorAction SilentlyContinue
Remove-Item Env:\DPS_TRAIN_ZARR -ErrorAction SilentlyContinue
$env:OMP_NUM_THREADS = "1"      # 1 core por proceso; si no, se pisan entre si
$env:MKL_NUM_THREADS = "1"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$env:DPS_WATERFALL = "1"        # cascada de despacho por orden de merito
$env:DPS_TORCH_THREADS = "1"
$env:DPS_EPS_SCALE = [string]$EpsScale

$nScen = & $py -c "import sys; sys.path.insert(0,'.'); from weap_dps.config_weap import DPS_N_SOW; print(DPS_N_SOW)"
if ($LASTEXITCODE -ne 0 -or -not $nScen) { $nScen = $NClimate * 3 }
$nScen = [int]$nScen

# 2.17 s por rollout de escenario, MEDIDO en la corrida iter02 (65.0 h / 4000
# evaluaciones / 27 escenarios), con 1 hilo de torch y 5 semillas EN PARALELO.
#
# La competencia entre semillas no es despreciable y el ETA debe incluirla: el
# mismo emulador en la misma maquina da 1.55 s/escenario con UN SOLO proceso
# (41.9 s/evaluacion) y 2.17 con cinco (58.5 s/evaluacion), o sea +40% por
# semilla. Un ETA medido con un proceso solo subestimaria el reloj en ese 40%.
# En otra maquina hay que re-medirlo con weap_dps/benchmark_eval.py --par N
$eta = [math]::Round($Evaluations * 2.17 * $nScen / 3600, 1)
$seedList = $Seeds -join ", "

# --- Aviso de RAM ---
$ramGB   = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB
# Nucleos FISICOS, no logicos. El hyperthreading no agrega capacidad para esta
# carga: dos hilos del mismo nucleo compiten por las mismas unidades de calculo.
# Medido el 22-09-2026 en el i7-8700 (6 fisicos / 12 logicos): con 5 semillas del
# DPS mas 5 procesos de otro trabajo -10 procesos sobre 6 nucleos- el segundo
# trabajo paso de 101 a 348 s por unidad, 3.4x mas lento. Contando logicos, el
# aviso de abajo no se habria disparado.
# NumberOfCores viene por procesador fisico: se suman por si hay mas de uno.
$cores   = (Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum
$logicos = (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
# 0.55 GB por proceso MEDIDO (working set en regimen, 27 escenarios, modelo
# iter02 cargado). El 0.9 + 0.005*nScen que usaba run_robust_server.ps1 era una
# estimacion declarada como no medida y sobrestimaba ~2x, de modo que en una
# maquina chica desaconsejaba semillas que si caben.
# Se reserva 2.5 GB para el SO y se deja 20% de margen sobre lo medido.
$perProc  = 0.66
$maxSeeds = [math]::Max(1, [math]::Floor(($ramGB - 2.5) / $perProc))
if ($Seeds.Count -gt $maxSeeds) {
    Write-Warning ("RAM insuficiente: {0} semillas x ~{1} GB, y el equipo tiene {2:N1} GB." -f `
                   $Seeds.Count, [math]::Round($perProc, 2), $ramGB)
    Write-Warning ("Con esta RAM caben ~{0} semillas. Los procesos de mas moriran sin error claro." -f $maxSeeds)
    $r = Read-Host "Continuar igual? (s/N)"
    if ($r -notmatch '^[sSyY]') { Write-Host "Cancelado."; return }
}
# En esta carga el cuello de botella es CPU, no RAM: cada semilla ocupa un nucleo
# entero (OMP=1) y solo 0.66 GB. Conviene dejar al menos uno al SO y a cualquier
# otro trabajo que corra en paralelo.
if ($Seeds.Count -gt ($cores - 1)) {
    Write-Warning ("{0} semillas sobre {1} nucleos FISICOS ({2} logicos): se pelearan por CPU " -f `
                   $Seeds.Count, $cores, $logicos)
    Write-Warning ("y el ETA se alarga mas alla del +40% ya incluido. Conviene no pasar de {0}." -f ($cores - 1))
}

$cont = if ($EpsProgress) { "eps-progreso (estilo Borg)" } else { "temporal adaptativa" }
Write-Host ""
Write-Host ("Robust DPS con eps-NSGA-II - salida {0}" -f $OutDir)
Write-Host ("  semillas      : {0}   (en paralelo)" -f $seedList)
Write-Host ("  evaluaciones  : {0}  | poblacion inicial: {1}  | tope: {2}" -f $Evaluations, $Population, $MaxPopulation)
Write-Host ("  escenarios    : {0}  | lambda={1}  | escala de eps={2}" -f $nScen, $Lambda, $EpsScale)
Write-Host ("  continuacion  : {0}, ventana {1} generaciones" -f $cont, $RestartWindow)
Write-Host ("  equipo        : {0} nucleos fisicos ({1} logicos), {2:N1} GB" -f $cores, $logicos, $ramGB)
Write-Host ("  ETA           : ~{0} h de reloj" -f $eta)
Write-Host ""

$procs = @()
foreach ($s in $Seeds) {
    $out = Join-Path $OutDir ("pareto_seed{0}.dat" -f $s)
    if (Test-Path $out) {
        Write-Host ("  SKIP seed={0} (ya existe)" -f $s)
        continue
    }
    $log = Join-Path $OutDir ("seed{0}.log" -f $s)
    $arg = @("weap_dps\main_eps_robust_weap.py",
             "--evaluations",    $Evaluations,
             "--population",     $Population,
             "--seed",           $s,
             "--n_climate",      $NClimate,
             "--lam",            $Lambda,
             "--restart_window", $RestartWindow,
             "--max_population", $MaxPopulation,
             "--output",         $out)
    if ($EpsProgress) { $arg += "--eps_progress" }
    $p = Start-Process -FilePath $py -ArgumentList $arg -NoNewWindow -PassThru `
                       -RedirectStandardOutput $log -RedirectStandardError ($log + ".err")
    $procs += [PSCustomObject]@{ Seed = $s; ProcId = $p.Id; Log = $log }
    Write-Host ("  LANZADA seed={0}  PID {1}  -> {2}" -f $s, $p.Id, $log)
}

# Registro de lanzamiento. El .log NO sirve para medir tiempo: al reescribir un
# archivo con el mismo nombre Windows conserva la CreationTime original.
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
Write-Host "Corriendo. El log trae el HV en cada checkpoint:"
Write-Host ("  Get-Content '{0}' -Tail 5 -Wait" -f $procs[0].Log)
Write-Host ""
Write-Host "Cuando haya resultados, la comparacion contra NSGA-II:"
Write-Host ("  python weap_dps\comparar_algoritmos.py --eps '{0}\pareto_seed*.dat'" -f $OutDir)
Write-Host ""
$procs | Format-Table -AutoSize
