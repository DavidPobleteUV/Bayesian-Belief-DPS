# Cola de evaluación COMPLETA: espera a que termine el entrenamiento, luego para
# CADA modelo (v3 cascada, v2 baseline) en orden:
#   1) selecciona el mejor checkpoint por KGE de rollout recursivo
#   2) corre la validación: visualize_results (+ metrics CSVs) → audit por tipo
#   3) genera predicciones (save_zarr) → plot costo/transmisión/palto obs-vs-sim
# Detached e independiente del harness.
Set-Location "C:\Users\David\Documents\GitHub_DPL\WEAP_HydroMLP_RecursiveGW"

# --- 0) Esperar a que el entrenamiento termine (sin python > 1.5 GB x2) ---
$idle = 0
while ($idle -lt 2) {
    $big = Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.WS -gt 1.5GB }
    if ($big) { $idle = 0 } else { $idle++ }
    Start-Sleep -Seconds 60
}
Start-Sleep -Seconds 20

$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONUTF8 = "1"
$env:OMP_NUM_THREADS = "4"
$py = ".\venv_HydroMLP\Scripts\python.exe"
$zarr = "data/weap_weekly.zarr"

# Modelos: etiqueta → carpeta de checkpoints; iter = bloque de run-ids del Pareto
# (iteration 1 → ids 2000+, iteration 2 → ids 2100+) para no colisionar.
$models = @(
    @{ label = "v3"; dir = "runs/iter02_cascade";  iter = 1 },
    @{ label = "v2"; dir = "runs/iter02_baseline"; iter = 2 }
)

foreach ($m in $models) {
    $label = $m.label; $dir = $m.dir
    $outdir = "results/eval_$label"
    $bestfile = "runs/best_$label.txt"

    # 1) Selección por KGE de rollout
    & $py compare_storage_kge.py --ckpt_dir $dir --label $label --out_best $bestfile `
        *> "runs/eval_${label}_select.log"
    if (-not (Test-Path $bestfile)) { continue }
    $best = (Get-Content $bestfile -Raw).Trim()

    # 2) Validación: series + métricas por variable, luego audit por tipo
    & $py src/scripts/evaluation/visualize_results.py --checkpoint $best --zarr $zarr `
        --output_dir $outdir --max_runs 4 --plot_individual_runs --plot_runs_n 4 `
        *> "runs/eval_${label}_visualize.log"
    & $py src/scripts/evaluation/audit_bad_metrics.py --zarr $zarr `
        --metrics_dir $outdir --warmup 104 `
        *> "runs/eval_${label}_audit.log"

    # 3) Predicciones + validación costo/transmisión/palto (obs vs sim)
    & $py src/scripts/evaluation/evaluate_recursive.py --checkpoint $best --zarr $zarr `
        --n_runs 4 --seed 42 --save_zarr "results/preds_$label.zarr" `
        *> "runs/eval_${label}_preds.log"
    & $py src/scripts/evaluation/plot_cost_avocado_validation.py `
        --obs_zarr $zarr --pred_zarr "results/preds_$label.zarr" `
        --output_dir "$outdir/cost_avocado" `
        *> "runs/eval_${label}_costval.log"
}

"EVAL DONE $(Get-Date -Format o)" | Out-File "runs/eval_queue_done.txt"

# ============================================================================
# 4) DPS por modelo: cada modelo (v2, v3) corre la optimización con SU mejor
#    checkpoint y guarda el frente con nombre del modelo (trazabilidad).
# ============================================================================
$DPS = "C:\Users\David\Documents\GitHub_DPL\Bayesian-Belief-DPS"
$dpsPy = "$DPS\venv_DPS\Scripts\python.exe"
$dataw = "$DPS\data_weap"

foreach ($m in $models) {
    $label = $m.label
    $bestfile = "runs/best_$label.txt"
    if (-not (Test-Path $bestfile)) { continue }
    $best = (Get-Content $bestfile -Raw).Trim()

    # Fijar el checkpoint del modelo como el activo del DPS (force, no copy_if_newer)
    Copy-Item -Force $best "$dataw\best_model.ckpt"
    Copy-Item -Force $best "$dataw\best_model_$label.ckpt"   # registro por modelo

    # Optimización DPS con nombre del modelo en la salida
    Set-Location $DPS
    $env:OMP_NUM_THREADS = "2"
    & $dpsPy weap_dps/main_par_weap.py --algorithm NSGAII --population 100 `
        --evaluations 6000 --seed 42 --workers 1 `
        --output "runs_weap/pareto_${label}_round1_seed42.dat" `
        *> "runs_weap/dps_${label}_round1_seed42.log"

    # Pareto → casos WEAP (bloque de run-ids por modelo, sin colisión)
    if (Test-Path "runs_weap/pareto_${label}_round1_seed42.dat") {
        & $dpsPy weap_dps/pareto_to_runids.py `
            --pareto "runs_weap/pareto_${label}_round1_seed42.dat" `
            --iteration $($m.iter) `
            --output_dir "data_weap/exports/pareto_${label}_round1" `
            *> "runs_weap/runids_${label}_round1.log"
    }
    Set-Location "C:\Users\David\Documents\GitHub_DPL\WEAP_HydroMLP_RecursiveGW"
}

"ALL DONE $(Get-Date -Format o)" | Out-File "runs/eval_queue_done.txt"
