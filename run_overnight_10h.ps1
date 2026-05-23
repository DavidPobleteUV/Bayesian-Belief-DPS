# run_overnight_10h.ps1
# ---------------------------------------------------------------------------
# Corrida overnight de ~10 horas: 5 seeds x 8000 evals x pop=80
# Tiempo estimado: ~9 horas (~0.82 seg/eval x 40000 evals)
#
# Resultados van a runs_weap/pareto_iter01_seed<S>_overnight10h.dat
# El script salta seeds ya completos si lo reinicias.
# ---------------------------------------------------------------------------

$ErrorActionPreference = "Continue"

$SEEDS = @(42, 123, 456, 789, 1024)
$EVALS = 8000
$POP   = 80

$LOG_DIR = "runs_weap\logs_overnight10h"
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
$MASTER_LOG = "$LOG_DIR\master_$(Get-Date -Format yyyyMMdd_HHmm).log"

function LogMsg($msg) {
    $stamped = "$(Get-Date -Format 'HH:mm:ss') $msg"
    Write-Host $stamped -ForegroundColor Cyan
    Add-Content -Path $MASTER_LOG -Value $stamped
}

LogMsg ("=" * 70)
LogMsg "Corrida overnight 10h con J4 anualizado + J3 NPV + clipping a 0"
LogMsg "Seeds: $($SEEDS -join ', ')"
LogMsg "Evals/seed: $EVALS | Pop/seed: $POP"
LogMsg ("Tiempo estimado: ~{0:N1} horas" -f ($SEEDS.Count * $EVALS * 0.82 / 3600))
LogMsg ("=" * 70)

$total_start = Get-Date
$completed = 0
$skipped   = 0

foreach ($seed in $SEEDS) {
    $output_path = "runs_weap\pareto_iter01_seed${seed}_overnight10h.dat"
    $seed_log    = "$LOG_DIR\seed_${seed}.log"

    if (Test-Path $output_path) {
        $size_kb = (Get-Item $output_path).Length / 1024
        if ($size_kb -gt 5) {
            LogMsg "[SKIP] seed=$seed (ya existe)"
            $skipped++
            continue
        }
    }

    LogMsg ("-" * 70)
    LogMsg "[START] seed=$seed -> $output_path"
    $seed_start = Get-Date

    python weap_dps/main_par_weap.py `
        --algorithm NSGAII `
        --evaluations $EVALS `
        --population $POP `
        --workers 1 `
        --seed $seed `
        --output $output_path 2>&1 | Tee-Object -FilePath $seed_log

    $exit_code = $LASTEXITCODE
    $seed_elapsed = (Get-Date) - $seed_start

    if ($exit_code -eq 0 -and (Test-Path $output_path)) {
        LogMsg ("[DONE]  seed=$seed in {0:N1} min" -f $seed_elapsed.TotalMinutes)
        $completed++
    } else {
        LogMsg "[FAIL]  seed=$seed (exit=$exit_code)"
    }
}

$total_elapsed = (Get-Date) - $total_start
LogMsg ("=" * 70)
LogMsg ("RESUMEN FINAL - Total: {0:N1} horas" -f $total_elapsed.TotalHours)
LogMsg "Completados: $completed | Saltados: $skipped"
LogMsg ("=" * 70)
LogMsg ""
LogMsg "Pasos siguientes:"
LogMsg "  python weap_dps/combine_pareto_fronts.py --glob 'runs_weap/pareto_iter01_seed*_overnight10h.dat' --output runs_weap/pareto_iter01_combined_overnight10h.dat"
LogMsg "  python weap_dps/plot_pareto.py --inputs runs_weap/pareto_iter01_combined_overnight10h.dat --output_dir runs_weap/plots_overnight10h"
LogMsg "  python weap_dps/plot_timeseries_from_pareto.py --pareto runs_weap/pareto_iter01_combined_overnight10h.dat --output_dir runs_weap/timeseries_overnight10h"
