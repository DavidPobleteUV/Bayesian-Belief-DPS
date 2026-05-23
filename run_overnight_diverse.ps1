# run_overnight_diverse.ps1
# ---------------------------------------------------------------------------
# Corrida extendida de maxima diversidad: 6 seeds x 20k evals x pop=100.
# Tiempo aprox: 24-30 horas en una sola PC (single worker).
#
# Cada seed guarda en runs_weap/pareto_iter01_seed<S>.dat
# Si se interrumpe, al re-correr saltea los seeds que ya esten completos.
#
# Uso:
#     .\run_overnight_diverse.ps1
# ---------------------------------------------------------------------------

$ErrorActionPreference = "Continue"

$SEEDS = @(42, 123, 456, 789, 1024, 2026)
$EVALS = 20000
$POP   = 100

$LOG_DIR = "runs_weap\logs_overnight"
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
$MASTER_LOG = "$LOG_DIR\master_$(Get-Date -Format yyyyMMdd_HHmm).log"

function LogMsg($msg) {
    $stamped = "$(Get-Date -Format 'HH:mm:ss') $msg"
    Write-Host $stamped -ForegroundColor Cyan
    Add-Content -Path $MASTER_LOG -Value $stamped
}

LogMsg ("=" * 70)
LogMsg "Corrida overnight de maxima diversidad"
LogMsg "Seeds: $($SEEDS -join ', ')"
LogMsg "Evals/seed: $EVALS | Pop/seed: $POP"
LogMsg ("Tiempo estimado total: ~{0:N1} horas" -f ($SEEDS.Count * $EVALS * 0.82 / 3600))
LogMsg ("=" * 70)

$total_start = Get-Date
$completed = 0
$skipped   = 0

foreach ($seed in $SEEDS) {
    $output_path = "runs_weap\pareto_iter01_seed$seed.dat"
    $seed_log    = "$LOG_DIR\seed_$seed.log"

    if (Test-Path $output_path) {
        $size_kb = (Get-Item $output_path).Length / 1024
        if ($size_kb -gt 5) {
            LogMsg "[SKIP] seed=$seed (ya existe: $output_path, ${size_kb} KB)"
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
        LogMsg "[FAIL]  seed=$seed (exit=$exit_code). Continuando con el siguiente..."
    }
}

$total_elapsed = (Get-Date) - $total_start
LogMsg ("=" * 70)
LogMsg ("RESUMEN FINAL - Total: {0:N1} horas" -f $total_elapsed.TotalHours)
LogMsg "Completados: $completed | Saltados (ya existian): $skipped"
LogMsg "Fallidos:    $($SEEDS.Count - $completed - $skipped)"
LogMsg ("=" * 70)
LogMsg ""
LogMsg "Proximo paso: combinar los frentes con weap_dps/combine_pareto_fronts.py"
LogMsg "             o exportar uno: python weap_dps/pareto_to_runids.py --pareto runs_weap/pareto_iter01_seed42.dat --iteration 1 --start_id 1100"
