# check_progress.ps1 - Avance de las semillas del Robust DPS.
# ASCII puro a proposito (ver nota en run_robust_server.ps1).

param(
    [string]$OutDir   = "runs_weap\robust_iter1_h128",
    [int]   $NScen    = 15,     # 5 climas x 3 demandas
    [int]   $Target   = 6000    # evaluaciones pedidas
)

Set-Location $PSScriptRoot
if (-not (Test-Path $OutDir)) { throw "No existe $OutDir" }

$rows = @()
foreach ($log in Get-ChildItem (Join-Path $OutDir "seed*.log") -ErrorAction SilentlyContinue) {
    $n = (Select-String -Path $log.FullName -Pattern "NPV TOTAL" -ErrorAction SilentlyContinue).Count
    $evals = [int]($n / $NScen)
    $seed = $log.BaseName -replace "^seed", ""
    $dat = Join-Path $OutDir ("pareto_seed{0}.dat" -f $seed)
    $el = ((Get-Date) - $log.CreationTime).TotalHours
    $eta = if ($evals -gt 0 -and -not (Test-Path $dat)) {
        [math]::Round(($Target - $evals) * ($el / $evals), 1)
    } else { 0 }
    $rows += [PSCustomObject]@{
        Semilla   = $seed
        Evals     = $evals
        Pct       = "{0:N0}%" -f (100 * $evals / $Target)
        Horas     = [math]::Round($el, 1)
        ETA_h     = $eta
        Terminada = (Test-Path $dat)
    }
}

if ($rows.Count -eq 0) { Write-Host "Sin logs todavia en $OutDir"; return }
$rows | Sort-Object Semilla | Format-Table -AutoSize

$vivos = (Get-Process python -ErrorAction SilentlyContinue | Measure-Object).Count
Write-Host ("procesos python vivos: {0}" -f $vivos)
