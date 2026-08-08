# check_progress.ps1 - Avance de las semillas del Robust DPS.
# ASCII puro a proposito (ver nota en run_robust_server.ps1).

param(
    [string]$OutDir   = "runs_weap\robust_iter1_h128",
    [int]   $NScen    = 15,     # 5 climas x 3 demandas
    [int]   $Target   = 4000    # evaluaciones pedidas
)

Set-Location $PSScriptRoot
if (-not (Test-Path $OutDir)) { throw "No existe $OutDir" }

# --- Cuando empezo esto de verdad ---
# La CreationTime del .log NO sirve: al reescribir un archivo con el mismo
# nombre Windows conserva la fecha original, asi que despues de un intento
# fallido el ETA sale medido desde horas antes. Orden de preferencia:
#   1. launched.csv que escribe run_robust_server.ps1 (exacto, por semilla)
#   2. el arranque mas antiguo entre los procesos python vivos (este lote)
#   3. CreationTime del log (ultimo recurso, puede mentir)
$startBySeed = @{}
$csv = Join-Path $OutDir "launched.csv"
if (Test-Path $csv) {
    foreach ($r in (Import-Csv $csv)) {
        $startBySeed[[string]$r.Seed] = [datetime]::Parse($r.StartUtc).ToLocalTime()
    }
}
$procs = @(Get-Process python -ErrorAction SilentlyContinue)
$batchStart = $null
if ($procs.Count -gt 0) {
    $batchStart = ($procs | Sort-Object StartTime | Select-Object -First 1).StartTime
}

$rows = @()
foreach ($log in Get-ChildItem (Join-Path $OutDir "seed*.log") -ErrorAction SilentlyContinue) {
    $seed = $log.BaseName -replace "^seed", ""
    $n = (Select-String -Path $log.FullName -Pattern "NPV TOTAL" -ErrorAction SilentlyContinue).Count
    $evals = [int]($n / $NScen)
    $dat = Join-Path $OutDir ("pareto_seed{0}.dat" -f $seed)
    $done = Test-Path $dat

    $t0 = if ($startBySeed.ContainsKey($seed)) { $startBySeed[$seed] }
          elseif ($batchStart) { $batchStart }
          else { $log.CreationTime }
    $el = ((Get-Date) - $t0).TotalHours

    # Un log sin actividad reciente y sin .dat es una semilla muerta, no una
    # semilla lenta. Se distingue por el ultimo escrito, no por el conteo.
    $idle = ((Get-Date) - $log.LastWriteTime).TotalMinutes
    $estado = if ($done) { "lista" }
              elseif ($idle -gt 30) { "MUERTA?" }
              else { "corriendo" }

    $eta = if ($evals -gt 0 -and -not $done) {
        [math]::Round(($Target - $evals) * ($el / $evals), 1)
    } else { 0 }

    $rows += [PSCustomObject]@{
        Semilla  = $seed
        Evals    = $evals
        Pct      = "{0:N0}%" -f (100 * $evals / $Target)
        Horas    = [math]::Round($el, 2)
        ETA_h    = $eta
        Inactiva = [math]::Round($idle, 0)
        Estado   = $estado
    }
}

if ($rows.Count -eq 0) { Write-Host "Sin logs todavia en $OutDir"; return }
$rows | Sort-Object { [int]$_.Semilla } | Format-Table -AutoSize

Write-Host ("procesos python vivos: {0}" -f $procs.Count)
if (-not (Test-Path $csv)) {
    Write-Host "(sin launched.csv: los tiempos salen del arranque de los procesos)"
}
$muertas = @($rows | Where-Object { $_.Estado -eq "MUERTA?" })
if ($muertas.Count -gt 0) {
    Write-Host ""
    Write-Host ("Logs sin actividad hace >30 min: {0}" -f (($muertas.Semilla) -join ", "))
    Write-Host "Pueden ser de un intento anterior. Revisa el .err correspondiente."
}
