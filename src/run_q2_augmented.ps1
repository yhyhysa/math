param(
    [ValidateSet('smoke', 'full')]
    [string]$Mode = 'smoke',
    [ValidateSet('T', 'TA', 'TV', 'TAV')]
    [string]$Modalities = 'TAV',
    [ValidateSet('concat', 'coverage_gate')]
    [string]$Fusion = 'concat',
    [switch]$Clean,
    [switch]$EffectiveMask,
    [ValidatePattern('^[A-Za-z0-9_-]+$')]
    [string]$Tag = '20260924'
)

$ErrorActionPreference = 'Stop'
if ($Clean -and $EffectiveMask) { throw '-EffectiveMask requires augmented training (omit -Clean).' }
$engine = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $engine
$python = Join-Path $engine '..\训练demo\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment is missing: $python"
}

$seeds = if ($Mode -eq 'smoke') { @(20260923) } else { @(20260923, 20260924, 20260925) }
foreach ($seed in $seeds) {
    $modalTag = if ($Modalities -eq 'TAV') { '' } else { "${Modalities}_" }
    $fusionTag = if ($Fusion -eq 'concat') { '' } else { 'gate_' }
    $effectiveTag = if ($EffectiveMask) { 'eff_' } else { '' }
    $kind = if ($Clean) { 'clean' } else { 'aug' }
    $runName = if ($Mode -eq 'smoke') { "q2_${kind}_smoke_${modalTag}${fusionTag}${effectiveTag}$Tag" } else { "q2_${kind}_${modalTag}${fusionTag}${effectiveTag}${Tag}_seed_$seed" }
    $out = Join-Path $engine "results\$runName"
    if (Test-Path -LiteralPath $out) {
        throw "Result folder already exists. Change -Tag to keep both runs: $out"
    }
    New-Item -ItemType Directory -Path $out | Out-Null
    $log = Join-Path $out 'terminal.log'
    $status = Join-Path $out 'exit_code.txt'
    'RUNNING' | Set-Content -LiteralPath $status -Encoding utf8
    Start-Transcript -LiteralPath $log | Out-Null
    $exitCode = 0
    try {
        Write-Host "Q2 step 2: $runName"
        Write-Host 'Checking raw inputs before training...'
        & $python -m src.tools.verify_raw_integrity
        if ($LASTEXITCODE -ne 0) { throw "Pre-run raw integrity check failed: $LASTEXITCODE" }

        Write-Host 'Training. Each epoch prints train losses and clean/masked valid metrics.'
        $trainArgs = @('-m', 'src.q2_train_baseline', '--modalities', $Modalities,
                       '--fusion', $Fusion, '--seed', [string]$seed, '--run-name', $runName)
        if (-not $Clean) { $trainArgs += '--augment' }
        if ($EffectiveMask) { $trainArgs += '--effective-mask' }
        if ($Mode -eq 'smoke') {
            $trainArgs += '--smoke'
        } else {
            $trainArgs += @('--epochs', '30', '--patience', '30')
        }
        & $python @trainArgs
        if ($LASTEXITCODE -ne 0) { throw "Training failed: $LASTEXITCODE" }

        Write-Host 'Checking raw inputs after training...'
        & $python -m src.tools.verify_raw_integrity
        if ($LASTEXITCODE -ne 0) { throw "Post-run raw integrity check failed: $LASTEXITCODE" }
        Write-Host "Completed: $out"
    } catch {
        $exitCode = 1
        Write-Host "FAILED: $_" -ForegroundColor Red
    } finally {
        $exitCode | Set-Content -LiteralPath $status -Encoding utf8
        Stop-Transcript | Out-Null
    }
    if ($exitCode -ne 0) { exit $exitCode }
}
Write-Host 'All requested Q2 augmented runs completed.'
