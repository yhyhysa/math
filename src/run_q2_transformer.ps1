param(
    [ValidateSet('smoke', 'full', 'grid')]
    [string]$Mode = 'smoke',
    [ValidateSet(1, 2)]
    [int]$Layers = 1,
    [switch]$Augment,
    [ValidateSet('early', 'separate')]
    [string]$Architecture = 'early',
    [ValidateSet('concat', 'reliability_gate')]
    [string]$Fusion = 'concat',
    [switch]$EffectiveMask,
    [switch]$MaskEmpty,
    [ValidateRange(0.0000001, 1.0)]
    [double]$LearningRate = 0.001,
    [ValidateRange(0.0, 10.0)]
    [double]$ConsistencyWeight = 0.0,
    [ValidateRange(0.0, 10.0)]
    [double]$AuxiliaryWeight = 0.0,
    [ValidateSet('single', 'multi')]
    [string]$MaskPattern = 'single',
    [ValidatePattern('^[A-Za-z0-9_-]+$')]
    [string]$Tag = '20260924'
)

$ErrorActionPreference = 'Stop'
if ($Architecture -eq 'early' -and $Fusion -ne 'concat') { throw '-Fusion is for the separate architecture.' }
if ($Architecture -eq 'separate' -and $MaskEmpty) { throw 'Separate encoders always mask empty positions; omit -MaskEmpty.' }
if ($EffectiveMask -and (-not $Augment -or $Mode -eq 'grid')) { throw '-EffectiveMask needs -Augment and cannot use grid mode.' }
if ($ConsistencyWeight -gt 0 -and -not $Augment) { throw '-ConsistencyWeight needs -Augment.' }
if ($AuxiliaryWeight -gt 0 -and $Architecture -ne 'separate') { throw '-AuxiliaryWeight needs -Architecture separate.' }
if ($MaskPattern -eq 'multi' -and (-not $Augment -or -not $EffectiveMask)) { throw '-MaskPattern multi needs -Augment -EffectiveMask.' }
if ($Architecture -eq 'separate' -and $Mode -eq 'grid') { throw 'Use -Mode full for one separate-encoder configuration.' }
$engine = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $engine
$python = Join-Path $engine '..\训练demo\.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment is missing: $python"
}

if ($Mode -eq 'grid') {
    $configs = @(
        @{ layers = 1; augment = $false },
        @{ layers = 1; augment = $true },
        @{ layers = 2; augment = $false },
        @{ layers = 2; augment = $true }
    )
} else {
    $configs = @(@{ layers = $Layers; augment = [bool]$Augment })
}
$seeds = if ($Mode -eq 'smoke') { @(20260923) } else { @(20260923, 20260924, 20260925) }

foreach ($config in $configs) {
    foreach ($seed in $seeds) {
        $kind = if ($config.augment) { 'aug' } else { 'clean' }
        if ($Architecture -eq 'early') {
            $runName = if ($Mode -eq 'smoke') {
                "q2_b_smoke_${kind}_$($config.layers)layer_$Tag"
            } else {
                "q2_b_${kind}_$($config.layers)layer_${Tag}_seed_$seed"
            }
        } else {
            $fusionName = if ($Fusion -eq 'concat') { 'concat' } else { 'gate' }
            $maskName = if ($EffectiveMask) { 'eff_' } else { '' }
            $runName = if ($Mode -eq 'smoke') {
                "q2_c_smoke_${kind}_$($config.layers)layer_${fusionName}_${maskName}$Tag"
            } else {
                "q2_c_${kind}_$($config.layers)layer_${fusionName}_${maskName}${Tag}_seed_$seed"
            }
        }
        $out = Join-Path $engine "results\$runName"
        $checkpoint = Join-Path $engine "checkpoints\$runName"
        if ((Test-Path -LiteralPath $out) -or (Test-Path -LiteralPath $checkpoint)) {
            throw "Result already exists. Change -Tag to keep both runs: $runName"
        }
        New-Item -ItemType Directory -Path $out | Out-Null
        $log = Join-Path $out 'terminal.log'
        $status = Join-Path $out 'exit_code.txt'
        'RUNNING' | Set-Content -LiteralPath $status -Encoding utf8
        Start-Transcript -LiteralPath $log | Out-Null
        $exitCode = 0
        try {
            Write-Host "Q2 temporal model: $runName"
            Write-Host 'Checking raw inputs before training...'
            & $python -m src.tools.verify_raw_integrity
            if ($LASTEXITCODE -ne 0) { throw "Pre-run raw integrity check failed: $LASTEXITCODE" }

            $arguments = @('-m', 'src.q2_train_transformer', '--layers', [string]$config.layers,
                           '--architecture', $Architecture, '--fusion', $Fusion,
                           '--learning-rate', $LearningRate.ToString('G17', [System.Globalization.CultureInfo]::InvariantCulture),
                           '--consistency-weight', $ConsistencyWeight.ToString('G17', [System.Globalization.CultureInfo]::InvariantCulture),
                           '--auxiliary-weight', $AuxiliaryWeight.ToString('G17', [System.Globalization.CultureInfo]::InvariantCulture),
                           '--mask-pattern', $MaskPattern,
                           '--seed', [string]$seed, '--run-name', $runName)
            if ($config.augment) { $arguments += '--augment' }
            if ($EffectiveMask) { $arguments += '--effective-mask' }
            if ($MaskEmpty) { $arguments += '--mask-empty' }
            if ($Mode -eq 'smoke') {
                $arguments += '--smoke'
            } else {
                $arguments += @('--epochs', '30', '--patience', '30')
            }
            Write-Host 'Training. Each epoch prints losses and clean/masked valid metrics.'
            & $python @arguments
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
}
Write-Host 'All requested Q2 model-B runs completed.'
if ($Mode -eq 'grid') {
    Write-Host 'Comparing all complete model-A and model-B runs...'
    & $python -m src.q2_compare_step3 --tag $Tag
    if ($LASTEXITCODE -ne 0) { throw "Model comparison failed: $LASTEXITCODE" }
}
