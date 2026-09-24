$ErrorActionPreference = 'Stop'
$engine = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $engine
$runner = Join-Path $PSScriptRoot 'run_q2_transformer.ps1'
$venvDir = ([char]0x8BAD) + ([char]0x7EC3) + 'demo'
$python = Join-Path $engine ("..\$venvDir\.venv\Scripts\python.exe")

foreach ($seed in @(20260923, 20260924, 20260925)) {
    $status = Join-Path $engine "results\q2_c_aug_1layer_gate_eff_tonight_gate_20260924_seed_$seed\exit_code.txt"
    if (-not (Test-Path -LiteralPath $status) -or (Get-Content -LiteralPath $status -Raw).Trim() -ne '0') {
        throw "Finish the same-learning-rate gate control first: $status"
    }
}

Write-Host 'Self-checking the new Q2 branches...'
& $python -m src.q2_train_transformer --selfcheck
if ($LASTEXITCODE -ne 0) { throw 'Model self-check failed' }
& $python -m src.q2_missing_patterns
if ($LASTEXITCODE -ne 0) { throw 'Multi-block self-check failed' }

$common = @{ Mode='smoke'; Architecture='separate'; Layers=1; Augment=$true; EffectiveMask=$true; LearningRate=0.0003 }
& $runner @common -Fusion concat -AuxiliaryWeight 0.1 -Tag tonight_uni_smoke_20260924
if ($LASTEXITCODE -ne 0) { throw 'Unimodal-head smoke failed' }
& $runner @common -Fusion reliability_gate -AuxiliaryWeight 0.1 -Tag tonight_gate_uni_smoke_20260924
if ($LASTEXITCODE -ne 0) { throw 'Gated unimodal-head smoke failed' }
& $runner @common -Fusion concat -MaskPattern multi -Tag tonight_multi_smoke_20260924
if ($LASTEXITCODE -ne 0) { throw 'Multi-block smoke failed' }

Write-Host 'Running the four remaining three-seed, 30-epoch comparisons...'
$common.Mode = 'full'
$common.LearningRate = 0.0001
& $runner @common -Fusion concat -Tag tonight_lr1e4_20260924
if ($LASTEXITCODE -ne 0) { throw 'Lower learning-rate comparison failed' }
$common.LearningRate = 0.0003
& $runner @common -Fusion concat -AuxiliaryWeight 0.1 -Tag tonight_uni_20260924
if ($LASTEXITCODE -ne 0) { throw 'Unimodal-head comparison failed' }
& $runner @common -Fusion reliability_gate -AuxiliaryWeight 0.1 -Tag tonight_gate_uni_20260924
if ($LASTEXITCODE -ne 0) { throw 'Gated unimodal-head comparison failed' }
& $runner @common -Fusion concat -MaskPattern multi -Tag tonight_multi_20260924
if ($LASTEXITCODE -ne 0) { throw 'Multi-block comparison failed' }

& $python -m src.q2_compare_tonight
if ($LASTEXITCODE -ne 0) { throw 'Final validation comparison failed' }
& $python -m src.tools.verify_raw_integrity
if ($LASTEXITCODE -ne 0) { throw 'Pre-diagnostic raw integrity check failed' }
& $python -m src.q2_neutral_diagnostic
if ($LASTEXITCODE -ne 0) { throw 'Neutral-class diagnostic failed' }
& $python -m src.tools.verify_raw_integrity
if ($LASTEXITCODE -ne 0) { throw 'Post-diagnostic raw integrity check failed' }
Write-Host 'Tonight Q2 candidate training and comparison completed.'
