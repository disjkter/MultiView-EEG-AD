param(
    [string]$PythonExe = "",
    [string]$ProjectRoot = "",
    [int[]]$Seeds = @(2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035),
    [switch]$SkipObjectivePretraining,
    [switch]$SkipObjectiveFineTuning,
    [switch]$SkipBranchAblation,
    [switch]$SkipAggregate
)

$ErrorActionPreference = "Stop"
if (-not $ProjectRoot) { $ProjectRoot = Split-Path -Parent $PSScriptRoot }
if (-not $PythonExe) {
    $PythonExe = if ($env:BIOFOUNDATION_PYTHON) { $env:BIOFOUNDATION_PYTHON } else { "python" }
}
$resultRoot = "outputs\jbhi_revision"
$splitRoot = "$resultRoot\fixed_splits"
$objectiveRoot = "$resultRoot\objective_ablation"
$branchRoot = "$resultRoot\branch_ablation"
$logPath = Join-Path $ProjectRoot "$resultRoot\server_progress.log"

Set-Location -LiteralPath $ProjectRoot
New-Item -ItemType Directory -Force -Path $resultRoot | Out-Null

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -LiteralPath $logPath -Value $line -Encoding UTF8
    Write-Host $line
}

function Run-Python {
    param([string]$Label, [string[]]$Arguments, [string]$DoneFile)
    $donePath = Join-Path $ProjectRoot $DoneFile
    if (Test-Path -LiteralPath $donePath) {
        Write-Log "SKIP $Label ($DoneFile exists)"
        return
    }
    Write-Log "START $Label"
    & $PythonExe @Arguments 2>&1 | Tee-Object -FilePath $logPath -Append
    if ($LASTEXITCODE -ne 0) { throw "$Label failed with exit code $LASTEXITCODE" }
    if (-not (Test-Path -LiteralPath $donePath)) { throw "$Label did not create $DoneFile" }
    Write-Log "DONE $Label"
}

function Run-PythonResumableBatch {
    param([string]$Label, [string[]]$Arguments)
    Write-Log "START/RESUME $Label"
    & $PythonExe @Arguments 2>&1 | Tee-Object -FilePath $logPath -Append
    if ($LASTEXITCODE -ne 0) { throw "$Label failed with exit code $LASTEXITCODE" }
    Write-Log "DONE $Label"
}

function Require-File {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { throw "Missing required input: $Path" }
}

$datasets = @(
    "CAUEEG_processed_19ch_ad_vs_hc_1s_lead",
    "ds004504_processed_19ch_1s_lead"
)
$sourceDirs = @(
    "BrainLat19_pretrain_19ch_1s_lead",
    "auditory40hz_processed_19ch_1s_lead",
    "ADSZ_processed_19ch_1s_lead"
)
foreach ($dataset in $datasets) {
    foreach ($fileName in @("x_19ch.npy", "y_19ch.npy", "sub_ids_19ch.npy")) {
        Require-File "$dataset\$fileName"
    }
}
foreach ($sourceDir in $sourceDirs) { Require-File "$sourceDir\x_19ch.npy" }

$seedText = @($Seeds | ForEach-Object { "$_" })
Run-PythonResumableBatch "create fixed participant splits" (@(
    "scripts\create_subject_splits.py", "--datasets", $datasets[0], $datasets[1], "--seeds"
) + $seedText + @(
    "--train-ratio", "0.7", "--val-ratio", "0.1", "--output-dir", $splitRoot
))

$conditions = @(
    [pscustomobject]@{ Key = "masked_signal_only"; Signal = "1"; Freq = "0";   Branch = "0";   Crop = "0" },
    [pscustomobject]@{ Key = "full";               Signal = "1"; Freq = "0.5"; Branch = "0.1"; Crop = "0.1" },
    [pscustomobject]@{ Key = "minus_signal";       Signal = "0"; Freq = "0.5"; Branch = "0.1"; Crop = "0.1" },
    [pscustomobject]@{ Key = "minus_spectrum";     Signal = "1"; Freq = "0";   Branch = "0.1"; Crop = "0.1" },
    [pscustomobject]@{ Key = "minus_branch";       Signal = "1"; Freq = "0.5"; Branch = "0";   Crop = "0.1" },
    [pscustomobject]@{ Key = "minus_crop";         Signal = "1"; Freq = "0.5"; Branch = "0.1"; Crop = "0" }
)

if (-not $SkipObjectivePretraining) {
    foreach ($condition in $conditions) {
        $key = $condition.Key
        $checkpoint = "checkpoints\jbhi_revision_$key.pth"
        Run-Python "pretrain objective condition $key" @(
            "scripts\pretrain_tage_timefreq_consistency.py",
            "--data-dirs", $sourceDirs[0], $sourceDirs[1], $sourceDirs[2],
            "--source-names", "BrainLat19", "Auditory40Hz", "ADSZ",
            "--sampling-mode", "shuffle",
            "--output-dir", "$objectiveRoot\pretrain_$key",
            "--checkpoint-out", $checkpoint,
            "--epochs", "50", "--batch-size", "64", "--lr", "1e-3", "--weight-decay", "1e-4",
            "--mask-ratio", "0.3", "--mask-mode", "hybrid",
            "--signal-weight", $condition.Signal,
            "--freq-weight", $condition.Freq,
            "--consistency-weight", $condition.Branch,
            "--crop-consistency-weight", $condition.Crop,
            "--crop-min-ratio", "0.5", "--crop-max-ratio", "1.0", "--seed", "2026"
        ) $checkpoint
    }
}

$commonFineTune = @(
    "scripts\run_dual_stream_subject_split.py",
    "--data-dir", $datasets[0],
    "--epochs", "20", "--batch-size", "32", "--lr", "1e-3", "--weight-decay", "1e-4",
    "--patience", "5", "--use-ea", "--max-ea-samples", "5000",
    "--max-train-segments-per-subject", "250", "--max-val-segments-per-subject", "250",
    "--max-test-segments-per-subject", "400", "--tune-threshold",
    "--threshold-objective", "balanced_acc", "--selection-metric", "auc"
)

if (-not $SkipObjectiveFineTuning) {
    $downstreamConditions = @([pscustomobject]@{ Key = "random_init"; Checkpoint = "" })
    foreach ($condition in $conditions) {
        $downstreamConditions += [pscustomobject]@{
            Key = $condition.Key
            Checkpoint = "checkpoints\jbhi_revision_$($condition.Key).pth"
        }
    }
    foreach ($condition in $downstreamConditions) {
        if ($condition.Checkpoint) { Require-File $condition.Checkpoint }
        foreach ($seed in $Seeds) {
            $outputDir = "$objectiveRoot\$($condition.Key)\seed_$seed"
            $conditionArgs = if ($condition.Checkpoint) {
                @("--checkpoint", $condition.Checkpoint)
            } else {
                @("--no-checkpoint")
            }
            $arguments = @() + $commonFineTune + $conditionArgs + @(
                "--split-file", "$splitRoot\$($datasets[0])_seed$seed.json",
                "--output-dir", $outputDir, "--seed", "$seed"
            )
            Run-Python "fine-tune $($condition.Key), seed $seed" $arguments "$outputDir\summary.json"
        }
    }
}

if (-not $SkipBranchAblation) {
    Require-File "checkpoints\jbhi_revision_full.pth"
    Run-PythonResumableBatch "dual-branch component ablation" (@(
        "scripts\run_current_ablation_batch.py",
        "--checkpoint", "checkpoints\jbhi_revision_full.pth",
        "--datasets", $datasets[0], $datasets[1],
        "--variants", "full", "no_spatial", "no_mscnn",
        "--seeds"
    ) + $seedText + @(
        "--output-root", $branchRoot, "--split-root", $splitRoot,
        "--epochs", "20", "--batch-size", "32", "--lr", "1e-3", "--weight-decay", "1e-4",
        "--patience", "5", "--use-ea", "--max-ea-samples", "5000",
        "--max-train-segments-per-subject", "250", "--max-val-segments-per-subject", "250",
        "--max-test-segments-per-subject", "400", "--threshold-objective", "balanced_acc",
        "--selection-metric", "auc"
    ))
}

if (-not $SkipAggregate) {
    Run-Python "verify and aggregate all revision experiments" (@(
        "scripts\aggregate_jbhi_revision_experiments.py",
        "--objective-root", $objectiveRoot, "--branch-root", $branchRoot,
        "--split-root", $splitRoot, "--output-prefix", "$resultRoot\aggregate",
        "--seeds"
    ) + $seedText + @("--datasets", $datasets[0], $datasets[1])) "$resultRoot\aggregate.json"
}

Write-Log "ALL JBHI REVISION CORE EXPERIMENTS COMPLETED"
