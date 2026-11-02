param(
    [string]$PythonExe = "",
    [string]$ProjectRoot = "",
    [int[]]$Seeds = @(2026, 2027, 2028)
)

$ErrorActionPreference = "Stop"
if (-not $ProjectRoot) { $ProjectRoot = Split-Path -Parent $PSScriptRoot }
if (-not $PythonExe) {
    $PythonExe = if ($env:BIOFOUNDATION_PYTHON) { $env:BIOFOUNDATION_PYTHON } else { "python" }
}
Set-Location -LiteralPath $ProjectRoot
$env:PYTHONUNBUFFERED = "1"

$dataset = "ds004504_processed_19ch_1s_lead"
$splitRoot = "outputs\jbhi_revision\locked_splits_65"
$resultRoot = "outputs\jbhi_revision\ds004504_matched_baselines"
$baselineRoot = "$resultRoot\supervised"
$tfcRoot = "$resultRoot\tfc"
$branchRoot = "outputs\jbhi_revision\branch_ablation"
$fullCheckpoint = "checkpoints\jbhi_revision_full.pth"
$tfcCheckpoint = "checkpoints\jbhi_revision_tfc.pth"
$logPath = "$resultRoot\server_progress.log"
New-Item -ItemType Directory -Force -Path $resultRoot | Out-Null

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -LiteralPath $logPath -Value $line -Encoding UTF8
    Write-Host $line
}

function Require-File {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { throw "Missing required input: $Path" }
}

function Run-Python {
    param([string]$Label, [string[]]$Arguments, [string]$DoneFile)
    if (Test-Path -LiteralPath $DoneFile) {
        Write-Log "SKIP $Label ($DoneFile exists)"
        return
    }
    Write-Log "START $Label"
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $PythonExe @Arguments 2>&1 |
            ForEach-Object { $_.ToString() } |
            Tee-Object -FilePath $logPath -Append
        $pythonExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($pythonExitCode -ne 0) { throw "$Label failed with exit code $pythonExitCode" }
    Require-File $DoneFile
    Write-Log "DONE $Label"
}

foreach ($fileName in @("x_19ch.npy", "y_19ch.npy", "sub_ids_19ch.npy")) {
    Require-File "$dataset\$fileName"
}
foreach ($seed in $Seeds) {
    Require-File "$splitRoot\${dataset}_seed$seed.json"
}

$sourceDirs = @(
    "BrainLat19_pretrain_19ch_1s_lead",
    "auditory40hz_processed_19ch_1s_lead",
    "ADSZ_processed_19ch_1s_lead"
)

if (-not (Test-Path -LiteralPath $fullCheckpoint)) {
    foreach ($sourceDir in $sourceDirs) { Require-File "$sourceDir\x_19ch.npy" }
    Run-Python "complete multi-task pretraining" @(
        "scripts\pretrain_tage_timefreq_consistency.py",
        "--data-dirs", $sourceDirs[0], $sourceDirs[1], $sourceDirs[2],
        "--source-names", "BrainLat19", "Auditory40Hz", "ADSZ",
        "--sampling-mode", "shuffle",
        "--output-dir", "$resultRoot\pretrain_proposed",
        "--checkpoint-out", $fullCheckpoint,
        "--epochs", "50", "--batch-size", "64", "--lr", "1e-3", "--weight-decay", "1e-4",
        "--mask-ratio", "0.3", "--mask-mode", "hybrid",
        "--signal-weight", "1", "--freq-weight", "0.5",
        "--consistency-weight", "0.1", "--crop-consistency-weight", "0.1",
        "--crop-min-ratio", "0.5", "--crop-max-ratio", "1.0", "--seed", "2026"
    ) $fullCheckpoint
}

if (-not (Test-Path -LiteralPath $tfcCheckpoint)) {
    foreach ($sourceDir in $sourceDirs) { Require-File "$sourceDir\x_19ch.npy" }
    Run-Python "TF-C external pretraining" @(
        "scripts\pretrain_tfc_baseline.py",
        "--data-dirs", $sourceDirs[0], $sourceDirs[1], $sourceDirs[2],
        "--source-names", "BrainLat19", "Auditory40Hz", "ADSZ",
        "--output-dir", "$resultRoot\pretrain_tfc",
        "--checkpoint-out", $tfcCheckpoint,
        "--config", "configs\tfc_baseline_jbhi.yaml",
        "--epochs", "50", "--batch-size", "64", "--lr", "1e-3", "--weight-decay", "1e-4",
        "--temperature", "0.2", "--within-domain-weight", "0.2",
        "--jitter-std", "0.02", "--frequency-remove-ratio", "0.1", "--frequency-add-ratio", "0.1",
        "--model-dim", "128", "--projection-dim", "128",
        "--num-layers", "2", "--num-heads", "4", "--seed", "2026"
    ) $tfcCheckpoint
}

$baselineModels = @("svm", "eegnet", "eegconformer", "ceednet")
foreach ($model in $baselineModels) {
    foreach ($seed in $Seeds) {
        $doneFile = "$baselineRoot\$model\seed_$seed\summary.json"
        Run-Python "$model seed $seed" @(
            "scripts\run_subject_split_baselines.py",
            "--data-dir", $dataset, "--model", $model,
            "--output-dir", "$baselineRoot\$model", "--seeds", "$seed",
            "--split-file", "$splitRoot\${dataset}_seed$seed.json",
            "--epochs", "20", "--batch-size", "32", "--lr", "1e-3", "--weight-decay", "1e-4",
            "--patience", "5", "--use-ea", "--max-ea-samples", "5000",
            "--max-train-segments-per-subject", "250",
            "--max-val-segments-per-subject", "250",
            "--max-test-segments-per-subject", "400",
            "--tune-threshold", "--threshold-objective", "balanced_acc", "--selection-metric", "auc"
        ) $doneFile
    }
}

foreach ($seed in $Seeds) {
    $outputDir = "$tfcRoot\seed_$seed"
    Run-Python "TF-C seed $seed" @(
        "scripts\run_dual_stream_subject_split.py",
        "--config", "configs\tfc_baseline_jbhi.yaml",
        "--data-dir", $dataset, "--model-type", "tfc", "--checkpoint", $tfcCheckpoint,
        "--split-file", "$splitRoot\${dataset}_seed$seed.json",
        "--output-dir", $outputDir, "--seed", "$seed",
        "--epochs", "20", "--batch-size", "32", "--lr", "1e-3", "--weight-decay", "1e-4",
        "--patience", "5", "--use-ea", "--max-ea-samples", "5000",
        "--max-train-segments-per-subject", "250",
        "--max-val-segments-per-subject", "250",
        "--max-test-segments-per-subject", "400",
        "--tune-threshold", "--threshold-objective", "balanced_acc", "--selection-metric", "auc",
        "--tfc-model-dim", "128", "--tfc-projection-dim", "128",
        "--tfc-num-layers", "2", "--tfc-num-heads", "4"
    ) "$outputDir\summary.json"
}

$missingProposed = $false
foreach ($seed in $Seeds) {
    if (-not (Test-Path -LiteralPath "$branchRoot\$dataset\full_s$seed\summary.json")) {
        $missingProposed = $true
    }
}
if ($missingProposed) {
    $seedText = @($Seeds | ForEach-Object { "$_" })
    Write-Log "START/RESUME proposed framework"
    $proposedArgs = @(
        "scripts\run_current_ablation_batch.py",
        "--checkpoint", $fullCheckpoint, "--datasets", $dataset,
        "--variants", "full", "--seeds"
    ) + $seedText + @(
        "--output-root", $branchRoot, "--split-root", $splitRoot,
        "--epochs", "20", "--batch-size", "32", "--lr", "1e-3", "--weight-decay", "1e-4",
        "--patience", "5", "--use-ea", "--max-ea-samples", "5000",
        "--max-train-segments-per-subject", "250",
        "--max-val-segments-per-subject", "250",
        "--max-test-segments-per-subject", "400",
        "--threshold-objective", "balanced_acc", "--selection-metric", "auc"
    )
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $PythonExe @proposedArgs 2>&1 |
            ForEach-Object { $_.ToString() } |
            Tee-Object -FilePath $logPath -Append
        $pythonExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($pythonExitCode -ne 0) { throw "Proposed framework failed with exit code $pythonExitCode" }
    Write-Log "DONE proposed framework"
}

$seedText = @($Seeds | ForEach-Object { "$_" })
Run-Python "verify and aggregate matched ds004504 baselines" (@(
    "scripts\aggregate_ds004504_matched_baselines.py",
    "--baseline-root", $baselineRoot,
    "--tfc-root", $tfcRoot,
    "--branch-root", $branchRoot,
    "--split-root", $splitRoot,
    "--dataset", $dataset,
    "--seeds"
) + $seedText + @(
    "--output-prefix", "$resultRoot\aggregate"
)) "$resultRoot\aggregate.json"

Write-Log "DS004504 MATCHED BASELINE EXPERIMENTS COMPLETED"
