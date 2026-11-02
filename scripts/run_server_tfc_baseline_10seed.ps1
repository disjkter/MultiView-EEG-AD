param(
    [string]$PythonExe = "",
    [string]$ProjectRoot = "",
    [int[]]$Seeds = @(2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035),
    [switch]$SkipPretraining,
    [switch]$SkipFineTuning
)

$ErrorActionPreference = "Stop"
if (-not $ProjectRoot) { $ProjectRoot = Split-Path -Parent $PSScriptRoot }
if (-not $PythonExe) {
    $PythonExe = if ($env:BIOFOUNDATION_PYTHON) { $env:BIOFOUNDATION_PYTHON } else { "python" }
}
Set-Location -LiteralPath $ProjectRoot
# Keep epoch progress visible while Python output is piped through Tee-Object.
$env:PYTHONUNBUFFERED = "1"

$dataset = "CAUEEG_processed_19ch_ad_vs_hc_1s_lead"
$sourceDirs = @(
    "BrainLat19_pretrain_19ch_1s_lead",
    "auditory40hz_processed_19ch_1s_lead",
    "ADSZ_processed_19ch_1s_lead"
)
$splitRoot = "outputs\jbhi_revision\fixed_splits"
$resultRoot = "outputs\jbhi_revision\tfc_baseline"
$checkpoint = "checkpoints\jbhi_revision_tfc.pth"
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
    # Windows PowerShell 5 wraps every native stderr line as an ErrorRecord.
    # PyTorch writes non-fatal warnings to stderr, so ErrorActionPreference=Stop
    # would otherwise terminate the script before LASTEXITCODE can be checked.
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
foreach ($sourceDir in $sourceDirs) { Require-File "$sourceDir\x_19ch.npy" }
foreach ($seed in $Seeds) { Require-File "$splitRoot\${dataset}_seed$seed.json" }

if (-not $SkipPretraining) {
    Run-Python "TF-C external pretraining" @(
        "scripts\pretrain_tfc_baseline.py",
        "--data-dirs", $sourceDirs[0], $sourceDirs[1], $sourceDirs[2],
        "--source-names", "BrainLat19", "Auditory40Hz", "ADSZ",
        "--output-dir", "$resultRoot\pretrain",
        "--checkpoint-out", $checkpoint,
        "--config", "configs\tfc_baseline_jbhi.yaml",
        "--epochs", "50", "--batch-size", "64", "--lr", "1e-3", "--weight-decay", "1e-4",
        "--temperature", "0.2", "--within-domain-weight", "0.2",
        "--jitter-std", "0.02", "--frequency-remove-ratio", "0.1", "--frequency-add-ratio", "0.1",
        "--model-dim", "128", "--projection-dim", "128", "--num-layers", "2", "--num-heads", "4",
        "--seed", "2026"
    ) $checkpoint
}
Require-File $checkpoint

if (-not $SkipFineTuning) {
    foreach ($seed in $Seeds) {
        $outputDir = "$resultRoot\seed_$seed"
        Run-Python "TF-C fine-tuning seed $seed" @(
            "scripts\run_dual_stream_subject_split.py",
            "--config", "configs\tfc_baseline_jbhi.yaml",
            "--data-dir", $dataset,
            "--model-type", "tfc", "--checkpoint", $checkpoint,
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
}

$seedText = @($Seeds | ForEach-Object { "$_" })
Write-Log "START verify and aggregate TF-C"
$aggregateArgs = @(
    "scripts\aggregate_tfc_baseline.py",
    "--input-root", $resultRoot,
    "--split-root", $splitRoot,
    "--output-prefix", "$resultRoot\aggregate",
    "--dataset", $dataset,
    "--seeds"
) + $seedText
$previousErrorActionPreference = $ErrorActionPreference
try {
    $ErrorActionPreference = "Continue"
    & $PythonExe @aggregateArgs 2>&1 |
        ForEach-Object { $_.ToString() } |
        Tee-Object -FilePath $logPath -Append
    $aggregateExitCode = $LASTEXITCODE
}
finally {
    $ErrorActionPreference = $previousErrorActionPreference
}
if ($aggregateExitCode -ne 0) { throw "TF-C aggregation failed with exit code $aggregateExitCode" }
Require-File "$resultRoot\aggregate.json"
Write-Log "TF-C BASELINE COMPLETED"
