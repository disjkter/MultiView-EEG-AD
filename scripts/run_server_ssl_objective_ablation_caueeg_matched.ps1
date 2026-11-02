param(
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $PythonExe) {
    $PythonExe = if ($env:BIOFOUNDATION_PYTHON) { $env:BIOFOUNDATION_PYTHON } else { "python" }
}
$resultRoot = "outputs\ssl_objective_ablation_caueeg_matched"
$logPath = Join-Path $projectRoot "$resultRoot.log"

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -LiteralPath $logPath -Value $line -Encoding UTF8
    Write-Host $line
}

function Run-Python {
    param([string]$Label, [string[]]$Arguments, [string]$DoneFile)
    $donePath = Join-Path $projectRoot $DoneFile
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

Set-Location -LiteralPath $projectRoot
New-Item -ItemType Directory -Force -Path "outputs" | Out-Null

if (-not (Test-Path -LiteralPath "CAUEEG_processed_19ch_ad_vs_hc_1s_lead\x_19ch.npy")) {
    throw "Missing CAUEEG input: CAUEEG_processed_19ch_ad_vs_hc_1s_lead\x_19ch.npy"
}

if (-not (Test-Path -LiteralPath "ADSSL_pretrain_19ch_1s_lead\x_19ch.npy")) {
    $sourceDirs = @(
        "BrainLat19_pretrain_19ch_1s_lead",
        "auditory40hz_processed_19ch_1s_lead",
        "ADSZ_processed_19ch_1s_lead"
    )
    $missing = @()
    foreach ($sourceDir in $sourceDirs) {
        foreach ($fileName in @("x_19ch.npy", "y_19ch.npy", "sub_ids_19ch.npy")) {
            $candidate = "$sourceDir\$fileName"
            if (-not (Test-Path -LiteralPath $candidate)) { $missing += $candidate }
        }
    }
    if ($missing.Count -gt 0) {
        throw "Cannot rebuild the merged pretraining data. Missing: $($missing -join ', ')"
    }
    Run-Python "rebuild merged pretraining data" @(
        "scripts\build_ssl_pretrain_dataset.py",
        "--inputs", $sourceDirs[0], $sourceDirs[1], $sourceDirs[2],
        "--names", "BrainLat19", "Auditory40Hz", "ADSZ",
        "--output-dir", "ADSSL_pretrain_19ch_1s_lead",
        "--target-sfreq", "128", "--window-seconds", "1"
    ) "ADSSL_pretrain_19ch_1s_lead\summary.json"
}
Write-Log "Using merged data with exhaustive shuffle (each sample once per epoch)"

Run-Python "create fixed CAUEEG splits" @(
    "scripts\create_subject_splits.py", "--datasets", "CAUEEG_processed_19ch_ad_vs_hc_1s_lead",
    "--seeds", "2026", "2027", "2028", "--output-dir", "outputs\paper_fixed_splits"
) "outputs\paper_fixed_splits\manifest.json"

$conditions = @(
    @{ Key = "signal"; Freq = "0"; Branch = "0"; Crop = "0" },
    @{ Key = "signal_freq"; Freq = "0.5"; Branch = "0"; Crop = "0" },
    @{ Key = "signal_freq_branch"; Freq = "0.5"; Branch = "0.1"; Crop = "0" }
)

foreach ($condition in $conditions) {
    $key = $condition.Key
    Run-Python "pretrain $key with matched sampling" @(
        "scripts\pretrain_tage_timefreq_consistency.py",
        "--data-dir", "ADSSL_pretrain_19ch_1s_lead", "--sampling-mode", "shuffle",
        "--output-dir", "$resultRoot\pretrain_$key",
        "--checkpoint-out", "checkpoints\tage_ssl_ablation_matched_$key.pth",
        "--epochs", "50", "--batch-size", "64", "--lr", "1e-3", "--weight-decay", "1e-4",
        "--mask-ratio", "0.3", "--mask-mode", "hybrid",
        "--freq-weight", $condition.Freq,
        "--consistency-weight", $condition.Branch,
        "--crop-consistency-weight", $condition.Crop,
        "--crop-min-ratio", "0.5", "--crop-max-ratio", "1.0", "--seed", "2026"
    ) "$resultRoot\pretrain_$key\summary.json"
}

$commonFineTune = @(
    "scripts\run_dual_stream_subject_split.py",
    "--data-dir", "CAUEEG_processed_19ch_ad_vs_hc_1s_lead",
    "--epochs", "20", "--batch-size", "32", "--lr", "1e-3", "--weight-decay", "1e-4",
    "--patience", "5", "--use-ea", "--max-ea-samples", "5000",
    "--max-train-segments-per-subject", "250", "--max-val-segments-per-subject", "250",
    "--max-test-segments-per-subject", "400", "--tune-threshold",
    "--threshold-objective", "balanced_acc", "--selection-metric", "auc"
)

foreach ($condition in $conditions) {
    $key = $condition.Key
    foreach ($seed in @(2026, 2027, 2028)) {
        $outputDir = "$resultRoot\$key\seed_$seed"
        $args = @() + $commonFineTune + @(
            "--checkpoint", "checkpoints\tage_ssl_ablation_matched_$key.pth",
            "--split-file", "outputs\paper_fixed_splits\CAUEEG_processed_19ch_ad_vs_hc_1s_lead_seed$seed.json",
            "--output-dir", $outputDir, "--seed", "$seed"
        )
        Run-Python "fine-tune $key seed $seed" $args "$outputDir\summary.json"
    }
}

Run-Python "aggregate matched objective ablation" @(
    "scripts\aggregate_ssl_objective_ablation.py",
    "--input-root", $resultRoot,
    "--output-prefix", "$resultRoot\aggregate"
) "$resultRoot\aggregate.json"

Write-Log "ALL MATCHED SSL OBJECTIVE ABLATIONS COMPLETED"
