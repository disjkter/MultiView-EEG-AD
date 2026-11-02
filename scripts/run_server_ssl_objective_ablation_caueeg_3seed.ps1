param(
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $PythonExe) {
    $PythonExe = if ($env:BIOFOUNDATION_PYTHON) { $env:BIOFOUNDATION_PYTHON } else { "python" }
}
$logPath = Join-Path $projectRoot "outputs\ssl_objective_ablation_caueeg.log"

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

$pretrainDataArgs = @()
if (Test-Path -LiteralPath "ADSSL_pretrain_19ch_1s_lead\x_19ch.npy") {
    $pretrainDataArgs = @("--data-dir", "ADSSL_pretrain_19ch_1s_lead")
    Write-Log "Using merged ADSSL_pretrain_19ch_1s_lead input"
} else {
    $sourceDirs = @(
        "BrainLat19_pretrain_19ch_1s_lead",
        "auditory40hz_processed_19ch_1s_lead",
        "ADSZ_processed_19ch_1s_lead"
    )
    $missingSources = @($sourceDirs | Where-Object { -not (Test-Path -LiteralPath "$_\x_19ch.npy") })
    if ($missingSources.Count -gt 0) {
        throw "Missing pretraining input. Supply ADSSL_pretrain_19ch_1s_lead, or all three component directories. Missing components: $($missingSources -join ', ')"
    }
    $pretrainDataArgs = @(
        "--data-dirs", $sourceDirs[0], $sourceDirs[1], $sourceDirs[2],
        "--source-names", "BrainLat19", "Auditory40Hz", "ADSZ",
        "--sampling-mode", "proportional"
    )
    Write-Log "Merged input absent; using the three component pretraining directories directly"
}

Run-Python "create fixed CAUEEG splits" @(
    "scripts\create_subject_splits.py", "--datasets", "CAUEEG_processed_19ch_ad_vs_hc_1s_lead",
    "--seeds", "2026", "2027", "2028", "--output-dir", "outputs\paper_fixed_splits"
) "outputs\paper_fixed_splits\manifest.json"

$pretrainConditions = @(
    @{ Key = "signal"; Freq = "0"; Branch = "0"; Crop = "0" },
    @{ Key = "signal_freq"; Freq = "0.5"; Branch = "0"; Crop = "0" },
    @{ Key = "signal_freq_branch"; Freq = "0.5"; Branch = "0.1"; Crop = "0" },
    @{ Key = "full"; Freq = "0.5"; Branch = "0.1"; Crop = "0.1" }
)

foreach ($condition in $pretrainConditions) {
    $key = $condition.Key
    $pretrainArgs = @(
        "scripts\pretrain_tage_timefreq_consistency.py",
        "--output-dir", "outputs\ssl_objective_ablation_caueeg\pretrain_$key",
        "--checkpoint-out", "checkpoints\tage_ssl_ablation_$key.pth",
        "--epochs", "50", "--batch-size", "64", "--lr", "1e-3", "--weight-decay", "1e-4",
        "--mask-ratio", "0.3", "--mask-mode", "hybrid",
        "--freq-weight", $condition.Freq,
        "--consistency-weight", $condition.Branch,
        "--crop-consistency-weight", $condition.Crop,
        "--crop-min-ratio", "0.5", "--crop-max-ratio", "1.0", "--seed", "2026"
    ) + $pretrainDataArgs
    Run-Python "pretrain $key" $pretrainArgs "outputs\ssl_objective_ablation_caueeg\pretrain_$key\summary.json"
}

$commonFineTune = @(
    "scripts\run_dual_stream_subject_split.py",
    "--data-dir", "CAUEEG_processed_19ch_ad_vs_hc_1s_lead",
    "--epochs", "20", "--batch-size", "32", "--lr", "1e-3", "--weight-decay", "1e-4",
    "--patience", "5", "--use-ea", "--max-ea-samples", "5000",
    "--max-train-segments-per-subject", "250",
    "--max-val-segments-per-subject", "250",
    "--max-test-segments-per-subject", "400",
    "--tune-threshold", "--threshold-objective", "balanced_acc", "--selection-metric", "auc"
)

foreach ($condition in $pretrainConditions) {
    $key = $condition.Key
    foreach ($seed in @(2026, 2027, 2028)) {
        $outputDir = "outputs\ssl_objective_ablation_caueeg\$key\seed_$seed"
        $splitFile = "outputs\paper_fixed_splits\CAUEEG_processed_19ch_ad_vs_hc_1s_lead_seed$seed.json"
        $args = @() + $commonFineTune + @(
            "--checkpoint", "checkpoints\tage_ssl_ablation_$key.pth",
            "--split-file", $splitFile,
            "--output-dir", $outputDir,
            "--seed", "$seed"
        )
        Run-Python "fine-tune $key seed $seed" $args "$outputDir\summary.json"
    }
}

Run-Python "aggregate objective ablation" @(
    "scripts\aggregate_ssl_objective_ablation.py"
) "outputs\ssl_objective_ablation_caueeg\aggregate.json"

Write-Log "ALL SSL OBJECTIVE ABLATIONS COMPLETED"
