param(
    [string]$PythonExe = "",
    [string]$ProjectRoot = "",
    [int[]]$Seeds = @(2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035)
)

$ErrorActionPreference = "Stop"
if (-not $ProjectRoot) { $ProjectRoot = Split-Path -Parent $PSScriptRoot }
if (-not $PythonExe) {
    $PythonExe = if ($env:BIOFOUNDATION_PYTHON) { $env:BIOFOUNDATION_PYTHON } else { "python" }
}
Set-Location -LiteralPath $ProjectRoot
$env:PYTHONUNBUFFERED = "1"

$caueeg = "CAUEEG_processed_19ch_ad_vs_hc_1s_lead"
$ds = "ds004504_processed_19ch_1s_lead"
$checkpoint = "checkpoints\jbhi_revision_full.pth"
$caueegSplits = "outputs\jbhi_revision\fixed_splits"
$dsSplits = "outputs\jbhi_revision\locked_splits_65"
$resultRoot = "outputs\jbhi_revision\representation_label_efficiency"
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
        Write-Log "SKIP $Label"
        return
    }
    Write-Log "START $Label"
    $previous = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $PythonExe @Arguments 2>&1 |
            ForEach-Object { $_.ToString() } |
            Tee-Object -FilePath $logPath -Append
        $exitCode = $LASTEXITCODE
    }
    finally { $ErrorActionPreference = $previous }
    if ($exitCode -ne 0) { throw "$Label failed with exit code $exitCode" }
    Require-File $DoneFile
    Write-Log "DONE $Label"
}

foreach ($dataset in @($caueeg, $ds)) {
    foreach ($file in @("x_19ch.npy", "y_19ch.npy", "sub_ids_19ch.npy")) {
        Require-File "$dataset\$file"
    }
}
Require-File $checkpoint
foreach ($seed in $Seeds) {
    Require-File "$caueegSplits\${caueeg}_seed$seed.json"
    Require-File "$dsSplits\${ds}_seed$seed.json"
}

$seedText = @($Seeds | ForEach-Object { "$_" })
Run-Python "verify locked ds004504 protocol" (@(
    "scripts\verify_locked_subject_splits.py", "--data-dir", $ds,
    "--split-root", $dsSplits, "--seeds"
) + $seedText + @(
    "--expected-subjects", "65", "--expected-test-subjects", "13",
    "--output", "$resultRoot\ds004504_split_audit.json"
)) "$resultRoot\ds004504_split_audit.json"

$common = @(
    "scripts\run_dual_stream_subject_split.py",
    "--epochs", "20", "--batch-size", "32", "--lr", "1e-3",
    "--weight-decay", "1e-4", "--patience", "5", "--use-ea",
    "--max-ea-samples", "5000", "--max-train-segments-per-subject", "250",
    "--max-val-segments-per-subject", "250", "--max-test-segments-per-subject", "400",
    "--tune-threshold", "--threshold-objective", "balanced_acc",
    "--selection-metric", "auc"
)

foreach ($datasetSpec in @(
    [pscustomobject]@{ Key = "caueeg"; Data = $caueeg; Splits = $caueegSplits },
    [pscustomobject]@{ Key = "ds004504"; Data = $ds; Splits = $dsSplits }
)) {
    foreach ($seed in $Seeds) {
        $split = "$($datasetSpec.Splits)\$($datasetSpec.Data)_seed$seed.json"
        foreach ($condition in @(
            [pscustomobject]@{ Key = "linear_probe"; Extra = @("--linear-probe") },
            [pscustomobject]@{ Key = "discriminative_ft"; Extra = @("--encoder-lr", "1e-5") }
        )) {
            $out = "$resultRoot\representation\$($datasetSpec.Key)\$($condition.Key)\seed_$seed"
            $arguments = @() + $common + @(
                "--data-dir", $datasetSpec.Data, "--checkpoint", $checkpoint,
                "--split-file", $split, "--output-dir", $out, "--seed", "$seed"
            ) + $condition.Extra
            Run-Python "$($datasetSpec.Key) $($condition.Key) seed $seed" $arguments "$out\summary.json"
        }
    }
}

foreach ($fractionSpec in @(
    [pscustomobject]@{ Key = "fraction25"; Value = "0.25" },
    [pscustomobject]@{ Key = "fraction50"; Value = "0.50" }
)) {
    foreach ($seed in $Seeds) {
        $split = "$caueegSplits\${caueeg}_seed$seed.json"
        foreach ($condition in @(
            [pscustomobject]@{ Key = "random_init"; Init = @("--no-checkpoint"); Extra = @() },
            [pscustomobject]@{ Key = "ssl_uniform"; Init = @("--checkpoint", $checkpoint); Extra = @() },
            [pscustomobject]@{ Key = "ssl_discriminative"; Init = @("--checkpoint", $checkpoint); Extra = @("--encoder-lr", "1e-5") }
        )) {
            $out = "$resultRoot\label_efficiency_$($fractionSpec.Key)\caueeg\$($condition.Key)\seed_$seed"
            $arguments = @() + $common + $condition.Init + @(
                "--data-dir", $caueeg, "--split-file", $split,
                "--train-subject-fraction", $fractionSpec.Value,
                "--output-dir", $out, "--seed", "$seed"
            ) + $condition.Extra
            Run-Python "$($fractionSpec.Key) $($condition.Key) seed $seed" $arguments "$out\summary.json"
        }
    }
}

Write-Log "START aggregate representation and label-efficiency experiments"
& $PythonExe "scripts\aggregate_representation_label_efficiency.py" `
    "--input-root" $resultRoot "--output-prefix" "$resultRoot\aggregate" `
    "--seeds" @($Seeds | ForEach-Object { "$_" }) 2>&1 |
    ForEach-Object { $_.ToString() } |
    Tee-Object -FilePath $logPath -Append
if ($LASTEXITCODE -ne 0) { throw "Aggregation failed with exit code $LASTEXITCODE" }
Require-File "$resultRoot\aggregate.json"
Write-Log "REPRESENTATION AND LABEL-EFFICIENCY EXPERIMENTS COMPLETED"
