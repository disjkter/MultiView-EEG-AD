param(
    [string]$PythonExe = "",
    [string]$ProjectRoot = "",
    [int[]]$Seeds = @(2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035),
    [switch]$SkipSourceTraining,
    [switch]$SkipTargetTraining
)

$ErrorActionPreference = "Stop"
if (-not $ProjectRoot) { $ProjectRoot = Split-Path -Parent $PSScriptRoot }
if (-not $PythonExe) {
    $PythonExe = if ($env:BIOFOUNDATION_PYTHON) { $env:BIOFOUNDATION_PYTHON } else { "python" }
}
Set-Location -LiteralPath $ProjectRoot
$env:PYTHONUNBUFFERED = "1"

$sourceDataset = "CAUEEG_processed_19ch_ad_vs_hc_1s_lead"
$targetDataset = "ds004504_processed_19ch_1s_lead"
$pretrained = "checkpoints\jbhi_revision_full.pth"
$splitRoot = "outputs\jbhi_revision\fixed_splits"
$resultRoot = "outputs\jbhi_revision\stage_transfer_ablation"
$config = "configs\stage_transfer_ablation_jbhi.yaml"
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

function Run-PythonBatch {
    param([string]$Label, [string[]]$Arguments)
    Write-Log "START/RESUME $Label"
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
    Write-Log "DONE $Label"
}

foreach ($dataset in @($sourceDataset, $targetDataset)) {
    foreach ($fileName in @("x_19ch.npy", "y_19ch.npy", "sub_ids_19ch.npy")) {
        Require-File "$dataset\$fileName"
    }
}
Require-File $pretrained
Require-File $config

$seedText = @($Seeds | ForEach-Object { "$_" })
Run-PythonBatch "create fixed participant splits" (@(
    "scripts\create_subject_splits.py", "--datasets", $sourceDataset, $targetDataset, "--seeds"
) + $seedText + @(
    "--train-ratio", "0.7", "--val-ratio", "0.1", "--output-dir", $splitRoot
))
foreach ($seed in $Seeds) {
    Require-File "$splitRoot\${sourceDataset}_seed$seed.json"
    Require-File "$splitRoot\${targetDataset}_seed$seed.json"
}

$common = @(
    "scripts\run_dual_stream_subject_split.py",
    "--config", $config,
    "--max-ea-samples", "5000",
    "--max-train-segments-per-subject", "250",
    "--max-val-segments-per-subject", "250",
    "--max-test-segments-per-subject", "400"
)

if (-not $SkipSourceTraining) {
    foreach ($seed in $Seeds) {
        $sourceSplit = "$splitRoot\${sourceDataset}_seed$seed.json"
        foreach ($sourceCondition in @("source_from_random", "source_from_ssl")) {
            $sourceDir = "$resultRoot\source\$sourceCondition\seed_$seed"
            $initArgs = if ($sourceCondition -eq "source_from_ssl") {
                @("--checkpoint", $pretrained)
            } else {
                @("--no-checkpoint")
            }
            $arguments = @() + $common + $initArgs + @(
                "--data-dir", $sourceDataset,
                "--split-file", $sourceSplit,
                "--output-dir", $sourceDir,
                "--seed", "$seed"
            )
            Run-Python "source training $sourceCondition seed $seed" $arguments "$sourceDir\summary.json"
            Require-File "$sourceDir\best_model.pth"
        }
    }
}

if (-not $SkipTargetTraining) {
    foreach ($seed in $Seeds) {
        $targetSplit = "$splitRoot\${targetDataset}_seed$seed.json"
        $conditions = @(
            [pscustomobject]@{ Key = "random_init"; Checkpoint = "" },
            [pscustomobject]@{ Key = "ssl_only"; Checkpoint = $pretrained },
            [pscustomobject]@{ Key = "source_only"; Checkpoint = "$resultRoot\source\source_from_random\seed_$seed\best_model.pth" },
            [pscustomobject]@{ Key = "ssl_source"; Checkpoint = "$resultRoot\source\source_from_ssl\seed_$seed\best_model.pth" }
        )
        foreach ($condition in $conditions) {
            $targetDir = "$resultRoot\target\$($condition.Key)\seed_$seed"
            $initArgs = if ($condition.Checkpoint) {
                Require-File $condition.Checkpoint
                @("--checkpoint", $condition.Checkpoint)
            } else {
                @("--no-checkpoint")
            }
            $arguments = @() + $common + $initArgs + @(
                "--data-dir", $targetDataset,
                "--split-file", $targetSplit,
                "--output-dir", $targetDir,
                "--seed", "$seed"
            )
            Run-Python "target $($condition.Key) seed $seed" $arguments "$targetDir\summary.json"
        }
    }
}

Write-Log "START verify and aggregate staged-transfer ablation"
$aggregateArgs = @(
    "scripts\aggregate_stage_transfer_ablation.py",
    "--input-root", $resultRoot,
    "--split-root", $splitRoot,
    "--output-prefix", "$resultRoot\aggregate",
    "--dataset", $targetDataset,
    "--seeds"
) + $seedText
& $PythonExe @aggregateArgs 2>&1 |
    ForEach-Object { $_.ToString() } |
    Tee-Object -FilePath $logPath -Append
$aggregateExitCode = $LASTEXITCODE
if ($aggregateExitCode -ne 0) { throw "Stage-transfer aggregation failed with exit code $aggregateExitCode" }
Require-File "$resultRoot\aggregate.json"
Write-Log "STAGED-TRANSFER ABLATION COMPLETED"
