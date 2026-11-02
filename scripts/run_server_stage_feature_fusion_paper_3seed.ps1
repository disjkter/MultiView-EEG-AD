param(
    [string]$PythonExe = "C:\Users\Administrator\.conda\envs\BioFoundation\python.exe"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$pretrained = "checkpoints\tage_timefreq_consistency_adssl_1s_lead.pth"
$modes = @("pre_only", "source_only", "fixed_average", "concat", "learned_scalar")
foreach ($seed in 2026..2028) {
    $split = "outputs\subject_splits\ds004504_processed_19ch_1s_lead_seed$seed.json"
    $source = "outputs\subject_split_caueeg_tage_timefreq_consistency_adssl_1s_lead_seed$seed\best_model.pth"
    if (-not (Test-Path $source)) { throw "Missing source-task checkpoint: $source" }

    foreach ($mode in $modes) {
        $output = "outputs\stage_feature_fusion_paper_pilot\seed$seed\$mode"
        if (Test-Path "$output\summary.json") {
            Write-Host "SKIP seed=$seed mode=$mode"
            continue
        }
        $splitArgs = if (Test-Path $split) { @("--split-file", $split) } else { @("--save-split-file", $split) }
        $runArgs = @(
            ".\scripts\run_dual_stream_subject_split.py",
            "--config", ".\configs\stage_feature_fusion.yaml",
            "--data-dir", "ds004504_processed_19ch_1s_lead",
            "--output-dir", $output,
            "--model-type", "stage_fusion",
            "--fusion-base-model", "dual_stream",
            "--pretrained-checkpoint", $pretrained,
            "--source-checkpoint", $source,
            "--fusion-mode", $mode,
            "--use-ea",
            "--max-ea-samples", "5000",
            "--epochs", "30",
            "--batch-size", "64",
            "--lr", "0.001",
            "--patience", "6",
            "--selection-metric", "loss",
            "--tune-threshold",
            "--threshold-objective", "balanced_acc",
            "--seed", "$seed",
            "--max-train-segments-per-subject", "250",
            "--max-val-segments-per-subject", "250",
            "--max-test-segments-per-subject", "400"
        )
        $runArgs += $splitArgs
        Write-Host "RUN seed=$seed mode=$mode"
        & $PythonExe @runArgs
        if ($LASTEXITCODE -ne 0) { throw "Fusion run failed: seed=$seed mode=$mode" }
    }
}

& $PythonExe ".\scripts\aggregate_stage_feature_fusion.py" `
    --input-root "outputs\stage_feature_fusion_paper_pilot" `
    --output-prefix "outputs\stage_feature_fusion_paper_pilot\aggregate"
if ($LASTEXITCODE -ne 0) { throw "Aggregation failed." }
Write-Host "Completed paper feature-fusion pilot."
