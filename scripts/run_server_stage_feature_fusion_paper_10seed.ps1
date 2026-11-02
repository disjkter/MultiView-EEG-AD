param(
    [string]$PythonExe = "C:\Users\Administrator\.conda\envs\BioFoundation\python.exe"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
$pretrained = "checkpoints\tage_timefreq_consistency_adssl_1s_lead.pth"
$modes = @("pre_only", "source_only", "fixed_average", "concat", "learned_scalar")

foreach ($seed in 2026..2035) {
    $sourceSplit = "outputs\subject_splits\CAUEEG_processed_19ch_ad_vs_hc_1s_lead_seed$seed.json"
    $sourceDir = "outputs\subject_split_caueeg_tage_timefreq_consistency_adssl_1s_lead_seed$seed"
    $source = "$sourceDir\best_model.pth"
    if (-not (Test-Path $source)) {
        Write-Host "TRAIN source-task encoder seed=$seed"
        & $PythonExe ".\scripts\run_dual_stream_subject_split.py" `
            --data-dir "CAUEEG_processed_19ch_ad_vs_hc_1s_lead" `
            --split-file $sourceSplit `
            --checkpoint $pretrained `
            --output-dir $sourceDir `
            --epochs 20 --batch-size 32 --lr 0.001 --weight-decay 0.0001 `
            --patience 5 --use-ea --max-ea-samples 5000 `
            --max-train-segments-per-subject 250 `
            --max-val-segments-per-subject 250 `
            --max-test-segments-per-subject 400 `
            --tune-threshold --threshold-objective balanced_acc `
            --selection-metric auc --seed $seed
        if ($LASTEXITCODE -ne 0) { throw "Source-task training failed: seed=$seed" }
    }

    $targetSplit = "outputs\subject_splits\ds004504_processed_19ch_1s_lead_seed$seed.json"
    foreach ($mode in $modes) {
        $output = "outputs\stage_feature_fusion_paper_10seed\seed$seed\$mode"
        if (Test-Path "$output\summary.json") {
            Write-Host "SKIP seed=$seed mode=$mode"
            continue
        }
        $splitArgs = if (Test-Path $targetSplit) { @("--split-file", $targetSplit) } else { @("--save-split-file", $targetSplit) }
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
            "--use-ea", "--max-ea-samples", "5000",
            "--epochs", "30", "--batch-size", "64", "--lr", "0.001",
            "--patience", "6", "--selection-metric", "loss",
            "--tune-threshold", "--threshold-objective", "balanced_acc",
            "--seed", "$seed",
            "--max-train-segments-per-subject", "250",
            "--max-val-segments-per-subject", "250",
            "--max-test-segments-per-subject", "400"
        )
        $runArgs += $splitArgs
        Write-Host "RUN target seed=$seed mode=$mode"
        & $PythonExe @runArgs
        if ($LASTEXITCODE -ne 0) { throw "Fusion run failed: seed=$seed mode=$mode" }
    }
}

& $PythonExe ".\scripts\aggregate_stage_feature_fusion.py" `
    --input-root "outputs\stage_feature_fusion_paper_10seed" `
    --output-prefix "outputs\stage_feature_fusion_paper_10seed\aggregate"
if ($LASTEXITCODE -ne 0) { throw "Aggregation failed." }
Write-Host "Completed paper feature-fusion confirmation."
