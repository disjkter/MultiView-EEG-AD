param(
    [string]$PythonExe = "C:\Users\Administrator\.conda\envs\BioFoundation\python.exe"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$pretrained = "checkpoints\graph_temporal_mae_multisource.pth"
if (-not (Test-Path $pretrained)) {
    throw "Missing $pretrained. Run multi-source Graph MAE pretraining first."
}

$modes = @("pre_only", "source_only", "fixed_average", "concat", "learned_scalar")
foreach ($seed in 2026..2028) {
    $split = "outputs\subject_splits_v2\ds004504_processed_19ch_1s_v2_seed$seed.json"
    $source = "outputs\source_models_local_lossselect\ADFSU_seed$seed\best_model.pth"
    if (-not (Test-Path $split)) { throw "Missing split: $split" }
    if (-not (Test-Path $source)) { throw "Missing source-task checkpoint: $source" }

    foreach ($mode in $modes) {
        $output = "outputs\stage_feature_fusion_graph_pilot\seed$seed\$mode"
        if (Test-Path "$output\summary.json") {
            Write-Host "SKIP seed=$seed mode=$mode"
            continue
        }
        $runArgs = @(
            ".\scripts\run_dual_stream_subject_split.py",
            "--config", ".\configs\stage_feature_fusion.yaml",
            "--data-dir", "ds004504_processed_19ch_1s_v2",
            "--split-file", $split,
            "--output-dir", $output,
            "--model-type", "stage_fusion",
            "--fusion-base-model", "graph_temporal",
            "--pretrained-checkpoint", $pretrained,
            "--source-checkpoint", $source,
            "--fusion-mode", $mode,
            "--seed", "$seed",
            "--max-train-segments-per-subject", "100",
            "--max-val-segments-per-subject", "100",
            "--max-test-segments-per-subject", "100"
        )
        Write-Host "RUN seed=$seed mode=$mode"
        & $PythonExe @runArgs
        if ($LASTEXITCODE -ne 0) { throw "Fusion run failed: seed=$seed mode=$mode" }
    }
}

& $PythonExe ".\scripts\aggregate_stage_feature_fusion.py" `
    --input-root "outputs\stage_feature_fusion_graph_pilot" `
    --output-prefix "outputs\stage_feature_fusion_graph_pilot\aggregate"
if ($LASTEXITCODE -ne 0) { throw "Aggregation failed." }
Write-Host "Completed graph feature-fusion pilot."
