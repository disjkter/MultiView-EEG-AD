param(
    [string]$PythonExe = "C:\Users\Administrator\.conda\envs\BioFoundation\python.exe"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

foreach ($seed in 2026..2035) {
    $split = "outputs\subject_splits_v2\ds004504_processed_19ch_1s_v2_seed$seed.json"
    $output = "outputs\source_models_ds004504_lossselect\ds004504_seed$seed"
    if (Test-Path "$output\best_model.pth") {
        Write-Host "SKIP source seed=$seed"
        continue
    }

    $runArgs = @(
        ".\scripts\run_dual_stream_subject_split.py",
        "--data-dir", "ds004504_processed_19ch_1s_v2",
        "--split-file", $split,
        "--output-dir", $output,
        "--no-checkpoint",
        "--model-type", "graph_temporal",
        "--epochs", "50",
        "--batch-size", "64",
        "--lr", "0.001",
        "--weight-decay", "0.0001",
        "--patience", "8",
        "--selection-metric", "loss",
        "--seed", "$seed",
        "--graph-num-neighbors", "4",
        "--spatial-out-dim", "256",
        "--temporal-out-dim", "256",
        "--classifier-dropout", "0.6",
        "--max-train-segments-per-subject", "100",
        "--max-val-segments-per-subject", "100",
        "--max-test-segments-per-subject", "100"
    )

    for ($attempt = 1; $attempt -le 3; $attempt++) {
        Write-Host "RUN source seed=$seed attempt=$attempt"
        & $PythonExe @runArgs
        if ($LASTEXITCODE -eq 0) { break }
    }
    if ($LASTEXITCODE -ne 0) { throw "Source training failed: seed=$seed" }
}

$count = (Get-ChildItem outputs\source_models_ds004504_lossselect -Recurse -Filter best_model.pth).Count
Write-Host "Completed source checkpoints: $count/10"
