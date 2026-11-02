param(
    [string]$PythonExe = "C:\Users\Administrator\.conda\envs\BioFoundation\python.exe"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

foreach ($seed in 2026..2035) {
    $split = "outputs\caueeg_support4_splits\CAUEEG_seed$seed.json"
    $checkpoint = "outputs\source_models_ds004504_lossselect\ds004504_seed$seed\best_model.pth"
    $output = "outputs\transfer_ds004504_caueeg_fewshot\support4_seed$seed\ds004504_calibrated"

    if (-not (Test-Path $split)) { throw "Missing split: $split" }
    if (-not (Test-Path $checkpoint)) { throw "Missing checkpoint: $checkpoint" }
    $splitPayload = Get-Content -LiteralPath $split -Raw | ConvertFrom-Json
    if ($splitPayload.train_subjects.Count -ne 8) {
        throw "Invalid support-4 split for seed=${seed}: expected 8 train subjects, found $($splitPayload.train_subjects.Count)"
    }
    if ($splitPayload.excluded_subjects.Count -eq 0) {
        throw "Invalid support-4 split for seed=${seed}: excluded_subjects is empty"
    }
    if (Test-Path "$output\summary.json") {
        Write-Host "SKIP few-shot transfer seed=$seed"
        continue
    }

    $runArgs = @(
        ".\scripts\run_dual_stream_subject_split.py",
        "--data-dir", "CAUEEG_processed_19ch_ad_vs_hc_1s_lead",
        "--split-file", $split,
        "--output-dir", $output,
        "--model-type", "graph_temporal",
        "--checkpoint", $checkpoint,
        "--checkpoint-scope", "encoders",
        "--epochs", "30",
        "--batch-size", "32",
        "--lr", "0.001",
        "--weight-decay", "0.0001",
        "--patience", "8",
        "--selection-metric", "loss",
        "--seed", "$seed",
        "--graph-num-neighbors", "4",
        "--spatial-out-dim", "256",
        "--temporal-out-dim", "256",
        "--classifier-dropout", "0.6",
        "--max-train-segments-per-subject", "20",
        "--max-val-segments-per-subject", "100",
        "--max-test-segments-per-subject", "100",
        "--tune-threshold",
        "--threshold-objective", "balanced_acc"
    )

    for ($attempt = 1; $attempt -le 3; $attempt++) {
        Write-Host "RUN few-shot transfer seed=$seed attempt=$attempt"
        & $PythonExe @runArgs
        if ($LASTEXITCODE -eq 0) { break }
    }
    if ($LASTEXITCODE -ne 0) { throw "Few-shot transfer failed: seed=$seed" }
}

$count = (Get-ChildItem outputs\transfer_ds004504_caueeg_fewshot -Recurse -Filter summary.json).Count
Write-Host "Completed few-shot transfer summaries: $count/10"
