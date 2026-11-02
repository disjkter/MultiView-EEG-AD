param(
    [string]$PythonExe = "C:\Users\Administrator\.conda\envs\BioFoundation\python.exe"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

foreach ($support in 1, 2, 8) {
    foreach ($seed in 2026..2035) {
        $split = "outputs\caueeg_support_curve_splits\support${support}_seed${seed}\CAUEEG_seed${seed}.json"
        $checkpoint = "outputs\source_models_ds004504_lossselect\ds004504_seed${seed}\best_model.pth"
        $output = "outputs\transfer_ds004504_caueeg_support_curve\support${support}_seed${seed}\ds004504_calibrated"
        $expectedTrainSubjects = 2 * $support

        if (-not (Test-Path $split)) { throw "Missing split: $split" }
        if (-not (Test-Path $checkpoint)) { throw "Missing checkpoint: $checkpoint" }
        $splitPayload = Get-Content -LiteralPath $split -Raw | ConvertFrom-Json
        if ($splitPayload.train_subjects.Count -ne $expectedTrainSubjects) {
            throw "Invalid support-${support} split for seed=${seed}: expected $expectedTrainSubjects train subjects, found $($splitPayload.train_subjects.Count)"
        }
        if ($splitPayload.excluded_subjects.Count -eq 0) {
            throw "Invalid support-${support} split for seed=${seed}: excluded_subjects is empty"
        }
        if (Test-Path "$output\summary.json") {
            Write-Host "SKIP support=$support seed=$seed"
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
            Write-Host "RUN support=$support seed=$seed attempt=$attempt"
            & $PythonExe @runArgs
            if ($LASTEXITCODE -eq 0) { break }
        }
        if ($LASTEXITCODE -ne 0) {
            throw "Support-curve transfer failed: support=$support seed=$seed"
        }
    }
}

$count = (Get-ChildItem outputs\transfer_ds004504_caueeg_support_curve -Recurse -Filter summary.json).Count
Write-Host "Completed support-curve summaries: $count/30"
