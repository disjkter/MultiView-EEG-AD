param(
    [string]$PythonExe = "C:\Users\Administrator\.conda\envs\BioFoundation\python.exe",
    [string]$ProjectRoot = "F:\lzy\BioFoundation-main",
    [int[]]$WindowSeconds = @(2, 5)
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $ProjectRoot

function Test-CompleteDataset([string]$Directory) {
    $required = @("x_19ch.npy", "y_19ch.npy", "sub_ids_19ch.npy", "summary.json")
    return ($required | Where-Object {
        -not (Test-Path -LiteralPath (Join-Path $Directory $_) -PathType Leaf)
    }).Count -eq 0
}

foreach ($window in $WindowSeconds) {
    $caueegOut = if ($window -eq 5) {
        "CAUEEG_processed_19ch_ad_vs_hc_5s"
    } else {
        "CAUEEG_processed_19ch_${window}s"
    }
    $dsOut = "ds004504_processed_19ch_${window}s"
    $adszOut = "ADSZ_processed_19ch_${window}s"

    if (-not (Test-CompleteDataset $caueegOut)) {
        if ($window -ne 2) {
            throw "Missing complete CAUEEG dataset: $caueegOut"
        }
        & $PythonExe scripts\rewindow_contiguous_npy.py `
            --input-dir CAUEEG_processed_19ch_ad_vs_hc_1s_lead `
            --output-dir $caueegOut --factor 2
        if ($LASTEXITCODE -ne 0) { throw "CAUEEG ${window}s preprocessing failed" }
    }

    if (-not (Test-CompleteDataset $dsOut)) {
        & $PythonExe scripts\preprocess_ds004504.py `
            --bids-root eeg-data --output-dir $dsOut `
            --target-sfreq 128 --window-sec $window --expected-subjects 65
        if ($LASTEXITCODE -ne 0) { throw "ds004504 ${window}s preprocessing failed" }
    }

    if (-not (Test-CompleteDataset $adszOut)) {
        & $PythonExe scripts\preprocess_adsz_windowed.py `
            --data-dir ADSZ\dataset\alzheimer --output-dir $adszOut `
            --fs 128 --window-sec $window --stride-sec $window --expected-subjects 24
        if ($LASTEXITCODE -ne 0) { throw "ADSZ ${window}s preprocessing failed" }
    }
}

Write-Host "Prepared 2 s and 5 s datasets from recording-level sources."
