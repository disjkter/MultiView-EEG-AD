param(
    [string]$PythonExe = "C:\Users\Administrator\.conda\envs\BioFoundation\python.exe",
    [string]$ProjectRoot = "F:\lzy\BioFoundation-main",
    [int[]]$Seeds = @(2026, 2027, 2028),
    [int[]]$WindowSeconds = @(2, 5),
    [string]$OutputRoot = "outputs\paper_upgrade\window_length"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $ProjectRoot
$targets = @("CAUEEG", "ds004504", "ADSZ")
$variants = @(
    @{ Name = "erm"; Method = "erm"; Ld = 0.0; Ls = 0.0; Lo = 0.0; Lm = 0.0 },
    @{ Name = "tacd_margin"; Method = "tacd"; Ld = 0.2; Ls = 0.1; Lo = 0.01; Lm = 0.05 }
)

foreach ($window in $WindowSeconds) {
    $caueegPath = if ($window -eq 5) {
        "CAUEEG_processed_19ch_ad_vs_hc_5s"
    } else {
        "CAUEEG_processed_19ch_${window}s"
    }
    $domains = @(
        "CAUEEG=$caueegPath",
        "ds004504=ds004504_processed_19ch_${window}s",
        "ADSZ=ADSZ_processed_19ch_${window}s"
    )
    foreach ($spec in $domains) {
        $path = $spec.Split("=", 2)[1]
        if (-not (Test-Path -LiteralPath $path)) {
            throw "Missing preprocessed window-length dataset: $path"
        }
    }
    foreach ($seed in $Seeds) {
        foreach ($variant in $variants) {
            foreach ($target in $targets) {
                $outputDir = Join-Path $OutputRoot "${window}s\seed$seed\$($variant.Name)_target_$target"
                if (Test-Path -LiteralPath (Join-Path $outputDir "lodo_summary.json")) {
                    Write-Host "SKIP window=${window}s seed=$seed variant=$($variant.Name) target=$target"
                    continue
                }
                & $PythonExe scripts\run_graph_temporal_lodo_tacd.py `
                    --domains $domains --target $target --method $variant.Method `
                    --variant-name $variant.Name `
                    --output-dir $outputDir --epochs 30 --batch-size 32 `
                    --lr 0.001 --weight-decay 0.0001 --patience 8 `
                    --max-source-subjects-per-class 50 --max-segments-per-subject 20 `
                    --max-target-segments-per-subject 20 --lambda-domain $variant.Ld `
                    --lambda-specific $variant.Ls --lambda-orthogonal $variant.Lo `
                    --lambda-margin $variant.Lm --lambda-alignment 0.0 `
                    --domain-balanced-batches --tune-threshold `
                    --threshold-objective balanced_acc --seed $seed --device cuda
                if ($LASTEXITCODE -ne 0) {
                    throw "Window run failed: window=${window}s seed=$seed variant=$($variant.Name) target=$target"
                }
            }
        }
    }
}

$completed = (Get-ChildItem -LiteralPath $OutputRoot -Recurse -Filter lodo_summary.json).Count
$expected = $WindowSeconds.Count * $Seeds.Count * $variants.Count * $targets.Count
Write-Host "Completed $completed/$expected window-length sensitivity runs"
