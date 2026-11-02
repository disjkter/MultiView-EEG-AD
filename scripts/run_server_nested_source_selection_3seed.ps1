param(
    [string]$PythonExe = "C:\Users\Administrator\.conda\envs\BioFoundation\python.exe",
    [string]$ProjectRoot = "F:\lzy\BioFoundation-main"
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

$domains = @(
    "CAUEEG=CAUEEG_processed_19ch_ad_vs_hc_1s_lead",
    "ds004504=ds004504_processed_19ch_1s_v2",
    "ADSZ=ADSZ_processed_19ch_1s_v3"
)
$variants = @(
    @{ Name = "erm_nested"; Method = "erm"; Ld = 0.0; Ls = 0.0; Lo = 0.0; Lm = 0.0 },
    @{ Name = "tacd_margin_nested"; Method = "tacd"; Ld = 0.2; Ls = 0.1; Lo = 0.01; Lm = 0.05 }
)
$seeds = 2026..2028
$outputRoot = "outputs\nested_source_selection_3seed"

foreach ($seed in $seeds) {
    foreach ($variant in $variants) {
        $outputDir = Join-Path $outputRoot "seed${seed}\$($variant.Name)"
        $summary = Join-Path $outputDir "lodo_summary.json"
        if (Test-Path -LiteralPath $summary) {
            Write-Host "SKIP seed=$seed variant=$($variant.Name)"
            continue
        }
        Write-Host "RUN nested source selection seed=$seed variant=$($variant.Name)"
        & $PythonExe scripts\run_graph_temporal_lodo_tacd.py `
            --domains $domains `
            --method $variant.Method `
            --variant-name $variant.Name `
            --backbone graph_temporal `
            --output-dir $outputDir `
            --epochs 30 `
            --batch-size 32 `
            --lr 0.001 `
            --weight-decay 0.0001 `
            --patience 8 `
            --max-source-subjects-per-class 50 `
            --max-segments-per-subject 20 `
            --max-target-segments-per-subject 20 `
            --lambda-domain $variant.Ld `
            --lambda-specific $variant.Ls `
            --lambda-orthogonal $variant.Lo `
            --lambda-margin $variant.Lm `
            --margin-variance-weight 0.1 `
            --domain-balanced-batches `
            --separate-selection-splits `
            --checkpoint-val-ratio 0.15 `
            --calibration-val-ratio 0.15 `
            --tune-threshold `
            --threshold-objective balanced_acc `
            --seed $seed `
            --device cuda
        if ($LASTEXITCODE -ne 0) {
            throw "Nested source-selection run failed: seed=$seed variant=$($variant.Name)"
        }
    }
}

$completed = (Get-ChildItem -LiteralPath $outputRoot -Recurse -Filter lodo_summary.json).Count
Write-Host "Completed $completed/6 nested source-selection runs"
if ($completed -eq 6) {
    & $PythonExe scripts\aggregate_margin_alignment.py `
        --results-dir $outputRoot `
        --output-prefix (Join-Path $outputRoot "nested_source_selection_3seed_summary") `
        --expected-seeds 3
}
