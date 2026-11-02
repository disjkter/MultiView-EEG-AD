param(
    [string]$PythonExe = "C:\Users\Administrator\.conda\envs\BioFoundation\python.exe",
    [string]$ProjectRoot = "F:\lzy\BioFoundation-main"
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

$required = @(
    "CAUEEG_processed_19ch_ad_vs_hc_1s_lead",
    "ds004504_processed_19ch_1s_v2",
    "ADFSU_processed_19ch_1s_dedup",
    "scripts\run_graph_temporal_lodo_tacd.py",
    "scripts\aggregate_margin_alignment.py"
)
foreach ($path in $required) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing required path: $path"
    }
}

# ADSZ is intentionally excluded because it contains recordings also present in ADFSU.
$domains = @(
    "CAUEEG=CAUEEG_processed_19ch_ad_vs_hc_1s_lead",
    "ds004504=ds004504_processed_19ch_1s_v2",
    "ADFSU=ADFSU_processed_19ch_1s_dedup"
)
if (($domains -join " ") -match "ADSZ") {
    throw "Protocol violation: ADSZ must not be a source in the ADFSU overlap-controlled analysis"
}

$variants = @(
    @{ Name = "erm";         Method = "erm";  Ld = 0.0; Ls = 0.0; Lo = 0.0;  Lm = 0.0 },
    @{ Name = "erm_margin";  Method = "erm";  Ld = 0.0; Ls = 0.0; Lo = 0.0;  Lm = 0.05 },
    @{ Name = "dann";        Method = "dann"; Ld = 0.2; Ls = 0.0; Lo = 0.0;  Lm = 0.0 },
    @{ Name = "tacd";        Method = "tacd"; Ld = 0.2; Ls = 0.1; Lo = 0.01; Lm = 0.0 },
    @{ Name = "tacd_margin"; Method = "tacd"; Ld = 0.2; Ls = 0.1; Lo = 0.01; Lm = 0.05 }
)
$seeds = 2026..2035
$outputRoot = "outputs\adfsu_overlap_controlled_10seed"

foreach ($seed in $seeds) {
    foreach ($variant in $variants) {
        $outputDir = Join-Path $outputRoot "seed${seed}\$($variant.Name)"
        $summary = Join-Path $outputDir "lodo_summary.json"
        if (Test-Path -LiteralPath $summary) {
            Write-Host "SKIP seed=$seed variant=$($variant.Name)"
            continue
        }

        Write-Host "RUN overlap-controlled ADFSU sensitivity seed=$seed variant=$($variant.Name)"
        & $PythonExe scripts\run_graph_temporal_lodo_tacd.py `
            --domains $domains `
            --target ADFSU `
            --method $variant.Method `
            --variant-name $variant.Name `
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
            --tune-threshold `
            --threshold-objective balanced_acc `
            --seed $seed `
            --device cuda

        if ($LASTEXITCODE -ne 0) {
            throw "Overlap-controlled ADFSU run failed: seed=$seed variant=$($variant.Name)"
        }
    }
}

$completed = (
    Get-ChildItem -LiteralPath $outputRoot -Recurse -Filter lodo_summary.json
).Count
Write-Host "Completed $completed/50 overlap-controlled ADFSU sensitivity runs"
if ($completed -eq ($seeds.Count * $variants.Count)) {
    & $PythonExe scripts\aggregate_margin_alignment.py `
        --results-dir $outputRoot `
        --output-prefix (Join-Path $outputRoot "adfsu_overlap_controlled_10seed_summary") `
        --expected-seeds 10
    if ($LASTEXITCODE -ne 0) {
        throw "Overlap-controlled ADFSU aggregation failed"
    }
}
