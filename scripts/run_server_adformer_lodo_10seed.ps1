param(
    [string]$PythonExe = "C:\Users\Administrator\.conda\envs\BioFoundation\python.exe",
    [string]$ProjectRoot = "F:\lzy\BioFoundation-main"
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

$required = @(
    "CAUEEG_processed_19ch_ad_vs_hc_1s_lead",
    "ds004504_processed_19ch_1s_v2",
    "ADSZ_processed_19ch_1s_v3",
    "external\ADformer\models\ADformer.py",
    "models\ADformerDomainAdapter.py",
    "scripts\run_graph_temporal_lodo_tacd.py",
    "scripts\aggregate_margin_alignment.py"
)
foreach ($path in $required) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Missing required path: $path"
    }
}

$domains = @(
    "CAUEEG=CAUEEG_processed_19ch_ad_vs_hc_1s_lead",
    "ds004504=ds004504_processed_19ch_1s_v2",
    "ADSZ=ADSZ_processed_19ch_1s_v3"
)
$seeds = 2026..2035
$outputRoot = "outputs\domain_generalization_adformer"

foreach ($seed in $seeds) {
    $outputDir = Join-Path $outputRoot "seed${seed}\adformer_erm"
    $summary = Join-Path $outputDir "lodo_summary.json"
    if (Test-Path -LiteralPath $summary) {
        Write-Host "SKIP seed=$seed"
        continue
    }

    Write-Host "RUN ADformer source-only LODO seed=$seed"
    & $PythonExe scripts\run_graph_temporal_lodo_tacd.py `
        --domains $domains `
        --method erm `
        --variant-name adformer_erm `
        --backbone adformer `
        --output-dir $outputDir `
        --epochs 30 `
        --batch-size 32 `
        --lr 0.0001 `
        --weight-decay 0.0001 `
        --patience 8 `
        --max-source-subjects-per-class 50 `
        --max-segments-per-subject 20 `
        --max-target-segments-per-subject 20 `
        --adformer-patch-lengths 2 `
        --adformer-up-dims 19 `
        --adformer-d-model 128 `
        --adformer-num-heads 8 `
        --adformer-num-layers 6 `
        --adformer-ff-dim 256 `
        --dropout 0.5 `
        --domain-balanced-batches `
        --tune-threshold `
        --threshold-objective balanced_acc `
        --seed $seed `
        --device cuda

    if ($LASTEXITCODE -ne 0) {
        throw "ADformer run failed: seed=$seed"
    }
}

$completed = (Get-ChildItem -LiteralPath $outputRoot -Recurse -Filter lodo_summary.json).Count
Write-Host "Completed $completed/10 ADformer LODO runs"
if ($completed -eq 10) {
    & $PythonExe scripts\aggregate_margin_alignment.py `
        --results-dir $outputRoot `
        --output-prefix (Join-Path $outputRoot "adformer_lodo_10seed_summary") `
        --expected-seeds 10
    if ($LASTEXITCODE -ne 0) {
        throw "ADformer aggregation failed"
    }
}
