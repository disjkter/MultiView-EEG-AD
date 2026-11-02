param(
    [string]$PythonExe = "C:\Users\Administrator\.conda\envs\BioFoundation\python.exe",
    [string]$ProjectRoot = "F:\lzy\BioFoundation-main",
    [int[]]$Seeds = @(2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035),
    [string]$OutputRoot = "outputs\erm_backbone_breadth"
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot
$backbones = @("shallow_convnet", "eeg_conformer")

foreach ($seed in $Seeds) {
    foreach ($backbone in $backbones) {
        $outputDir = Join-Path $OutputRoot "$backbone\seed${seed}\erm"
        $summary = Join-Path $outputDir "lodo_summary.json"
        if (Test-Path -LiteralPath $summary) {
            Write-Host "SKIP backbone=$backbone seed=$seed"
            continue
        }

        Write-Host "RUN backbone=$backbone seed=$seed"
        & $PythonExe scripts\run_graph_temporal_lodo_tacd.py `
            --domains `
                CAUEEG=CAUEEG_processed_19ch_ad_vs_hc_1s_lead `
                ds004504=ds004504_processed_19ch_1s_v2 `
                ADSZ=ADSZ_processed_19ch_1s_v3 `
            --backbone $backbone `
            --method erm `
            --variant-name erm `
            --output-dir $outputDir `
            --epochs 30 `
            --batch-size 32 `
            --lr 0.001 `
            --weight-decay 0.0001 `
            --patience 8 `
            --max-source-subjects-per-class 50 `
            --max-segments-per-subject 20 `
            --max-target-segments-per-subject 20 `
            --lambda-domain 0 `
            --lambda-specific 0 `
            --lambda-orthogonal 0 `
            --lambda-margin 0 `
            --lambda-alignment 0 `
            --domain-balanced-batches `
            --tune-threshold `
            --threshold-objective balanced_acc `
            --seed $seed `
            --device cuda

        if ($LASTEXITCODE -ne 0) {
            throw "ERM backbone run failed: backbone=$backbone seed=$seed"
        }
    }
}

$expected = $Seeds.Count * $backbones.Count
$completed = 0
foreach ($seed in $Seeds) {
    foreach ($backbone in $backbones) {
        $summary = Join-Path $OutputRoot "$backbone\seed${seed}\erm\lodo_summary.json"
        if (Test-Path -LiteralPath $summary) { $completed += 1 }
    }
}
Write-Host "Completed $completed/$expected requested ERM backbone runs"

if ($completed -eq $expected) {
    & $PythonExe scripts\aggregate_erm_backbone_lodo.py `
        --results-dir $OutputRoot `
        --output-prefix (Join-Path $OutputRoot "erm_backbone_breadth_10seed") `
        --expected-seeds $Seeds.Count
    if ($LASTEXITCODE -ne 0) { throw "ERM backbone aggregation failed" }
}
