param(
    [string]$PythonExe = "C:\Users\Administrator\.conda\envs\BioFoundation\python.exe",
    [string]$ProjectRoot = "F:\lzy\BioFoundation-main"
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

$variants = @(
    @{ Name = "erm";         Method = "erm";  Ld = 0.0; Ls = 0.0; Lo = 0.0;  Lm = 0.0 },
    @{ Name = "erm_margin";  Method = "erm";  Ld = 0.0; Ls = 0.0; Lo = 0.0;  Lm = 0.05 },
    @{ Name = "dann";        Method = "dann"; Ld = 0.2; Ls = 0.0; Lo = 0.0;  Lm = 0.0 },
    @{ Name = "tacd";        Method = "tacd"; Ld = 0.2; Ls = 0.1; Lo = 0.01; Lm = 0.0 },
    @{ Name = "tacd_margin"; Method = "tacd"; Ld = 0.2; Ls = 0.1; Lo = 0.01; Lm = 0.05 }
)
$seeds = 2026..2035
$outputRoot = "outputs\adfsu_confirmatory_10seed"

foreach ($seed in $seeds) {
    foreach ($variant in $variants) {
        $outputDir = Join-Path $outputRoot "seed${seed}\$($variant.Name)"
        $summary = Join-Path $outputDir "lodo_summary.json"
        if (Test-Path -LiteralPath $summary) {
            Write-Host "SKIP seed=$seed variant=$($variant.Name)"
            continue
        }

        Write-Host "RUN ADFSU confirmation seed=$seed variant=$($variant.Name)"
        & $PythonExe scripts\run_graph_temporal_lodo_tacd.py `
            --domains `
                CAUEEG=CAUEEG_processed_19ch_ad_vs_hc_1s_lead `
                ds004504=ds004504_processed_19ch_1s_v2 `
                ADSZ=ADSZ_processed_19ch_1s_v3 `
                ADFSU=ADFSU_processed_19ch_1s_v2 `
            --target ADFSU `
            --method $variant.Method `
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
            --tune-threshold `
            --threshold-objective balanced_acc `
            --seed $seed `
            --device cuda

        if ($LASTEXITCODE -ne 0) {
            throw "ADFSU confirmation failed: seed=$seed variant=$($variant.Name)"
        }
    }
}

$completed = (
    Get-ChildItem -LiteralPath $outputRoot -Recurse -Filter lodo_summary.json
).Count
Write-Host "Completed $completed/50 ADFSU confirmation runs"
if ($completed -eq ($seeds.Count * $variants.Count)) {
    & $PythonExe scripts\aggregate_margin_alignment.py `
        --results-dir $outputRoot `
        --output-prefix (Join-Path $outputRoot "adfsu_confirmatory_10seed_summary") `
        --expected-seeds 10
    if ($LASTEXITCODE -ne 0) {
        throw "ADFSU confirmation aggregation failed"
    }
}
