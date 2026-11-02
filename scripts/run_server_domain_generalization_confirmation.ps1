param(
    [string]$PythonExe = "C:\Users\Administrator\.conda\envs\BioFoundation\python.exe",
    [string]$ProjectRoot = "F:\lzy\BioFoundation-main",
    [int[]]$Seeds = @(2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035),
    [string]$OutputRoot = "outputs\domain_generalization_adfsu_confirmation"
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

$variants = @(
    @{ Name = "erm";         Method = "erm";   Ld = 0.0; Ls = 0.0; Lo = 0.0;  Lm = 0.0;  La = 0.0 },
    @{ Name = "erm_margin";  Method = "erm";   Ld = 0.0; Ls = 0.0; Lo = 0.0;  Lm = 0.05; La = 0.0 },
    @{ Name = "dann";        Method = "dann";  Ld = 0.2; Ls = 0.0; Lo = 0.0;  Lm = 0.0;  La = 0.0 },
    @{ Name = "coral";       Method = "coral"; Ld = 0.0; Ls = 0.0; Lo = 0.0;  Lm = 0.0;  La = 1.0 },
    @{ Name = "mmd";         Method = "mmd";   Ld = 0.0; Ls = 0.0; Lo = 0.0;  Lm = 0.0;  La = 1.0 },
    @{ Name = "tacd";        Method = "tacd";  Ld = 0.2; Ls = 0.1; Lo = 0.01; Lm = 0.0;  La = 0.0 },
    @{ Name = "tacd_margin"; Method = "tacd";  Ld = 0.2; Ls = 0.1; Lo = 0.01; Lm = 0.05; La = 0.0 }
)

foreach ($seed in $Seeds) {
    foreach ($variant in $variants) {
        $outputDir = Join-Path $OutputRoot "seed${seed}\$($variant.Name)"
        $summary = Join-Path $outputDir "lodo_summary.json"
        if (Test-Path -LiteralPath $summary) {
            Write-Host "SKIP confirmation seed=$seed variant=$($variant.Name)"
            continue
        }

        Write-Host "RUN locked ADFSU confirmation seed=$seed variant=$($variant.Name)"
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
            --lambda-alignment $variant.La `
            --margin-variance-weight 0.1 `
            --domain-balanced-batches `
            --tune-threshold `
            --threshold-objective balanced_acc `
            --seed $seed `
            --device cuda

        if ($LASTEXITCODE -ne 0) {
            throw "ADFSU confirmation failed: seed=$seed variant=$($variant.Name)"
        }
    }
}

$expected = $Seeds.Count * $variants.Count
$completed = 0
foreach ($seed in $Seeds) {
    foreach ($variant in $variants) {
        $summary = Join-Path $OutputRoot "seed${seed}\$($variant.Name)\lodo_summary.json"
        if (Test-Path -LiteralPath $summary) {
            $completed += 1
        }
    }
}
Write-Host "Completed $completed/$expected requested confirmation runs"
if ($completed -eq $expected) {
    & $PythonExe scripts\aggregate_margin_alignment.py `
        --results-dir $OutputRoot `
        --output-prefix (Join-Path $OutputRoot "adfsu_confirmation_summary_$($Seeds.Count)seed") `
        --expected-seeds $Seeds.Count
    if ($LASTEXITCODE -ne 0) { throw "Confirmation aggregation failed" }

    & $PythonExe scripts\analyze_cross_cohort_calibration.py `
        --results "confirmation=$OutputRoot" `
        --output-prefix (Join-Path $OutputRoot "calibration_$($Seeds.Count)seed") `
        --expected-seeds $Seeds.Count
    if ($LASTEXITCODE -ne 0) { throw "Confirmation calibration analysis failed" }
}
