param(
    [string]$PythonExe = "C:\Users\Administrator\.conda\envs\BioFoundation\python.exe",
    [string]$ProjectRoot = "F:\lzy\BioFoundation-main",
    [int[]]$Seeds = @(2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035),
    [int[]]$Caps = @(25, 100, 0),
    [string]$OutputRoot = "outputs\paper_upgrade\training_size"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $ProjectRoot

$domains = @(
    "CAUEEG=CAUEEG_processed_19ch_ad_vs_hc_1s_lead",
    "ds004504=ds004504_processed_19ch_1s_v2",
    "ADSZ=ADSZ_processed_19ch_1s_v3"
)
$targets = @("CAUEEG", "ds004504", "ADSZ")
$variants = @(
    @{ Name = "erm"; Method = "erm"; Ld = 0.0; Lm = 0.0 },
    @{ Name = "erm_margin"; Method = "erm"; Ld = 0.0; Lm = 0.05 },
    @{ Name = "dann"; Method = "dann"; Ld = 0.2; Lm = 0.0 }
)

foreach ($cap in $Caps) {
    $capName = if ($cap -eq 0) { "all" } else { "cap$cap" }
    foreach ($seed in $Seeds) {
        foreach ($variant in $variants) {
            foreach ($target in $targets) {
                if ($target -eq "CAUEEG" -and $cap -in @(100, 0)) {
                    Write-Host "SKIP equivalent cap=$capName target=CAUEEG (same participants as locked cap50)"
                    continue
                }
                $outputDir = Join-Path $OutputRoot "$capName\seed$seed\$($variant.Name)_target_$target"
                $summary = Join-Path $outputDir "lodo_summary.json"
                if (Test-Path -LiteralPath $summary) {
                    Write-Host "SKIP cap=$capName seed=$seed variant=$($variant.Name) target=$target"
                    continue
                }
                Write-Host "RUN cap=$capName seed=$seed variant=$($variant.Name) target=$target"
                & $PythonExe scripts\run_graph_temporal_lodo_tacd.py `
                    --domains $domains `
                    --target $target `
                    --method $variant.Method `
                    --variant-name $variant.Name `
                    --output-dir $outputDir `
                    --epochs 30 --batch-size 32 --lr 0.001 --weight-decay 0.0001 `
                    --patience 8 --max-source-subjects-per-class $cap `
                    --max-segments-per-subject 20 --max-target-segments-per-subject 20 `
                    --lambda-domain $variant.Ld --lambda-specific 0.0 `
                    --lambda-orthogonal 0.0 --lambda-margin $variant.Lm `
                    --lambda-alignment 0.0 --margin-variance-weight 0.1 `
                    --domain-balanced-batches --tune-threshold `
                    --threshold-objective balanced_acc --seed $seed --device cuda
                if ($LASTEXITCODE -ne 0) {
                    throw "Training-size run failed: cap=$capName seed=$seed variant=$($variant.Name) target=$target"
                }
            }
        }
    }
}

Write-Host "Training-size sensitivity jobs finished. cap50 remains in the locked main result directory."
