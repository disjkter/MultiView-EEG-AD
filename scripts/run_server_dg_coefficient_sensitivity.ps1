param(
    [string]$PythonExe = "C:\Users\Administrator\.conda\envs\BioFoundation\python.exe",
    [string]$ProjectRoot = "F:\lzy\BioFoundation-main",
    [int[]]$Seeds = @(2026, 2027, 2028),
    [string]$OutputRoot = "outputs\paper_upgrade\dg_coefficient_sensitivity"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $ProjectRoot

$domains = @(
    "CAUEEG=CAUEEG_processed_19ch_ad_vs_hc_1s_lead",
    "ds004504=ds004504_processed_19ch_1s_v2",
    "ADSZ=ADSZ_processed_19ch_1s_v3"
)
$variants = @(
    @{ Name = "dann_ld005"; Method = "dann"; Ld = 0.05; Ls = 0.0; Lo = 0.0; La = 0.0 },
    @{ Name = "dann_ld020"; Method = "dann"; Ld = 0.20; Ls = 0.0; Lo = 0.0; La = 0.0 },
    @{ Name = "dann_ld080"; Method = "dann"; Ld = 0.80; Ls = 0.0; Lo = 0.0; La = 0.0 },
    @{ Name = "coral_la01"; Method = "coral"; Ld = 0.0; Ls = 0.0; Lo = 0.0; La = 0.1 },
    @{ Name = "coral_la1"; Method = "coral"; Ld = 0.0; Ls = 0.0; Lo = 0.0; La = 1.0 },
    @{ Name = "coral_la10"; Method = "coral"; Ld = 0.0; Ls = 0.0; Lo = 0.0; La = 10.0 },
    @{ Name = "mmd_la01"; Method = "mmd"; Ld = 0.0; Ls = 0.0; Lo = 0.0; La = 0.1 },
    @{ Name = "mmd_la1"; Method = "mmd"; Ld = 0.0; Ls = 0.0; Lo = 0.0; La = 1.0 },
    @{ Name = "mmd_la10"; Method = "mmd"; Ld = 0.0; Ls = 0.0; Lo = 0.0; La = 10.0 },
    @{ Name = "tacd_scale025"; Method = "tacd"; Ld = 0.05; Ls = 0.025; Lo = 0.0025; La = 0.0 },
    @{ Name = "tacd_scale100"; Method = "tacd"; Ld = 0.20; Ls = 0.100; Lo = 0.0100; La = 0.0 },
    @{ Name = "tacd_scale400"; Method = "tacd"; Ld = 0.80; Ls = 0.400; Lo = 0.0400; La = 0.0 }
)

foreach ($seed in $Seeds) {
    foreach ($variant in $variants) {
        $outputDir = Join-Path $OutputRoot "seed${seed}\$($variant.Name)"
        $summary = Join-Path $outputDir "lodo_summary.json"
        if (Test-Path -LiteralPath $summary) {
            Write-Host "SKIP seed=$seed variant=$($variant.Name)"
            continue
        }
        Write-Host "RUN coefficient sensitivity seed=$seed variant=$($variant.Name)"
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
            --lambda-margin 0.0 `
            --lambda-alignment $variant.La `
            --margin-variance-weight 0.1 `
            --domain-balanced-batches `
            --tune-threshold `
            --threshold-objective balanced_acc `
            --seed $seed `
            --device cuda
        if ($LASTEXITCODE -ne 0) {
            throw "DG coefficient run failed: seed=$seed variant=$($variant.Name)"
        }
    }
}

$completed = (Get-ChildItem -LiteralPath $OutputRoot -Recurse -Filter lodo_summary.json).Count
$expected = $Seeds.Count * $variants.Count
Write-Host "Completed $completed/$expected DG coefficient runs"
