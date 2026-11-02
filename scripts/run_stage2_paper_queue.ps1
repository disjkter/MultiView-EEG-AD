$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = "python"
$logPath = Join-Path $projectRoot "outputs\stage2_paper_queue.log"

function Write-Log {
    param(
        [string]$Message
    )
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -LiteralPath $logPath -Value $line -Encoding UTF8
    Write-Host $line
}

function Run-Experiment {
    param(
        [string]$Label,
        [string]$OutputDir,
        [string[]]$Args
    )

    $summaryPath = Join-Path $projectRoot ($OutputDir + "\summary.json")
    if (Test-Path -LiteralPath $summaryPath) {
        Write-Log "SKIP $Label (summary exists: $summaryPath)"
        return
    }

    Write-Log "START $Label"
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $pythonExe
    $psi.Arguments = [string]::Join(" ", $Args)
    $psi.WorkingDirectory = $projectRoot
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true

    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $psi
    [void]$proc.Start()
    $stdout = $proc.StandardOutput.ReadToEnd()
    $stderr = $proc.StandardError.ReadToEnd()
    $proc.WaitForExit()

    if ($stdout) {
        Add-Content -LiteralPath $logPath -Value $stdout -Encoding UTF8
    }
    if ($stderr) {
        Add-Content -LiteralPath $logPath -Value $stderr -Encoding UTF8
    }

    if ($proc.ExitCode -ne 0) {
        Write-Log "FAIL $Label (exit code $($proc.ExitCode))"
        throw "Experiment failed: $Label"
    }

    Write-Log "DONE $Label"
}

New-Item -ItemType Directory -Force -Path (Join-Path $projectRoot "outputs") | Out-Null
Write-Log "QUEUE START"

$commonArgs = @(
    "scripts\run_dual_stream_subject_split.py",
    "--epochs", "20",
    "--batch-size", "32",
    "--lr", "1e-3",
    "--weight-decay", "1e-4",
    "--train-ratio", "0.7",
    "--val-ratio", "0.1",
    "--patience", "5",
    "--use-ea",
    "--max-ea-samples", "5000",
    "--max-train-segments-per-subject", "250",
    "--max-val-segments-per-subject", "250",
    "--max-test-segments-per-subject", "400",
    "--tune-threshold",
    "--threshold-objective", "balanced_acc",
    "--selection-metric", "auc"
)

$experiments = @(
    @{
        Label = "ds004504 random seed2029"
        OutputDir = "outputs\subject_split_ds004504_tage_randominit_1s_lead_seed2029"
        Args = @("--data-dir", "ds004504_processed_19ch_1s_lead", "--no-checkpoint", "--output-dir", "outputs\subject_split_ds004504_tage_randominit_1s_lead_seed2029", "--seed", "2029")
    },
    @{
        Label = "ds004504 random seed2030"
        OutputDir = "outputs\subject_split_ds004504_tage_randominit_1s_lead_seed2030"
        Args = @("--data-dir", "ds004504_processed_19ch_1s_lead", "--no-checkpoint", "--output-dir", "outputs\subject_split_ds004504_tage_randominit_1s_lead_seed2030", "--seed", "2030")
    },
    @{
        Label = "ds004504 multi-source seed2029"
        OutputDir = "outputs\subject_split_ds004504_tage_timefreq_consistency_adssl_1s_lead_seed2029"
        Args = @("--data-dir", "ds004504_processed_19ch_1s_lead", "--checkpoint", "checkpoints\tage_timefreq_consistency_adssl_1s_lead.pth", "--output-dir", "outputs\subject_split_ds004504_tage_timefreq_consistency_adssl_1s_lead_seed2029", "--seed", "2029")
    },
    @{
        Label = "ds004504 multi-source seed2030"
        OutputDir = "outputs\subject_split_ds004504_tage_timefreq_consistency_adssl_1s_lead_seed2030"
        Args = @("--data-dir", "ds004504_processed_19ch_1s_lead", "--checkpoint", "checkpoints\tage_timefreq_consistency_adssl_1s_lead.pth", "--output-dir", "outputs\subject_split_ds004504_tage_timefreq_consistency_adssl_1s_lead_seed2030", "--seed", "2030")
    },
    @{
        Label = "ds004504 auditory40hz seed2029"
        OutputDir = "outputs\subject_split_ds004504_tage_timefreq_consistency_auditory40hz_1s_lead_seed2029"
        Args = @("--data-dir", "ds004504_processed_19ch_1s_lead", "--checkpoint", "checkpoints\tage_timefreq_consistency_auditory40hz_1s_lead.pth", "--output-dir", "outputs\subject_split_ds004504_tage_timefreq_consistency_auditory40hz_1s_lead_seed2029", "--seed", "2029")
    },
    @{
        Label = "ds004504 auditory40hz seed2030"
        OutputDir = "outputs\subject_split_ds004504_tage_timefreq_consistency_auditory40hz_1s_lead_seed2030"
        Args = @("--data-dir", "ds004504_processed_19ch_1s_lead", "--checkpoint", "checkpoints\tage_timefreq_consistency_auditory40hz_1s_lead.pth", "--output-dir", "outputs\subject_split_ds004504_tage_timefreq_consistency_auditory40hz_1s_lead_seed2030", "--seed", "2030")
    },
    @{
        Label = "CAUEEG 1s multi-source seed2026"
        OutputDir = "outputs\subject_split_caueeg_tage_timefreq_consistency_adssl_1s_lead_seed2026"
        Args = @("--data-dir", "CAUEEG_processed_19ch_ad_vs_hc_1s_lead", "--checkpoint", "checkpoints\tage_timefreq_consistency_adssl_1s_lead.pth", "--output-dir", "outputs\subject_split_caueeg_tage_timefreq_consistency_adssl_1s_lead_seed2026", "--seed", "2026")
    },
    @{
        Label = "CAUEEG 1s multi-source seed2027"
        OutputDir = "outputs\subject_split_caueeg_tage_timefreq_consistency_adssl_1s_lead_seed2027"
        Args = @("--data-dir", "CAUEEG_processed_19ch_ad_vs_hc_1s_lead", "--checkpoint", "checkpoints\tage_timefreq_consistency_adssl_1s_lead.pth", "--output-dir", "outputs\subject_split_caueeg_tage_timefreq_consistency_adssl_1s_lead_seed2027", "--seed", "2027")
    },
    @{
        Label = "CAUEEG 1s multi-source seed2028"
        OutputDir = "outputs\subject_split_caueeg_tage_timefreq_consistency_adssl_1s_lead_seed2028"
        Args = @("--data-dir", "CAUEEG_processed_19ch_ad_vs_hc_1s_lead", "--checkpoint", "checkpoints\tage_timefreq_consistency_adssl_1s_lead.pth", "--output-dir", "outputs\subject_split_caueeg_tage_timefreq_consistency_adssl_1s_lead_seed2028", "--seed", "2028")
    },
    @{
        Label = "CAUEEG 1s multi-source seed2029"
        OutputDir = "outputs\subject_split_caueeg_tage_timefreq_consistency_adssl_1s_lead_seed2029"
        Args = @("--data-dir", "CAUEEG_processed_19ch_ad_vs_hc_1s_lead", "--checkpoint", "checkpoints\tage_timefreq_consistency_adssl_1s_lead.pth", "--output-dir", "outputs\subject_split_caueeg_tage_timefreq_consistency_adssl_1s_lead_seed2029", "--seed", "2029")
    },
    @{
        Label = "CAUEEG 1s multi-source seed2030"
        OutputDir = "outputs\subject_split_caueeg_tage_timefreq_consistency_adssl_1s_lead_seed2030"
        Args = @("--data-dir", "CAUEEG_processed_19ch_ad_vs_hc_1s_lead", "--checkpoint", "checkpoints\tage_timefreq_consistency_adssl_1s_lead.pth", "--output-dir", "outputs\subject_split_caueeg_tage_timefreq_consistency_adssl_1s_lead_seed2030", "--seed", "2030")
    }
)

foreach ($exp in $experiments) {
    $args = @() + $commonArgs + $exp.Args
    Run-Experiment -Label $exp.Label -OutputDir $exp.OutputDir -Args $args
}

Write-Log "QUEUE DONE"
