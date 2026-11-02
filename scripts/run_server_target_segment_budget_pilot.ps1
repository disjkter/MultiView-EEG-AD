param(
    [string]$PythonExe = "",
    [string]$Config = "configs\target_segment_budget_pilot.yaml"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not $PythonExe) {
    $PythonExe = if ($env:BIOFOUNDATION_PYTHON) { $env:BIOFOUNDATION_PYTHON } else { "python" }
}
$env:PYTHONUNBUFFERED = "1"
$resultRoot = "outputs\jbhi_revision\target_segment_budget_pilot"
$logPath = "$resultRoot\server_progress.log"
New-Item -ItemType Directory -Force -Path $resultRoot | Out-Null

& $PythonExe "scripts\run_target_segment_budget_pilot.py" `
    "--config" $Config "--python-exe" $PythonExe 2>&1 |
    ForEach-Object { $_.ToString() } |
    Tee-Object -FilePath $logPath -Append
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) { throw "Target segment-budget pilot failed with exit code $exitCode" }

