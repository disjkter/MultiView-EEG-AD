param(
    [string]$PythonExe = "",
    [string]$Config = "configs\staged_ds004504_to_caueeg.yaml"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not $PythonExe) {
    $PythonExe = if ($env:BIOFOUNDATION_PYTHON) { $env:BIOFOUNDATION_PYTHON } else { "python" }
}
$env:PYTHONUNBUFFERED = "1"
$logDir = "outputs\jbhi_revision\staged_ds004504_to_caueeg"
$logPath = "$logDir\server_progress.log"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

& $PythonExe "scripts\run_staged_transfer_support_curve.py" `
    "--config" $Config "--python-exe" $PythonExe 2>&1 |
    ForEach-Object { $_.ToString() } |
    Tee-Object -FilePath $logPath -Append
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) { throw "Staged reverse-transfer experiment failed with exit code $exitCode" }

