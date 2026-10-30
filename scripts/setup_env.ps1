# CrashTwin environment setup for a NEW Windows machine.
# Creates <project>\.venv (Python 3.10), installs the pinned packages, downloads the YOLO weights
# (checked against their SHA-256) and runs the environment check. Safe to run again: it skips what exists.
#
# Run from anywhere:
#   powershell -ExecutionPolicy Bypass -File <project>\crashtwin\scripts\setup_env.ps1
$ErrorActionPreference = "Stop"
$proj = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)   # folder that contains crashtwin\
Set-Location $proj
Write-Host "Project folder: $proj"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating .venv with Python 3.10 ..."
    py -3.10 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Python 3.10 not found. Install Python 3.10.11 (python.org) and tick 'py launcher'." }
}
$py = Join-Path $proj ".venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install -r crashtwin\requirements.txt
if ($LASTEXITCODE -ne 0) { throw "pip install failed (see above)" }

$w = Join-Path $proj "models\yolo\yolo11s.pt"
if (-not (Test-Path $w)) {
    New-Item -ItemType Directory -Force (Split-Path $w) | Out-Null
    Write-Host "Downloading YOLO11s weights ..."
    Invoke-WebRequest "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt" -OutFile "$w.partial"
    Move-Item "$w.partial" $w
}
$sha = (Get-FileHash $w -Algorithm SHA256).Hash.ToLower()
if ($sha -ne "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5") { throw "yolo11s.pt SHA-256 mismatch: $sha" }
Write-Host "YOLO weights OK"

& $py crashtwin\scripts\check_env.py
Write-Host "`nNext: set CARLA_ROOT if CARLA is not in the lab-PC location, e.g.  setx CARLA_ROOT C:\CARLA_0.9.15\WindowsNoEditor"
