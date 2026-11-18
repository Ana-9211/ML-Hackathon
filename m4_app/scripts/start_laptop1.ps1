# From m4_app: powershell -ExecutionPolicy Bypass -File .\scripts\start_laptop1.ps1 -Mode trained
param([ValidateSet("trained", "demo")][string]$Mode = "demo")
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))
$env:M4_MODE = $Mode
Write-Host "PathSnap API on Laptop 1: mode=$Mode, port=8000" -ForegroundColor Cyan
if ($Mode -eq "demo") {
    Write-Host "Using included synthetic trips from .\demo_data. This is for wiring/UI checks only." -ForegroundColor Yellow
} else {
    Write-Host "Using private evaluator data from M4_EVAL_DIR (default: .\private_evaluator)." -ForegroundColor Yellow
    Write-Host "Required per trip: matching .track, .nodes and .arcs files. See README.md." -ForegroundColor Yellow
}
Write-Host "Use http://<LAPTOP1_IPV4>:8000 on Laptop 2. See README.md." -ForegroundColor Yellow
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --access-log
