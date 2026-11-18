# From m4_app: powershell -ExecutionPolicy Bypass -File .\scripts\start_laptop2.ps1
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))
Write-Host "PathSnap UI on Laptop 2: http://localhost:5500" -ForegroundColor Cyan
python -m http.server 5500 --directory frontend --bind 0.0.0.0
