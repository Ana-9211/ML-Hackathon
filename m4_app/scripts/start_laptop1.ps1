# Laptop 1 (API). From m4_app:  powershell -ExecutionPolicy Bypass -File .\scripts\start_laptop1.ps1
#   -Mode trained   (default) full pipeline: M1 features -> M2 tuned Gradient Boosting -> M3 HMM/Viterbi
#   -Mode demo      synthetic demo_data trips with a geometry-only baseline (wiring check only)
#   -EvalDir <path> folder of evaluator trips (default: <repo>\data\evaluator_test)
param(
    [ValidateSet("trained", "demo")][string]$Mode = "trained",
    [string]$EvalDir = ""
)
$ErrorActionPreference = "Stop"
$AppDir  = Resolve-Path (Join-Path $PSScriptRoot "..")
$RepoDir = Resolve-Path (Join-Path $AppDir "..")
Set-Location $AppDir

$env:M4_MODE = $Mode
$env:PYTHONPATH = "$RepoDir;$AppDir"
$env:M4_TEAM_PREDICTOR = "m3_hmm.inference:predict_trip"
if ($EvalDir -ne "") { $env:M4_EVAL_DIR = (Resolve-Path $EvalDir) } else { $env:M4_EVAL_DIR = Join-Path $RepoDir "data\evaluator_test" }

Write-Host "PathSnap API on Laptop 1  (port 8000)" -ForegroundColor Cyan
Write-Host "  mode       : $Mode"
if ($Mode -eq "trained") {
    Write-Host "  predictor  : $env:M4_TEAM_PREDICTOR"
    Write-Host "  trips from : $env:M4_EVAL_DIR"
} else {
    Write-Host "  trips from : demo_data (synthetic, wiring check only)" -ForegroundColor Yellow
}
Write-Host "  Laptop 2 connects to http://<LAPTOP1_IPV4>:8000  (find it with: ipconfig)" -ForegroundColor Yellow
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --access-log
