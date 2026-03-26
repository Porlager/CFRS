$ErrorActionPreference = "Stop"

Write-Host "[CFRS] Running dashboard e2e tests..." -ForegroundColor Cyan

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    Write-Error "Python launcher (py) not found. Please install Python 3 first."
    exit 1
}

$depsCheck = py -3 -c "import flask, requests, pytest" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[CFRS] Installing test dependencies (flask, requests, pytest)..." -ForegroundColor Yellow
    py -3 -m pip install flask requests pytest
}

py -3 -m pytest
