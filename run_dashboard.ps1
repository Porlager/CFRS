param(
    [switch]$DashboardOnly
)

$ErrorActionPreference = "Stop"

$pythonExe = $null
$pythonArgs = @()

$localVenvPython = Join-Path $PSScriptRoot ".venv310uv\Scripts\python.exe"
if (Test-Path $localVenvPython) {
    $pythonExe = $localVenvPython
}
elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $pythonExe = "py"
    $pythonArgs = @("-3")
}
else {
    Write-Error "No usable Python runtime found. Install Python or create .venv310uv first."
    exit 1
}

function Invoke-Python {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$ArgsList
    )

    if ($pythonExe -eq "py") {
        & py @pythonArgs @ArgsList
    }
    else {
        & $pythonExe @ArgsList
    }
}

if ($DashboardOnly) {
    Invoke-Python @("-c", "import flask, requests") 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[CFRS] Installing dashboard dependencies (flask, requests)..." -ForegroundColor Yellow
        Invoke-Python @("-m", "pip", "install", "flask", "requests")
    }

    Write-Host "[CFRS] Starting Dashboard-only mode..." -ForegroundColor Cyan
    Write-Host "[CFRS] Dashboard URL: http://127.0.0.1:5000" -ForegroundColor Green
    Invoke-Python @("dashboard_api.py")
    exit $LASTEXITCODE
}

Invoke-Python @("-c", "import cv2, face_recognition") 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Error "Missing camera backend dependencies (cv2 / face_recognition) in the selected Python runtime."
    exit 1
}

Write-Host "[CFRS] Starting full stack (dashboard + check-in backend)..." -ForegroundColor Cyan
$oldDashboard = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq "python.exe" -and $_.CommandLine -match "dashboard_api.py"
}
foreach ($p in $oldDashboard) {
    Stop-Process -Id $p.ProcessId -Force
}

$dashboardStartArgs = @("dashboard_api.py")
if ($pythonExe -eq "py") {
    $dashboardStartArgs = @("-3", "dashboard_api.py")
}

$dashboardProc = Start-Process -FilePath $pythonExe -ArgumentList $dashboardStartArgs -PassThru -WindowStyle Minimized

$healthReady = $false
for ($i = 0; $i -lt 20; $i++) {
    try {
        $resp = Invoke-RestMethod -Uri "http://127.0.0.1:5000/health" -TimeoutSec 2
        if ($resp.ok -eq $true) {
            $healthReady = $true
            break
        }
    }
    catch {
        Start-Sleep -Milliseconds 500
    }
}

if ($healthReady) {
    Write-Host "[CFRS] Dashboard ready: http://127.0.0.1:5000" -ForegroundColor Green
}
else {
    Write-Warning "Dashboard health check timeout, but will continue starting camera backend."
}

Write-Host "[CFRS] Starting check-in backend (camera). Press 'q' in camera window to stop." -ForegroundColor Green

if (-not $env:CFRS_FRAME_RESIZE_SCALE) { $env:CFRS_FRAME_RESIZE_SCALE = "0.42" }
if (-not $env:CFRS_PROCESS_EVERY_N_FRAMES) { $env:CFRS_PROCESS_EVERY_N_FRAMES = "3" }
if (-not $env:CFRS_BODY_DETECT_EVERY_N_FRAMES) { $env:CFRS_BODY_DETECT_EVERY_N_FRAMES = "4" }
if (-not $env:CFRS_BODY_RESIZE_SCALE) { $env:CFRS_BODY_RESIZE_SCALE = "0.4" }
if (-not $env:CFRS_CAMERA_WIDTH) { $env:CFRS_CAMERA_WIDTH = "640" }
if (-not $env:CFRS_CAMERA_HEIGHT) { $env:CFRS_CAMERA_HEIGHT = "480" }
if (-not $env:CFRS_FRAME_WRITE_INTERVAL_SEC) { $env:CFRS_FRAME_WRITE_INTERVAL_SEC = "0.45" }

try {
    Invoke-Python @("main.py")
    $backendExit = $LASTEXITCODE
}
finally {
    if ($dashboardProc -and -not $dashboardProc.HasExited) {
        Stop-Process -Id $dashboardProc.Id -Force
    }
}

exit $backendExit
