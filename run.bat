@echo off
setlocal
cd /d "%~dp0"

if exist "venv310\Scripts\python.exe" (
	set "PYTHON_EXE=venv310\Scripts\python.exe"
) else (
	set "PYTHON_EXE=python"
)

echo Starting Classroom Monitoring System...
echo Using Python: %PYTHON_EXE%
"%PYTHON_EXE%" main.py

if errorlevel 1 (
	echo.
	echo Launch failed. Check dependency installation in README.md.
)
pause
