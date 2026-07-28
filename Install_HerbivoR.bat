@echo off
REM HerbivoR one-click installer (Windows).
REM Double-click this file. Python is installed automatically if needed.
setlocal
cd /d "%~dp0"

if defined SystemRoot (
    set "PATH=%SystemRoot%\System32;%SystemRoot%;%PATH%"
)

set "BOOTSTRAP=%~dp0packaging\bootstrap_install.py"
if not exist "%BOOTSTRAP%" (
    echo ERROR: packaging\bootstrap_install.py not found.
    echo Extract the full HerbivoR release ZIP and try again.
    pause
    exit /b 1
)

set "PY="
set "PYARGS="

REM 1) Prefer private HerbivoR Python from a previous run
if exist "%LOCALAPPDATA%\HerbivoR\Python\python.exe" (
    set "PY=%LOCALAPPDATA%\HerbivoR\Python\python.exe"
    goto :run
)

REM 2) Windows py launcher
where py >nul 2>&1
if not errorlevel 1 (
    py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "PY=py"
        set "PYARGS=-3"
        goto :run
    )
)

REM 3) python on PATH
where python >nul 2>&1
if not errorlevel 1 (
    python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "PY=python"
        set "PYARGS="
        goto :run
    )
)

echo No suitable Python found. The installer will download a private copy...
echo This requires internet and may take a few minutes.
echo.

REM Use PowerShell to fetch a tiny bootstrap python via the official installer
REM by invoking bootstrap_install with the Windows built-in bits — we need SOME python.
REM Download embeddable is complex; call PowerShell to run the silent installer first.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0packaging\ensure_windows_python.ps1"
if errorlevel 1 (
    echo Failed to install private Python.
    pause
    exit /b 1
)
if exist "%LOCALAPPDATA%\HerbivoR\Python\python.exe" (
    set "PY=%LOCALAPPDATA%\HerbivoR\Python\python.exe"
    goto :run
)

echo ERROR: Private Python still missing after install.
pause
exit /b 1

:run
echo Starting HerbivoR installer...
"%PY%" %PYARGS% "%BOOTSTRAP%" --gui --flavor auto
set "ERR=%ERRORLEVEL%"
if not "%ERR%"=="0" (
    echo.
    echo Installer exited with code %ERR%.
    pause
)
exit /b %ERR%
