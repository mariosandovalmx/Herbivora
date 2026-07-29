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
set "PRIVATE=%LOCALAPPDATA%\HerbivoR\Python\python.exe"

REM Always ensure a private per-user CPython with Tcl/Tk (GUI dependency).
REM System Python installs are often missing init.tcl and break customtkinter.
echo Ensuring private HerbivoR Python (with Tcl/Tk)...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0packaging\ensure_windows_python.ps1"
if errorlevel 1 (
    echo WARNING: Private Python helper failed; will try system Python next.
) else if exist "%PRIVATE%" (
    set "PY=%PRIVATE%"
    goto :run
)

REM Fallback: Windows py launcher / python on PATH
where py >nul 2>&1
if not errorlevel 1 (
    py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "PY=py"
        set "PYARGS=-3"
        goto :run
    )
)

where python >nul 2>&1
if not errorlevel 1 (
    python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "PY=python"
        set "PYARGS="
        goto :run
    )
)

echo ERROR: No suitable Python found and private install failed.
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
