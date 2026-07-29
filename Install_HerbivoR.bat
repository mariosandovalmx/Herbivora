@echo off
REM HerbivoR one-click installer (Windows).
REM   Install_HerbivoR.bat        → GUI installer (repair / manual)
REM   Install_HerbivoR.bat /auto  → fully automatic (used by Setup.exe)
setlocal EnableDelayedExpansion
cd /d "%~dp0"

if defined SystemRoot (
    set "PATH=%SystemRoot%\System32;%SystemRoot%;%PATH%"
)

set "AUTO=0"
if /i "%~1"=="/auto" set "AUTO=1"
if /i "%~1"=="--auto" set "AUTO=1"
if /i "%~1"=="/silent" set "AUTO=1"

set "BOOTSTRAP=%~dp0packaging\bootstrap_install.py"
if not exist "%BOOTSTRAP%" (
    echo ERROR: packaging\bootstrap_install.py not found.
    echo Extract the full HerbivoR release ZIP and try again.
    if not "%AUTO%"=="1" pause
    exit /b 1
)

set "PY="
set "PYARGS="
set "PRIVATE=%LOCALAPPDATA%\HerbivoR\Python\python.exe"

echo Ensuring private HerbivoR Python (with Tcl/Tk)...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0packaging\ensure_windows_python.ps1"
if errorlevel 1 (
    echo WARNING: Private Python helper reported an error.
) 
if exist "%PRIVATE%" (
    set "PY=%PRIVATE%"
    goto :run
)

REM Fallbacks: launcher / PATH / common install locations
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

for %%P in (
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    "%ProgramFiles%\Python312\python.exe"
    "%ProgramFiles%\Python311\python.exe"
) do (
    if exist %%~P (
        set "PY=%%~P"
        set "PYARGS="
        goto :run
    )
)

echo ERROR: Could not install or find a usable Python 3.10+.
echo Check your internet connection and re-run this installer.
if not "%AUTO%"=="1" pause
exit /b 1

:run
echo Using Python: !PY! !PYARGS!
echo Starting HerbivoR dependency install...
if "%AUTO%"=="1" (
    "!PY!" !PYARGS! "%BOOTSTRAP%" --yes --flavor auto
) else (
    "!PY!" !PYARGS! "%BOOTSTRAP%" --gui --flavor auto
)
set "ERR=!ERRORLEVEL!"
if not "!ERR!"=="0" (
    echo.
    echo Installer exited with code !ERR!.
    if not "%AUTO%"=="1" pause
)
exit /b !ERR!
