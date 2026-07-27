@echo off
setlocal EnableDelayedExpansion

REM Restore minimum Windows PATH (Explorer sometimes omits System32)
if defined SystemRoot (
    set "PATH=%SystemRoot%\System32;%SystemRoot%;%PATH%"
)

cd /d "%~dp0"

echo ============================================
echo   HerbivoR - Dependency Installation
echo ============================================
echo.

set "PYEXE="
set "PYARGS="

REM 1) Reuse existing venv
if exist ".venv\Scripts\python.exe" (
    set "PYEXE=.venv\Scripts\python.exe"
    set "PYARGS="
    goto :found_python
)

REM 2) py launcher
where py >nul 2>&1
if not errorlevel 1 (
    py -3 --version >nul 2>&1
    if not errorlevel 1 (
        set "PYEXE=py"
        set "PYARGS=-3"
        goto :found_python
    )
)

REM 3) python in PATH
where python >nul 2>&1
if not errorlevel 1 (
    python --version >nul 2>&1
    if not errorlevel 1 (
        set "PYEXE=python"
        set "PYARGS="
        goto :found_python
    )
)

REM 4) Common install locations
for %%P in (
    "%USERPROFILE%\miniconda3\python.exe"
    "%USERPROFILE%\anaconda3\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
) do (
    if exist %%P (
        set "PYEXE=%%~P"
        set "PYARGS="
        goto :found_python
    )
)

echo ERROR: Python 3 not found.
echo.
echo Options:
echo   1. Install Python 3.10+ from https://www.python.org/  (check "Add to PATH")
echo   2. Or open a terminal where "python" works and run:
echo        cd /d "%~dp0"
echo        python -m venv .venv
echo        .venv\Scripts\pip install -r requirements.txt
echo        .venv\Scripts\python download_models.py
echo.
pause
exit /b 1

:found_python
echo Using: !PYEXE! !PYARGS!
!PYEXE! !PYARGS! --version
if errorlevel 1 (
    echo ERROR: The Python interpreter does not respond.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo Creating virtual environment .venv ...
    !PYEXE! !PYARGS! -m venv .venv
    if errorlevel 1 (
        echo ERROR creating .venv
        pause
        exit /b 1
    )
)

echo.
echo Installing packages (may take several minutes)...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR in pip install
    pause
    exit /b 1
)

echo.
echo Downloading model weights (~226 MB, first time only)...
.venv\Scripts\python.exe download_models.py
if errorlevel 1 (
    echo WARNING: Model download failed. You can retry with:
    echo   .venv\Scripts\python.exe download_models.py
    echo Or place files manually in the models\ folder. See models\README.md
)

echo.
echo ============================================
echo   Installation completed
echo ============================================
echo.
echo Required models in models\:
echo   - mobile_sam.pt
echo   - best_unet_shape.pth
echo   - best_model.pth
echo.
echo Open the application with: HerbivoR.bat
echo.
pause
exit /b 0
