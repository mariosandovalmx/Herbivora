@echo off
REM Shared installer body. Arg1 = cpu | cuda
REM Called by Install.bat / Install_CPU.bat / Install_CUDA.bat
setlocal EnableDelayedExpansion

if defined SystemRoot (
    set "PATH=%SystemRoot%\System32;%SystemRoot%;%PATH%"
)

cd /d "%~dp0"

set "TORCH_FLAVOR=%~1"
if /i "%TORCH_FLAVOR%"=="" set "TORCH_FLAVOR=cpu"
if /i not "%TORCH_FLAVOR%"=="cpu" if /i not "%TORCH_FLAVOR%"=="cuda" (
    echo ERROR: TORCH_FLAVOR must be cpu or cuda, got: %TORCH_FLAVOR%
    exit /b 1
)

echo ============================================
echo   HerbivoR - Dependency Installation
echo   PyTorch: %TORCH_FLAVOR%
echo ============================================
echo.

set "PYEXE="
set "PYARGS="

if exist ".venv\Scripts\python.exe" (
    set "PYEXE=.venv\Scripts\python.exe"
    set "PYARGS="
    goto :found_python
)

where py >nul 2>&1
if not errorlevel 1 (
    py -3 --version >nul 2>&1
    if not errorlevel 1 (
        set "PYEXE=py"
        set "PYARGS=-3"
        goto :found_python
    )
)

where python >nul 2>&1
if not errorlevel 1 (
    python --version >nul 2>&1
    if not errorlevel 1 (
        set "PYEXE=python"
        set "PYARGS="
        goto :found_python
    )
)

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
echo Install Python 3.10+ from https://www.python.org/  (check "Add to PATH")
echo Then re-run this installer.
echo.
if /i "%HERBIVOR_INSTALL_PAUSE%"=="0" exit /b 1
pause
exit /b 1

:found_python
echo Using: !PYEXE! !PYARGS!
!PYEXE! !PYARGS! --version
if errorlevel 1 (
    echo ERROR: The Python interpreter does not respond.
    if /i not "%HERBIVOR_INSTALL_PAUSE%"=="0" pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo Creating virtual environment .venv ...
    !PYEXE! !PYARGS! -m venv .venv
    if errorlevel 1 (
        echo ERROR creating .venv
        if /i not "%HERBIVOR_INSTALL_PAUSE%"=="0" pause
        exit /b 1
    )
)

echo.
echo Upgrading pip...
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 (
    echo ERROR upgrading pip
    if /i not "%HERBIVOR_INSTALL_PAUSE%"=="0" pause
    exit /b 1
)

echo.
if /i "%TORCH_FLAVOR%"=="cuda" (
    echo Installing PyTorch CUDA 12.4 wheels (NVIDIA GPU)...
    echo Requires a recent NVIDIA driver. See https://pytorch.org
    .venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
) else (
    echo Installing PyTorch CPU wheels...
    .venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)
if errorlevel 1 (
    echo ERROR installing PyTorch
    if /i not "%HERBIVOR_INSTALL_PAUSE%"=="0" pause
    exit /b 1
)

echo.
echo Installing HerbivoR packages from requirements.txt...
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR in pip install -r requirements.txt
    if /i not "%HERBIVOR_INSTALL_PAUSE%"=="0" pause
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
echo   Installation completed  [%TORCH_FLAVOR%]
echo ============================================
echo.
echo Required models in models\:
echo   - mobile_sam.pt
echo   - best_unet_shape.pth
echo   - best_model.pth
echo.
echo Open the application with: HerbivoR.bat
if /i "%TORCH_FLAVOR%"=="cuda" (
    echo.
    echo CUDA tip: run check_gpu.py if inference still uses CPU.
)
echo.
if /i not "%HERBIVOR_INSTALL_PAUSE%"=="0" pause
exit /b 0
