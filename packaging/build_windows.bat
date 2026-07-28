@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."

set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"

echo [HerbivoR] Installing packaging deps...
"%PY%" -m pip install -r requirements-dev.txt
if errorlevel 1 exit /b 1

for /f "usebackq delims=" %%V in ("VERSION") do set "VER=%%V"
if "%VER%"=="" set "VER=0.0.0"

echo [HerbivoR] Building PyInstaller onedir (this can take a long time)...
"%PY%" -m PyInstaller --noconfirm --clean packaging\herbivor.spec
if errorlevel 1 exit /b 1

set "OUT_DIR=dist\HerbivoR"
set "ZIP=dist\HerbivoR-windows-v%VER%.zip"
if exist "%ZIP%" del /f /q "%ZIP%"

echo [HerbivoR] Creating %ZIP% ...
powershell -NoProfile -Command "Compress-Archive -Path '%OUT_DIR%\*' -DestinationPath '%ZIP%' -Force"
if errorlevel 1 exit /b 1

echo.
echo [HerbivoR] Done.
echo   Folder: %CD%\%OUT_DIR%
echo   ZIP:    %CD%\%ZIP%
echo Copy the ZIP to another PC, extract, run HerbivoR.exe, then Check installation.
endlocal
