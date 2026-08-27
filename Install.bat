@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo ============================================
echo   Herbivora - Choose PyTorch build
echo ============================================
echo.
echo   1  CPU   - works everywhere (recommended if unsure)
echo   2  CUDA  - NVIDIA GPU, faster (needs drivers + CUDA 12.4 wheels)
echo.
echo Or double-click Install_CPU.bat / Install_CUDA.bat next time.
echo.

set "CHOICE="
set /p CHOICE="Enter 1 or 2 [default 1]: "
if "%CHOICE%"=="" set "CHOICE=1"

if "%CHOICE%"=="1" (
    call "%~dp0_install_windows.bat" cpu
    exit /b %ERRORLEVEL%
)
if "%CHOICE%"=="2" (
    call "%~dp0_install_windows.bat" cuda
    exit /b %ERRORLEVEL%
)

echo Invalid choice. Use 1 or 2.
pause
exit /b 1
