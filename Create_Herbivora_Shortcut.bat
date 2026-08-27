@echo off
REM Create Herbivora.lnk in this folder (and optionally Desktop) with the leaf icon.
setlocal
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0packaging\create_windows_shortcut.ps1"
if errorlevel 1 (
  echo Failed to create shortcut.
  pause
  exit /b 1
)
echo.
echo Shortcut created: Herbivora.lnk  ^(leaf icon; launches GUI without a console window^)
pause
exit /b 0
