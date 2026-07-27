@echo off
if defined SystemRoot (
    set "PATH=%SystemRoot%\System32;%SystemRoot%;%PATH%"
)
cd /d "%~dp0"

if exist ".venv\Scripts\pythonw.exe" (
    .venv\Scripts\pythonw.exe -m gui.main 2>>gui_error.log
    if not errorlevel 1 goto :done
    echo The GUI failed. Retrying with visible console...
    .venv\Scripts\python.exe -m gui.main
    if not errorlevel 1 goto :done
)

if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe -m gui.main
    if not errorlevel 1 goto :done
)

echo The .venv folder does not exist or startup failed.
echo Please run Install.bat first.
echo If the window does not appear, check gui_error.log in this folder.
pause
exit /b 1

:done
exit /b 0
