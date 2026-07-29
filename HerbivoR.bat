@echo off
if defined SystemRoot (
    set "PATH=%SystemRoot%\System32;%SystemRoot%;%PATH%"
)
cd /d "%~dp0"

REM Auto-setup on first launch if Setup finished copying files but deps were skipped.
if not exist ".venv\Scripts\python.exe" (
    echo HerbivoR needs a one-time setup ^(Python packages + models^).
    echo This can take 5-20 minutes and requires internet...
    echo.
    call "%~dp0Install_HerbivoR.bat" /auto
    if errorlevel 1 (
        echo.
        echo Setup failed. See messages above, then re-run Install_HerbivoR.bat.
        pause
        exit /b 1
    )
    if not exist ".venv\Scripts\python.exe" (
        echo Setup finished but .venv is still missing.
        pause
        exit /b 1
    )
)

REM Prefer pythonw + start so this console can close (GUI only, no black window left open).
if exist ".venv\Scripts\pythonw.exe" (
    start "" "%~dp0.venv\Scripts\pythonw.exe" -m gui.main
    exit /b 0
)

if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe -m gui.main
    if not errorlevel 1 exit /b 0
)

echo HerbivoR failed to start.
echo If the window does not appear, check gui_error.log in this folder.
echo You can also re-run Install_HerbivoR.bat to repair the install.
pause
exit /b 1
