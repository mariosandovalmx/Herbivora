@echo off
REM Build Herbivora-Setup-v*.exe with Inno Setup (maintainers).
setlocal
cd /d "%~dp0.."

set "VER="
for /f "usebackq delims=" %%V in ("VERSION") do set "VER=%%V"
if "%VER%"=="" set "VER=0.0.0"

where iscc >nul 2>&1
if errorlevel 1 (
    if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" (
        set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
    ) else if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" (
        set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
    ) else (
        echo ERROR: Inno Setup 6 not found. Install from https://jrsoftware.org/isinfo.php
        exit /b 1
    )
) else (
    set "ISCC=iscc"
)

if not exist "dist" mkdir dist

echo Building installer license text (LICENSE + third-party notices)...
where python >nul 2>&1
if not errorlevel 1 (
    python "packaging\build_installer_license.py"
) else if exist "%LOCALAPPDATA%\Herbivora\Python\python.exe" (
    "%LOCALAPPDATA%\Herbivora\Python\python.exe" "packaging\build_installer_license.py"
) else if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" "packaging\build_installer_license.py"
) else (
    echo ERROR: Python required to build packaging\installer_license.txt
    exit /b 1
)
if errorlevel 1 exit /b 1
if not exist "packaging\installer_license.txt" (
    echo ERROR: packaging\installer_license.txt was not created.
    exit /b 1
)

echo Building Herbivora-Setup-v%VER%.exe ...
"%ISCC%" /DMyAppVersion=%VER% "packaging\Herbivora.iss"
if errorlevel 1 exit /b 1
echo.
echo Output: dist\Herbivora-Setup-v%VER%.exe
echo Attach this file to the GitHub Release (not the multi-GB PyInstaller bundle).
exit /b 0
