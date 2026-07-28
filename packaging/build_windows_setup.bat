@echo off
REM Build HerbivoR-Setup-v*.exe with Inno Setup (maintainers).
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
echo Building HerbivoR-Setup-v%VER%.exe ...
"%ISCC%" /DMyAppVersion=%VER% "packaging\HerbivoR.iss"
if errorlevel 1 exit /b 1
echo.
echo Output: dist\HerbivoR-Setup-v%VER%.exe
echo Attach this file to the GitHub Release (not the multi-GB PyInstaller bundle).
exit /b 0
