@echo off
REM Install Herbivora with PyTorch CPU (recommended if no NVIDIA GPU).
cd /d "%~dp0"
call "%~dp0_install_windows.bat" cpu
exit /b %ERRORLEVEL%
