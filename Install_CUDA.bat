@echo off
REM Install HerbivoR with PyTorch CUDA 12.4 (NVIDIA GPU + recent drivers).
cd /d "%~dp0"
call "%~dp0_install_windows.bat" cuda
exit /b %ERRORLEVEL%
