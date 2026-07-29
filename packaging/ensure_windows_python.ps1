# Ensures a private per-user CPython for HerbivoR (Windows).
# Used by Install_HerbivoR.bat. Always includes Tcl/Tk for the GUI.
$ErrorActionPreference = "Stop"
$Version = "3.12.10"
$Target = Join-Path $env:LOCALAPPDATA "HerbivoR\Python"
$PythonExe = Join-Path $Target "python.exe"

function Test-HerbivoRPython {
    param([string]$Exe)
    if (-not (Test-Path $Exe)) { return $false }
    & $Exe -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)"
    if ($LASTEXITCODE -ne 0) { return $false }
    $probe = @'
import os, sys
from pathlib import Path
base = Path(sys.base_prefix)
pairs = [
 (base/"tcl"/"tcl8.6", base/"tcl"/"tk8.6"),
 (base/"lib"/"tcl8.6", base/"lib"/"tk8.6"),
 (base/"Library"/"lib"/"tcl8.6", base/"Library"/"lib"/"tk8.6"),
]
for tcl_dir, tk_dir in pairs:
    if (tcl_dir/"init.tcl").is_file() and (tk_dir/"tk.tcl").is_file():
        os.environ["TCL_LIBRARY"] = str(tcl_dir)
        os.environ["TK_LIBRARY"] = str(tk_dir)
        break
import tkinter as t
r = t.Tk(); r.withdraw(); r.destroy()
'@
    & $Exe -c $probe
    return ($LASTEXITCODE -eq 0)
}

if (Test-HerbivoRPython $PythonExe) {
    Write-Host "Private Python already present: $PythonExe"
    exit 0
}

if (Test-Path $Target) {
    Write-Host "Removing incomplete private Python (missing or broken Tcl/Tk) ..."
    Remove-Item -Recurse -Force $Target -ErrorAction SilentlyContinue
}

$Url = "https://www.python.org/ftp/python/$Version/python-$Version-amd64.exe"
$Tmp = Join-Path $env:TEMP "herbivor-python-$Version-amd64.exe"
Write-Host "Downloading Python $Version ..."
Write-Host $Url
Invoke-WebRequest -Uri $Url -OutFile $Tmp -UseBasicParsing

New-Item -ItemType Directory -Force -Path (Split-Path $Target) | Out-Null
Write-Host "Installing to $Target (per-user, no PATH changes) ..."
$installArgs = @(
    "/quiet",
    "InstallAllUsers=0",
    "PrependPath=0",
    "Include_doc=0",
    "Include_launcher=0",
    "Include_test=0",
    "Include_tools=0",
    "Include_tcltk=1",
    "Shortcuts=0",
    "AssociateFiles=0",
    "TargetDir=$Target"
)
$p = Start-Process -FilePath $Tmp -ArgumentList $installArgs -Wait -PassThru
Remove-Item -Force $Tmp -ErrorAction SilentlyContinue

if ($p.ExitCode -ne 0) {
    Write-Error "Python installer exited with code $($p.ExitCode)"
    exit $p.ExitCode
}
if (-not (Test-Path $PythonExe)) {
    Write-Error "python.exe not found at $PythonExe"
    exit 1
}
if (-not (Test-HerbivoRPython $PythonExe)) {
    Write-Error "Private Python installed but tkinter/Tcl-Tk is not usable"
    exit 1
}
Write-Host "Private Python ready: $PythonExe"
exit 0
