# Ensures a private per-user CPython for HerbivoR (Windows).
# Used by Install_HerbivoR.bat when no system Python 3.10+ is available.
$ErrorActionPreference = "Stop"
$Version = "3.12.10"
$Target = Join-Path $env:LOCALAPPDATA "HerbivoR\Python"
$PythonExe = Join-Path $Target "python.exe"

if (Test-Path $PythonExe) {
    & $PythonExe -c "import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Private Python already present: $PythonExe"
        exit 0
    }
}

$Url = "https://www.python.org/ftp/python/$Version/python-$Version-amd64.exe"
$Tmp = Join-Path $env:TEMP "herbivor-python-$Version-amd64.exe"
Write-Host "Downloading Python $Version ..."
Write-Host $Url
Invoke-WebRequest -Uri $Url -OutFile $Tmp -UseBasicParsing

New-Item -ItemType Directory -Force -Path (Split-Path $Target) | Out-Null
Write-Host "Installing to $Target (per-user, no PATH changes) ..."
$args = @(
    "/quiet",
    "InstallAllUsers=0",
    "PrependPath=0",
    "Include_doc=0",
    "Include_launcher=0",
    "Include_test=0",
    "Include_tools=0",
    "Shortcuts=0",
    "AssociateFiles=0",
    "TargetDir=$Target"
)
$p = Start-Process -FilePath $Tmp -ArgumentList $args -Wait -PassThru
Remove-Item -Force $Tmp -ErrorAction SilentlyContinue

if ($p.ExitCode -ne 0) {
    Write-Error "Python installer exited with code $($p.ExitCode)"
    exit $p.ExitCode
}
if (-not (Test-Path $PythonExe)) {
    Write-Error "python.exe not found at $PythonExe"
    exit 1
}
Write-Host "Private Python ready: $PythonExe"
exit 0
