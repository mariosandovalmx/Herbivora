# Creates HerbivoR.lnk next to the app (and on Desktop) with assets\herbivor.ico.
# Prefer pythonw.exe so Windows does not show a black console window.
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Bat = Join-Path $Root "HerbivoR.bat"
$Pythonw = Join-Path $Root ".venv\Scripts\pythonw.exe"
$Ico = Join-Path $Root "assets\herbivor.ico"
$Lnk = Join-Path $Root "HerbivoR.lnk"

if (-not (Test-Path $Bat)) { throw "HerbivoR.bat not found at $Bat" }
if (-not (Test-Path $Ico)) { throw "Icon not found at $Ico" }

function Set-HerbivoRShortcut($Path) {
    $Wsh = New-Object -ComObject WScript.Shell
    $Sc = $Wsh.CreateShortcut($Path)
    if (Test-Path $Pythonw) {
        # GUI-only process: no cmd.exe / console window
        $Sc.TargetPath = $Pythonw
        $Sc.Arguments = "-m gui.main"
    } else {
        $Sc.TargetPath = $Bat
        $Sc.Arguments = ""
    }
    $Sc.WorkingDirectory = $Root
    $Sc.IconLocation = "$Ico,0"
    $Sc.Description = "HerbivoR - Leaf Damage Analysis"
    $Sc.Save()
    Write-Host "Created $Path"
}

Set-HerbivoRShortcut $Lnk

$Desktop = [Environment]::GetFolderPath("Desktop")
$DeskLnk = Join-Path $Desktop "HerbivoR.lnk"
Set-HerbivoRShortcut $DeskLnk
