# Creates Herbivora.lnk next to the app (and on Desktop) with assets\herbivor.ico.
# Prefer pythonw.exe so Windows does not show a black console window.
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Bat = Join-Path $Root "Herbivora.bat"
$Pythonw = Join-Path $Root ".venv\Scripts\pythonw.exe"
$Ico = Join-Path $Root "assets\herbivor.ico"
$Lnk = Join-Path $Root "Herbivora.lnk"

if (-not (Test-Path $Bat)) { throw "Herbivora.bat not found at $Bat" }
if (-not (Test-Path $Ico)) { throw "Icon not found at $Ico" }

function Set-HerbivoraShortcut($Path) {
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
    $Sc.Description = "Herbivora - Leaf Damage Analysis"
    $Sc.Save()
    Write-Host "Created $Path"
}

Set-HerbivoraShortcut $Lnk

$Desktop = [Environment]::GetFolderPath("Desktop")
$DeskLnk = Join-Path $Desktop "Herbivora.lnk"
Set-HerbivoraShortcut $DeskLnk
