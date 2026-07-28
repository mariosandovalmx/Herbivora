# Creates HerbivoR.lnk next to HerbivoR.bat with assets\herbivor.ico
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Bat = Join-Path $Root "HerbivoR.bat"
$Ico = Join-Path $Root "assets\herbivor.ico"
$Lnk = Join-Path $Root "HerbivoR.lnk"

if (-not (Test-Path $Bat)) { throw "HerbivoR.bat not found at $Bat" }
if (-not (Test-Path $Ico)) { throw "Icon not found at $Ico" }

$Wsh = New-Object -ComObject WScript.Shell
$Sc = $Wsh.CreateShortcut($Lnk)
$Sc.TargetPath = $Bat
$Sc.WorkingDirectory = $Root
$Sc.IconLocation = "$Ico,0"
$Sc.Description = "HerbivoR - Leaf Damage Analysis"
$Sc.Save()
Write-Host "Created $Lnk"

$Desktop = [Environment]::GetFolderPath("Desktop")
$DeskLnk = Join-Path $Desktop "HerbivoR.lnk"
$Sc2 = $Wsh.CreateShortcut($DeskLnk)
$Sc2.TargetPath = $Bat
$Sc2.WorkingDirectory = $Root
$Sc2.IconLocation = "$Ico,0"
$Sc2.Description = "HerbivoR - Leaf Damage Analysis"
$Sc2.Save()
Write-Host "Created $DeskLnk"
