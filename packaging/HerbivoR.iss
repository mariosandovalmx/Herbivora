; Inno Setup script — build HerbivoR-Setup-vX.Y.Z.exe for end users.
; Requires Inno Setup 6+: https://jrsoftware.org/isinfo.php
;
; From a clean source tree (or after git archive):
;   packaging\build_windows_setup.bat
;
; The Setup.exe unpacks the app and launches Install_HerbivoR.bat (GUI bootstrap).

#define MyAppName "HerbivoR"
#ifndef MyAppVersion
  #define MyAppVersion "1.3.0"
#endif
#define MyAppPublisher "HerbivoR"
#define MyAppURL "https://github.com/mariosandovalmx/HerbivoR"

[Setup]
AppId={{A8E3C2F1-9B4D-4E6A-8F21-HerbivoRSetup}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
DefaultDirName={localappdata}\HerbivoR
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=..\dist
OutputBaseFilename=HerbivoR-Setup-v{#MyAppVersion}
SetupIconFile=..\assets\herbivor.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\assets\herbivor.ico
InfoBeforeFile=..\packaging\setup_info_before.txt
LicenseFile=..\LICENSE

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a Desktop shortcut after install"; Flags: checkedonce

[Files]
; Ship the full source tree except heavy / local junk
Source: "..\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; \
  Excludes: ".git\*,.venv\*,__pycache__\*,*.pyc,dist\*,build\*,.cursor\*,hf_cache\*,models\*.pt,models\*.pth,*.lnk,gui_error.log,.pytest_cache\*,*.egg-info\*"

[Icons]
Name: "{group}\Install / Repair HerbivoR"; Filename: "{app}\Install_HerbivoR.bat"; WorkingDir: "{app}"; IconFilename: "{app}\assets\herbivor.ico"
Name: "{group}\HerbivoR (after install)"; Filename: "{app}\HerbivoR.bat"; WorkingDir: "{app}"; IconFilename: "{app}\assets\herbivor.ico"
Name: "{group}\Uninstall HerbivoR"; Filename: "{uninstallexe}"

[Run]
Filename: "{app}\Install_HerbivoR.bat"; Description: "Install Python packages and download models (required)"; Flags: nowait postinstall runasoriginaluser
Filename: "{app}\HerbivoR.lnk"; Description: "Launch HerbivoR"; Flags: nowait postinstall skipifdoesntexist unchecked shellexec

[UninstallDelete]
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\models"
Type: files; Name: "{app}\gui_error.log"
Type: files; Name: "{app}\HerbivoR.lnk"
