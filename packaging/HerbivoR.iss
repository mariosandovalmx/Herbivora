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
#define MyAppPublisher "Mario Sandoval"
#define MyAppURL "https://github.com/mariosandovalmx/HerbivoR"
#define MyAppCopyright "Copyright (C) 2026 Mario Sandoval"

[Setup]
AppId={{A8E3C2F1-9B4D-4E6A-8F21-HerbivoRSetup}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppCopyright={#MyAppCopyright}
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
; Full agreement: LICENSE + citation + THIRD_PARTY_NOTICES (built by build_installer_license.py)
LicenseFile=..\packaging\installer_license.txt
InfoAfterFile=..\packaging\setup_info_after.txt

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a Desktop shortcut after install"; Flags: checkedonce

[Files]
; Ship the full source tree except heavy / local junk
Source: "..\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; \
  Excludes: ".git\*,.venv\*,__pycache__\*,*.pyc,dist\*,build\*,.cursor\*,hf_cache\*,models\*.pt,models\*.pth,*.lnk,gui_error.log,debug-*.log,.pytest_cache\*,*.egg-info\*"

[Icons]
Name: "{group}\Install or Repair HerbivoR"; Filename: "{app}\Install_HerbivoR.bat"; WorkingDir: "{app}"; IconFilename: "{app}\assets\herbivor.ico"
; pythonw.exe = no black console (path exists after Install_HerbivoR.bat in [Run];
; the .lnk is fine even if created slightly earlier — target appears before first use)
Name: "{group}\HerbivoR"; Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: "-m gui.main"; WorkingDir: "{app}"; IconFilename: "{app}\assets\herbivor.ico"
Name: "{autodesktop}\HerbivoR"; Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: "-m gui.main"; WorkingDir: "{app}"; IconFilename: "{app}\assets\herbivor.ico"; Tasks: desktopicon
Name: "{group}\Uninstall HerbivoR"; Filename: "{uninstallexe}"

[Run]
; Always run during Setup (no checkbox). Must finish before the Finished page.
Filename: "{app}\Install_HerbivoR.bat"; Parameters: "/auto"; StatusMsg: "Installing Python, packages, and models (5–20 min, internet required)..."; Flags: waituntilterminated runasoriginaluser
; Only optional checkbox on the Finished page (pythonw = GUI only, no console):
Filename: "{app}\.venv\Scripts\pythonw.exe"; Parameters: "-m gui.main"; WorkingDir: "{app}"; Description: "Launch HerbivoR"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\models"
Type: files; Name: "{app}\gui_error.log"
Type: files; Name: "{app}\HerbivoR.lnk"
