#define MyAppName "Photo-W"
#define MyAppVersion "1.0"
#define MyAppPublisher "Wamphyre"
#define MyAppExeName "Photo-W.exe"

[Setup]
AppId={{8FE86485-9BF5-4A0C-89E1-D1A7A13DF6FD}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DisableProgramGroupPage=yes
OutputBaseFilename=Photo-W-Setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "dist\Photo-W\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist\Photo-W\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Registry]
Root: HKCR; Subkey: ".jpg"; ValueType: string; ValueName: ""; ValueData: "Photo-W.Image"; Flags: uninsdeletevalue
Root: HKCR; Subkey: ".jpeg"; ValueType: string; ValueName: ""; ValueData: "Photo-W.Image"; Flags: uninsdeletevalue
Root: HKCR; Subkey: ".png"; ValueType: string; ValueName: ""; ValueData: "Photo-W.Image"; Flags: uninsdeletevalue
Root: HKCR; Subkey: ".bmp"; ValueType: string; ValueName: ""; ValueData: "Photo-W.Image"; Flags: uninsdeletevalue
Root: HKCR; Subkey: "Photo-W.Image"; ValueType: string; ValueName: ""; ValueData: "Photo-W Image"; Flags: uninsdeletekey
Root: HKCR; Subkey: "Photo-W.Image\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#MyAppExeName},0"
Root: HKCR; Subkey: "Photo-W.Image\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
