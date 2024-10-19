#define MyAppName "Photo-W"
#define MyAppVersion "1.6"
#define MyAppPublisher "Wamphyre"
#define MyAppURL "https://github.com/Wamphyre/Photo-W"
#define MyAppExeName "Photo-W.exe"

[Setup]
AppId={{2F2AFFD6-B9B6-4114-98AB-345DD98836E3}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DisableProgramGroupPage=yes
OutputDir=.
OutputBaseFilename=Photo-W-Setup-x64
SetupIconFile=icon.ico
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64os
ArchitecturesInstallIn64BitMode=x64os
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "dist\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion 64bit
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion isreadme
Source: "icon.ico"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Registry]
Root: HKLM64; Subkey: "SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{#MyAppExeName}"; ValueType: string; ValueName: ""; ValueData: "{app}\{#MyAppExeName}"; Flags: uninsdeletekey
Root: HKCR; Subkey: ".jpg"; ValueType: string; ValueName: ""; ValueData: "Photo-W.Image"; Flags: uninsdeletevalue
Root: HKCR; Subkey: ".jpeg"; ValueType: string; ValueName: ""; ValueData: "Photo-W.Image"; Flags: uninsdeletevalue
Root: HKCR; Subkey: ".png"; ValueType: string; ValueName: ""; ValueData: "Photo-W.Image"; Flags: uninsdeletevalue
Root: HKCR; Subkey: ".bmp"; ValueType: string; ValueName: ""; ValueData: "Photo-W.Image"; Flags: uninsdeletevalue
Root: HKCR; Subkey: ".gif"; ValueType: string; ValueName: ""; ValueData: "Photo-W.Image"; Flags: uninsdeletevalue
Root: HKCR; Subkey: ".tiff"; ValueType: string; ValueName: ""; ValueData: "Photo-W.Image"; Flags: uninsdeletevalue
Root: HKCR; Subkey: "Photo-W.Image"; ValueType: string; ValueName: ""; ValueData: "Archivo de imagen Photo-W"; Flags: uninsdeletekey
Root: HKCR; Subkey: "Photo-W.Image\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#MyAppExeName},0"
Root: HKCR; Subkey: "Photo-W.Image\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""

[Code]
function Is64BitInstallMode: Boolean;
begin
  Result := Is64BitInstallMode;
end;

function IsWin64: Boolean;
begin
  Result := IsWin64;
end;