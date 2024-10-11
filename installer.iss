#define MyAppName "Photo-W"
#define MyAppVersion "1.2"
#define MyAppPublisher "Wamphyre"
#define MyAppURL "https://github.com/Wamphyre/Photo-W"
#define MyAppExeName "Photo-W.exe"

[Setup]
; NOTE: The value of AppId uniquely identifies this application. Do not use the same AppId value in installers for other applications.
; (To generate a new GUID, click Tools | Generate GUID inside the IDE.)
AppId={{8FE86485-9BF5-4A0C-89E1-D1A7A13DF6FD}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
;AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DisableProgramGroupPage=yes
; Uncomment the following line to run in non administrative install mode (install for current user only.)
;PrivilegesRequired=lowest
OutputDir=.
OutputBaseFilename=Photo-W-Setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "dist\Photo-W\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist\Photo-W\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; NOTE: Don't use "Flags: ignoreversion" on any shared system files

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Registry]
Root: HKCR; Subkey: ".jpg"; ValueType: string; ValueName: ""; ValueData: "Photo-W.Image"; Flags: uninsdeletevalue
Root: HKCR; Subkey: ".jpeg"; ValueType: string; ValueName: ""; ValueData: "Photo-W.Image"; Flags: uninsdeletevalue
Root: HKCR; Subkey: ".png"; ValueType: string; ValueName: ""; ValueData: "Photo-W.Image"; Flags: uninsdeletevalue
Root: HKCR; Subkey: ".bmp"; ValueType: string; ValueName: ""; ValueData: "Photo-W.Image"; Flags: uninsdeletevalue
Root: HKCR; Subkey: "Photo-W.Image"; ValueType: string; ValueName: ""; ValueData: "Photo-W Image"; Flags: uninsdeletekey
Root: HKCR; Subkey: "Photo-W.Image\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#MyAppExeName},0"
Root: HKCR; Subkey: "Photo-W.Image\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#MyAppExeName}"" ""%1"""

[Code]
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  mRes : integer;
begin
  case CurUninstallStep of
    usUninstall:
      begin
        mRes := MsgBox('¿Desea eliminar todos los archivos de configuración y datos de usuario?', mbConfirmation, MB_YESNO or MB_DEFBUTTON2)
        if mRes = IDYES then
        begin
          DelTree(ExpandConstant('{userappdata}\{#MyAppName}'), True, True, True);
        end;
      end;
  end;
end;