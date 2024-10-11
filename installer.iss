#define MyAppName "Photo-W"
#define MyAppVersion "1.4.0"
#define MyAppPublisher "Wamphyre"
#define MyAppURL "https://github.com/Wamphyre/Photo-W"
#define MyAppExeName "Photo-W.exe"

[Setup]
AppId={{8FE86485-9BF5-4A0C-89E1-D1A7A13DF6FD}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DisableProgramGroupPage=yes
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

function InitializeSetup(): Boolean;
var
  UninstallKey: String;
  UninstallString: String;
  ResultCode: Integer;
begin
  Result := True;
  
  // Comprobar si existe una versión anterior
  UninstallKey := 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{#SetupSetting("AppId")}_is1';
  if RegQueryStringValue(HKLM, UninstallKey, 'UninstallString', UninstallString) then
  begin
    // Desinstalar la versión anterior
    if MsgBox('Se detectó una versión anterior de {#MyAppName}. ¿Desea desinstalarla antes de continuar?', mbConfirmation, MB_YESNO) = IDYES then
    begin
      Exec(RemoveQuotes(UninstallString), '/SILENT', '', SW_SHOW, ewWaitUntilTerminated, ResultCode);
    end;
  end;
end;