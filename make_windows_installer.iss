; Script generated by the Inno Setup Script Wizard.
; SEE THE DOCUMENTATION FOR DETAILS ON CREATING INNO SETUP SCRIPT FILES!

#define PUBLISHER "Carson Swope"
#define APP_EXE "3d_bz.exe"

[Setup]
; NOTE: The value of AppId uniquely identifies this application. Do not use the same AppId value in installers for other applications.
; (To generate a new GUID, click Tools | Generate GUID inside the IDE.)
AppId={{E8E83B0A-4C84-458C-BA7B-CAEF3B56768B}
AppName={#APP_NAME}
AppVersion={#APP_VERSION}
AppVerName={#APP_NAME} {#APP_VERSION}
AppCopyright=Copyright (C) 2021 Carson Swope
AppPublisher=Carson Swope
AppPublisherURL=https://www.3d-beats.com/
DefaultDirName={autopf}\{#APP_NAME}
DisableProgramGroupPage=yes
; Install for current user, not system-wide
PrivilegesRequired=lowest
OutputBaseFilename={#APP_NAME}-setup-{#APP_VERSION}
Compression=lzma
SolidCompression=yes
WizardStyle=modern
VersionInfoVersion={#APP_VERSION}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#build_dir}/*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs
Source: "{#model_dir}/*"; DestDir: "{app}/{#model_dir}"; Flags: ignoreversion recursesubdirs
Source: "{#fatbin_dir}/*"; DestDir: "{app}/{#fatbin_dir}"; Flags: ignoreversion recursesubdirs
Source: "hand_config.json"; DestDir: "{app}"; Flags: ignoreversion

[Dirs]
Name:  "{userappdata}\3d-beats"

[UninstallDelete]
Type: filesandordirs; Name: "{userappdata}\3d-beats"
Type: filesandordirs; Name: "{app}"

[Icons]
; Name: "{app}\{#APP_NAME}"; Filename: "{app}\{#APP_NAME}"
Name: "{app}\{#APP_NAME}"; Filename: "{app}\{#APP_EXE}"; \
    Parameters: "-cfg {app}\{#model_cfg} --fatbin_in {app}\{#fatbin_dir} --no_debug --rs_half_resolution"
Name: "{autoprograms}\{#APP_NAME}"; Filename: "{app}\{#APP_EXE}"; \
    Parameters: "-cfg {app}\{#model_cfg} --fatbin_in {app}\{#fatbin_dir} --no_debug --rs_half_resolution"
Name: "{autodesktop}\{#APP_NAME}"; Filename: "{app}\{#APP_EXE}";  \
    Parameters: "-cfg {#model_cfg} --fatbin_in {#fatbin_dir} --no_debug --rs_half_resolution"; Tasks: desktopicon

