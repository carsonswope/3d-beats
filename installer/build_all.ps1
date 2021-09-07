# run this to create an installer.
# you must already have precompiled fatbins, and declare the model / config / fatbins you want to include
# run this from the root of the repo. i.e. you should be running the command: ./installer/build_all.ps1

$pyi_build = './installer/pyi_build'
$pyi_temp = './installer/pyi_temp'

if (Test-Path $pyi_build) { Remove-Item -Recurse -Force $pyi_build }
if (Test-Path $pyi_temp) { Remove-Item -Recurse -Force $pyi_temp }

# use pyinstaller to assemble standalone binary (and many many DLLs and other libs)
pyinstaller --workpath $pyi_temp --distpath $pyi_build ./installer/3d_bz.spec

if (Test-Path $pyi_temp) { Remove-Item -Recurse -Force $pyi_temp }

# you need Inno Setup (6) installed to make the installer!

$model_dir = 'models/m1'
$model_cfg = 'models/m1/m_cfg.json'
$fatbin_dir = 'cuda_fatbin'

&'C:\Program Files (x86)\Inno Setup 6\ISCC.exe' `
  make_windows_installer.iss `
  /Dbuild_dir=$pyi_build/3d_bz `
  /Dmodel_dir=$model_dir `
  /Dmodel_cfg=$model_cfg `
  /Dfatbin_dir=$fatbin_dir `
  /DAPP_NAME=3d-beats `
  /DAPP_VERSION=2.1 `
  /Oinstaller
