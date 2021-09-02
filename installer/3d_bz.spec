# -*- mode: python ; coding: utf-8 -*-


block_cipher = None

SITE_PACKAGES = 'C:/Users/Carson/miniconda3/envs/env_3d_beats/Lib/site-packages/'
REPO_ROOT = 'C:/Users/Carson/code/hand_decision_trees'

a = Analysis([REPO_ROOT + '/src/3d_bz.py'],
             pathex=[REPO_ROOT],
             binaries=[ (SITE_PACKAGES + 'glfw/glfw3.dll', '.') ],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts, 
          [],
          exclude_binaries=True,
          name='3d_bz',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas, 
               strip=False,
               upx=True,
               upx_exclude=[],
               name='3d_bz')
