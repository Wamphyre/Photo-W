# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Recoger todos los submódulos de ttkbootstrap
ttkbootstrap_submodules = collect_submodules('ttkbootstrap')

a = Analysis(
    ['photo_w.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('icon.ico', '.'),
        *collect_data_files('ttkbootstrap', include_py_files=True)
    ],
    hiddenimports=[
        'ttkbootstrap',
        *ttkbootstrap_submodules,
        'PIL._tkinter_finder',
        'win32api',
        'win32com.client',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Excluir archivos innecesarios de opencv
a.binaries = [x for x in a.binaries if not x[0].startswith("opencv_videoio")]

# Asegurarse de incluir los DLLs necesarios para win32api
a.binaries += [('win32api.pyd', 'C:\\Windows\\System32\\win32api.pyd', 'BINARY')]
a.binaries += [('win32com.shell.shell.pyd', 'C:\\Windows\\System32\\win32com.shell.shell.pyd', 'BINARY')]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Photo-W',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico',
    version='file_version_info.txt',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Photo-W'
)