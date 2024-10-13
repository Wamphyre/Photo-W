# -*- mode: python ; coding: utf-8 -*-

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Definir el directorio base del proyecto
BASEDIR = os.path.abspath(os.path.dirname('photo_w.py'))

a = Analysis(
    ['photo_w.py'],
    pathex=[BASEDIR],
    binaries=[],
    datas=[
        ('icon.ico', '.'),
        ('file_version_info.txt', '.'),
    ],
    hiddenimports=[
        'ttkbootstrap',
        'PIL._tkinter_finder',
        'win32api',
        'win32com.client',
        'cv2',
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

# Recopilar archivos de datos adicionales
a.datas += collect_data_files('ttkbootstrap')
a.datas += collect_data_files('cv2')

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

# Función de depuración para imprimir información sobre los elementos
def debug_print(collection):
    for item in collection:
        print(f"Item: {item}")
        if not isinstance(item, tuple) or len(item) != 3:
            print(f"  WARNING: Invalid item structure: {item}")

print("Debugging a.binaries:")
debug_print(a.binaries)
print("\nDebugging a.zipfiles:")
debug_print(a.zipfiles)
print("\nDebugging a.datas:")
debug_print(a.datas)

# Asegurarse de que todos los elementos tengan la estructura correcta
def ensure_tuple_structure(collection):
    return [(item if isinstance(item, tuple) and len(item) == 3 else (str(item), '', 'DATA')) for item in collection]

coll = COLLECT(
    exe,
    ensure_tuple_structure(a.binaries),
    ensure_tuple_structure(a.zipfiles),
    ensure_tuple_structure(a.datas),
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Photo-W',
)