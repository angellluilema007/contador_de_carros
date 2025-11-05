# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller Configuration - Contador de Carros
Copyright (c) 2025 Angel Lluilema
Todos los derechos reservados.
"""

block_cipher = None

# Análisis del script principal
a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        # Incluir archivos de recursos
        ('recursos', 'recursos'),
        ('yolo*.pt', '.'),  # Incluir modelos YOLO
        ('usuarios.json', '.'),
        # Incluir otros scripts Python que se ejecutan desde main.py
        ('login.py', '.'),
        ('paravideo.py', '.'),
        ('coordinates.py', '.'),
        ('count_cars.py', '.'),
        ('sort.py', '.'),
    ],
    hiddenimports=[
        'cv2',
        'numpy',
        'pandas',
        'openpyxl',
        'matplotlib',
        'PIL',
        'tkinter',
        'ultralytics',
        'torch',
        'torchvision',
        'scipy',
        'seaborn',
        'sklearn',
        'sklearn.utils._cython_blas',
        'sklearn.neighbors.typedefs',
        'sklearn.neighbors.quad_tree',
        'sklearn.tree._utils',
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

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ContadorDeCarros',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Cambiar a False si quieres ocultar la consola
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Puedes agregar un icono aquí: icon='icono.ico'
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ContadorDeCarros',
)
