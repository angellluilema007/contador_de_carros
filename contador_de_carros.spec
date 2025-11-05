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
        # Incluir modelos YOLO específicos (PyInstaller no soporta wildcards)
        ('yolo11m.pt', '.'),
        ('yolo11n.pt', '.'),
        ('yolov8m.pt', '.'),
        ('yolov5n.pt', '.'),
        # Archivos de configuración
        ('usuarios.json', '.'),
        # Documentación y licencias
        ('LICENSE', '.'),
        ('COPYRIGHT.md', '.'),
        ('README.md', '.'),
    ],
    hiddenimports=[
        # Procesamiento de imágenes y video
        'cv2',
        'numpy',
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        # Análisis de datos
        'pandas',
        'openpyxl',
        # Visualización
        'matplotlib',
        'matplotlib.pyplot',
        'matplotlib.backends.backend_tkagg',
        'seaborn',
        # GUI
        'tkinter',
        'tkinter.ttk',
        'tkinter.filedialog',
        'tkinter.messagebox',
        # Deep Learning
        'ultralytics',
        'ultralytics.engine',
        'ultralytics.models',
        'ultralytics.models.yolo',
        'torch',
        'torchvision',
        'torch.nn',
        'torch.nn.functional',
        # Procesamiento científico
        'scipy',
        'scipy.special',
        'scipy.linalg',
        'scipy.spatial',
        # Tracking
        'filterpy',
        'filterpy.kalman',
        # Utilidades
        'json',
        'hashlib',
        'subprocess',
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
