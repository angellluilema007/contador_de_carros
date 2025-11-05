#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Construcción de Ejecutable - Contador de Carros
Copyright (c) 2025 Angel Lluilema
Todos los derechos reservados.

Este script facilita la creación del .exe con todas las configuraciones necesarias
"""

import subprocess
import sys
import os

def build_executable():
    """Construye el ejecutable usando PyInstaller"""
    
    print("=" * 60)
    print("CONSTRUCCIÓN DEL EJECUTABLE - CONTADOR DE CARROS")
    print("Copyright (c) 2025 Angel Lluilema")
    print("=" * 60)
    
    # Verificar archivos requeridos
    print("\nVerificando archivos requeridos...")
    required_files = [
        'main.py', 'login.py', 'paravideo.py', 'coordinates.py', 
        'count_cars.py', 'sort.py', 'usuarios.json', 'contador_de_carros.spec'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} - NO ENCONTRADO")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n✗ ERROR: Faltan archivos requeridos: {', '.join(missing_files)}")
        sys.exit(1)
    
    # Verificar modelos YOLO (al menos uno)
    print("\nVerificando modelos YOLO...")
    yolo_models = ['yolo11m.pt', 'yolo11n.pt', 'yolov8m.pt', 'yolov5n.pt']
    found_models = [m for m in yolo_models if os.path.exists(m)]
    
    if not found_models:
        print("  ⚠ ADVERTENCIA: No se encontraron modelos YOLO (.pt)")
        print("  El ejecutable se creará pero necesitarás agregar los modelos manualmente.")
    else:
        for model in found_models:
            print(f"  ✓ {model}")
    
    # Verificar que PyInstaller esté instalado
    print("\nVerificando PyInstaller...")
    try:
        import PyInstaller
        print("  ✓ PyInstaller encontrado")
    except ImportError:
        print("  ✗ PyInstaller no está instalado")
        print("  Instalando PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("  ✓ PyInstaller instalado")
    
    # Limpiar builds anteriores
    print("\nLimpiando builds anteriores...")
    dirs_to_clean = ['build', 'dist']
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            import shutil
            shutil.rmtree(dir_name)
            print(f"  - {dir_name}/ eliminado")
    
    # Construir el ejecutable
    print("\nConstruyendo el ejecutable...")
    print("Esto puede tomar varios minutos, por favor espera...\n")
    
    try:
        # Usar el archivo .spec personalizado
        result = subprocess.run(
            [sys.executable, "-m", "PyInstaller", "contador_de_carros.spec", "--clean"],
            check=True
        )
        
        print("\n" + "=" * 60)
        print("✓ CONSTRUCCIÓN EXITOSA")
        print("=" * 60)
        print("\nEl ejecutable se encuentra en:")
        print("  dist/ContadorDeCarros/ContadorDeCarros.exe")
        print("\nContenido del paquete:")
        print("  ✓ Ejecutable principal")
        print("  ✓ Bibliotecas Python necesarias")
        print("  ✓ Modelos YOLO (.pt)")
        print("  ✓ Recursos y configuración")
        print("  ✓ Licencias y documentación")
        print("\n📦 Distribución:")
        print("  Puedes distribuir toda la carpeta 'dist/ContadorDeCarros/'")
        print("  o crear un ZIP con todo su contenido.")
        print("\n⚠️  Importante:")
        print("  - La carpeta 'recursos/videos/' debe contener los videos a analizar")
        print("  - Los resultados se guardarán en 'resultados/'")
        print("\nDesarrollado por Angel Lluilema © 2025")
        
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 60)
        print("✗ ERROR EN LA CONSTRUCCIÓN")
        print("=" * 60)
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    build_executable()
