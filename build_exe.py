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
    print("=" * 60)
    
    # Verificar que PyInstaller esté instalado
    try:
        import PyInstaller
        print("✓ PyInstaller encontrado")
    except ImportError:
        print("✗ PyInstaller no está instalado")
        print("Instalando PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("✓ PyInstaller instalado")
    
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
        print("\nPuedes distribuir toda la carpeta 'dist/ContadorDeCarros/'")
        print("que contiene el ejecutable y todos los archivos necesarios.")
        
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 60)
        print("✗ ERROR EN LA CONSTRUCCIÓN")
        print("=" * 60)
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    build_executable()
