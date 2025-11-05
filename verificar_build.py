#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Verificación Pre-Build - Contador de Carros
Copyright (c) 2025 Angel Lluilema

Este script verifica que todos los archivos necesarios estén presentes
antes de construir el ejecutable.
"""

import os
import sys

def verificar_proyecto():
    """Verifica la integridad del proyecto antes de empaquetar"""
    
    print("=" * 70)
    print("VERIFICACIÓN DEL PROYECTO - CONTADOR DE CARROS")
    print("Copyright (c) 2025 Angel Lluilema")
    print("=" * 70)
    
    errores = []
    advertencias = []
    
    # 1. Verificar archivos Python principales
    print("\n📄 Verificando archivos Python...")
    archivos_python = {
        'main.py': 'Punto de entrada principal',
        'login.py': 'Sistema de autenticación',
        'paravideo.py': 'Reproductor de videos',
        'coordinates.py': 'Selector de coordenadas',
        'count_cars.py': 'Motor de conteo',
        'sort.py': 'Algoritmo de tracking'
    }
    
    for archivo, descripcion in archivos_python.items():
        if os.path.exists(archivo):
            size = os.path.getsize(archivo)
            print(f"  ✓ {archivo:20s} ({size:,} bytes) - {descripcion}")
        else:
            print(f"  ✗ {archivo:20s} - NO ENCONTRADO")
            errores.append(f"Falta archivo crítico: {archivo}")
    
    # 2. Verificar modelos YOLO
    print("\n🤖 Verificando modelos YOLO...")
    modelos_yolo = ['yolo11m.pt', 'yolo11n.pt', 'yolov8m.pt', 'yolov5n.pt']
    modelos_encontrados = []
    
    for modelo in modelos_yolo:
        if os.path.exists(modelo):
            size = os.path.getsize(modelo) / (1024 * 1024)  # MB
            print(f"  ✓ {modelo:20s} ({size:.1f} MB)")
            modelos_encontrados.append(modelo)
        else:
            print(f"  ✗ {modelo:20s} - NO ENCONTRADO")
    
    if not modelos_encontrados:
        errores.append("No se encontró ningún modelo YOLO (.pt)")
    elif len(modelos_encontrados) < len(modelos_yolo):
        advertencias.append(f"Solo se encontraron {len(modelos_encontrados)} de {len(modelos_yolo)} modelos YOLO")
    
    # 3. Verificar directorios
    print("\n📁 Verificando directorios...")
    directorios = {
        'recursos': 'Carpeta de recursos',
        'recursos/videos': 'Videos para análisis',
    }
    
    for directorio, descripcion in directorios.items():
        if os.path.exists(directorio) and os.path.isdir(directorio):
            archivos = len(os.listdir(directorio))
            print(f"  ✓ {directorio:20s} ({archivos} archivos) - {descripcion}")
        else:
            print(f"  ⚠ {directorio:20s} - NO EXISTE")
            advertencias.append(f"Directorio opcional no encontrado: {directorio}")
    
    # 4. Verificar archivos de configuración
    print("\n⚙️  Verificando configuración...")
    configs = {
        'usuarios.json': 'Base de datos de usuarios',
        'contador_de_carros.spec': 'Configuración PyInstaller',
        'requirements.txt': 'Dependencias Python',
        'pyproject.toml': 'Metadatos del proyecto'
    }
    
    for archivo, descripcion in configs.items():
        if os.path.exists(archivo):
            print(f"  ✓ {archivo:30s} - {descripcion}")
        else:
            print(f"  ⚠ {archivo:30s} - NO ENCONTRADO")
            if archivo == 'contador_de_carros.spec':
                errores.append(f"Falta archivo crítico: {archivo}")
            else:
                advertencias.append(f"Archivo opcional no encontrado: {archivo}")
    
    # 5. Verificar documentación y licencias
    print("\n📚 Verificando documentación...")
    docs = ['README.md', 'LICENSE', 'COPYRIGHT.md', 'NOTICE']
    
    for doc in docs:
        if os.path.exists(doc):
            print(f"  ✓ {doc}")
        else:
            print(f"  ⚠ {doc} - NO ENCONTRADO")
            advertencias.append(f"Documentación faltante: {doc}")
    
    # 6. Verificar dependencias instaladas
    print("\n📦 Verificando dependencias Python...")
    dependencias_criticas = [
        'cv2', 'numpy', 'pandas', 'torch', 'ultralytics', 
        'matplotlib', 'PIL', 'scipy', 'filterpy'
    ]
    
    dependencias_faltantes = []
    for dep in dependencias_criticas:
        try:
            __import__(dep)
            print(f"  ✓ {dep}")
        except ImportError:
            print(f"  ✗ {dep} - NO INSTALADO")
            dependencias_faltantes.append(dep)
    
    if dependencias_faltantes:
        errores.append(f"Faltan dependencias: {', '.join(dependencias_faltantes)}")
    
    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN DE VERIFICACIÓN")
    print("=" * 70)
    
    if not errores and not advertencias:
        print("✅ PERFECTO: El proyecto está listo para empaquetar")
        print("\nPuedes ejecutar:")
        print("  python build_exe.py")
        return 0
    
    if advertencias:
        print(f"\n⚠️  ADVERTENCIAS ({len(advertencias)}):")
        for adv in advertencias:
            print(f"  - {adv}")
    
    if errores:
        print(f"\n❌ ERRORES CRÍTICOS ({len(errores)}):")
        for err in errores:
            print(f"  - {err}")
        print("\n⛔ NO PUEDES EMPAQUETAR hasta resolver estos errores")
        return 1
    
    print("\n✓ Puedes continuar, pero revisa las advertencias")
    return 0

if __name__ == "__main__":
    sys.exit(verificar_proyecto())
