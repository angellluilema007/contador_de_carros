#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contador de Carros - Sistema de Conteo de Vehículos con YOLO
Copyright (c) 2025 Angel Lluilema
Todos los derechos reservados.
"""

import subprocess
import sys

scripts = [
    ("login.py", "Iniciando sesión..."),
    ("paravideo.py", "Seleccionando y reproduciendo video..."),
    ("coordinates.py", "Eligiendo área o coordenadas para el conteo... (el conteo iniciará al presionar 'Siguiente')")
]

for script, mensaje in scripts:
    print(f"\n{mensaje}")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"Error al ejecutar {script}. Deteniendo el flujo.")
        sys.exit(result.returncode)
print("\nFlujo finalizado. Nota: count_cars.py se ejecuta desde coordinates.py cuando presionas 'Siguiente'.")
