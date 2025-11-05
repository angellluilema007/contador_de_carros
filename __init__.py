"""
Contador de Carros - Sistema de Conteo Vehicular con IA
Copyright (c) 2025 Angel Lluilema
Todos los derechos reservados.

Este paquete contiene todas las herramientas necesarias para el
análisis y conteo de vehículos en videos de tráfico utilizando
modelos de deep learning (YOLO) y algoritmos de tracking (SORT).

Módulos principales:
    - main: Punto de entrada principal del sistema
    - login: Sistema de autenticación de usuarios
    - paravideo: Reproductor y selector de videos
    - coordinates: Selector interactivo de áreas de conteo
    - count_cars: Motor de detección y conteo de vehículos
    - sort: Algoritmo SORT para seguimiento de objetos

Autor: Angel Lluilema
Licencia: MIT License
"""

__version__ = "1.0.0"
__author__ = "Angel Lluilema"
__copyright__ = "Copyright (c) 2025 Angel Lluilema"
__license__ = "MIT"

__all__ = [
    'main',
    'login',
    'paravideo',
    'coordinates',
    'count_cars',
    'sort'
]
