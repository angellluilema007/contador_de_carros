# 🚗 Contador de Carros - Sistema de Conteo Vehicular con IA

Sistema inteligente de detección y conteo de vehículos utilizando modelos YOLO (You Only Look Once) y algoritmos de seguimiento en tiempo real.

## 👨‍💻 Autor

**Angel Lluilema**

Copyright © 2025 Angel Lluilema. Todos los derechos reservados.

## 📋 Descripción

Sistema avanzado de visión por computadora para el conteo automático de vehículos en videos de tráfico. Utiliza redes neuronales profundas (YOLO) para la detección y clasificación de vehículos, combinado con algoritmos de seguimiento (SORT) para mantener la persistencia de los objetos detectados.

### Características Principales

- 🎯 **Detección Precisa**: Utiliza modelos YOLO (YOLOv5, YOLOv8, YOLOv11) para detección en tiempo real
- 📊 **Múltiples Análisis**: Conteo por tipo de vehículo, por carril y estadísticas generales
- 🎨 **Interfaz Futurista**: HUD moderno con visualización en tiempo real
- 🔐 **Sistema de Autenticación**: Login seguro con encriptación de contraseñas
- 📈 **Exportación de Datos**: Resultados en formato CSV y Excel
- 🎥 **Reproducción Flexible**: Control de reproducción con pausa, velocidad variable
- 📍 **Selección de Áreas**: Definición interactiva de zonas de conteo

## 🛠️ Tecnologías Utilizadas

- **Python 3.8+**
- **OpenCV** - Procesamiento de video e imágenes
- **PyTorch** - Framework de deep learning
- **Ultralytics** - Implementación de YOLO
- **NumPy** - Operaciones numéricas
- **Pandas** - Análisis y exportación de datos
- **Matplotlib/Seaborn** - Visualización de resultados
- **Tkinter** - Interfaz gráfica
- **SORT Algorithm** - Tracking de objetos

## 📦 Instalación

### Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Instalación de Dependencias

```bash
pip install -r requirements.txt
```

### Modelos YOLO

El proyecto incluye varios modelos pre-entrenados:
- `yolo11n.pt` - YOLO11 Nano (más rápido)
- `yolo11m.pt` - YOLO11 Medium (balanceado)
- `yolov8m.pt` - YOLOv8 Medium
- `yolov5n.pt` - YOLOv5 Nano

## 🚀 Uso

### Ejecución del Sistema

```bash
python main.py
```

El sistema ejecutará secuencialmente:
1. **Login** - Autenticación de usuario
2. **Selector de Video** - Elección del video a analizar
3. **Selector de Coordenadas** - Definición del área de conteo
4. **Contador** - Análisis y conteo en tiempo real

### Ejecutable Windows

Para crear un ejecutable `.exe`:

```bash
python build_exe.py
```

El ejecutable se generará en `dist/ContadorDeCarros/`

## 📁 Estructura del Proyecto

```
contador_de_carros/
├── main.py                 # Punto de entrada principal
├── login.py                # Sistema de autenticación
├── paravideo.py           # Reproductor y selector de videos
├── coordinates.py         # Selector de áreas de conteo
├── count_cars.py          # Motor de detección y conteo
├── sort.py                # Algoritmo SORT para tracking
├── usuarios.json          # Base de datos de usuarios
├── recursos/              # Recursos del proyecto
│   ├── videos/           # Videos para analizar
│   ├── coords_selected.json
│   └── video_selected.json
├── resultados/           # Resultados de conteo (CSV)
├── requirements.txt      # Dependencias del proyecto
└── LICENSE              # Licencia MIT

```

## 📊 Resultados

El sistema genera tres tipos de archivos CSV:

1. **Resumen General** - Totales por tipo de vehículo
2. **Por Tipo** - Detalle temporal de cada tipo
3. **Por Carril** - Análisis separado por carril (si aplica)

## 🎮 Controles

### Durante el Conteo

- `ESPACIO` - Pausar/Reanudar
- `+` / `-` - Aumentar/Disminuir velocidad
- `S` - Captura de pantalla
- `Q` / `ESC` - Salir

### Selector de Coordenadas

- `Click izquierdo` - Agregar punto
- `Click derecho` - Eliminar último punto
- `ENTER` - Confirmar área
- `R` - Reiniciar selección

## 🔒 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

```
MIT License

Copyright (c) 2025 Angel Lluilema

Se concede permiso, libre de cargos, a cualquier persona que obtenga una copia
de este software y de los archivos de documentación asociados (el "Software"),
para utilizar el Software sin restricción...
```

## 📧 Contacto

**Angel Lluilema**

Para consultas, sugerencias o reportes de bugs, por favor contacta al autor.

## 🙏 Agradecimientos

- Ultralytics por la implementación de YOLO
- Alex Bewley por el algoritmo SORT
- Comunidad de OpenCV y PyTorch

---

**Desarrollado con ❤️ por Angel Lluilema**

*Proyecto Prometheo - Sistema de Conteo Vehicular Inteligente*
