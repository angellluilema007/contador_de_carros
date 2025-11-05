# Instrucciones para Crear el Ejecutable

## Requisitos Previos
- Python 3.8 o superior instalado
- Todas las dependencias del proyecto instaladas (ver requirements.txt)

## Método 1: Script Automático (Recomendado)

### Paso 1: Verificar el proyecto (Opcional pero recomendado)

```powershell
python verificar_build.py
```

Este script verifica que todos los archivos necesarios estén presentes:
- Archivos Python principales
- Modelos YOLO (.pt)
- Dependencias instaladas
- Configuración y documentación

### Paso 2: Construir el ejecutable

```powershell
python build_exe.py
```

Este script:
1. Verifica archivos requeridos y modelos YOLO
2. Verifica que PyInstaller esté instalado
3. Limpia builds anteriores
4. Construye el ejecutable con todas las configuraciones necesarias

## Método 2: Manual con PyInstaller

Si prefieres ejecutar PyInstaller manualmente:

```powershell
# Instalar PyInstaller (si no está instalado)
pip install pyinstaller

# Construir el ejecutable usando el archivo .spec
pyinstaller contador_de_carros.spec --clean
```

## Resultado

El ejecutable se encontrará en:
```
dist/ContadorDeCarros/ContadorDeCarros.exe
```

**IMPORTANTE**: Debes distribuir toda la carpeta `dist/ContadorDeCarros/`, no solo el .exe. Esta carpeta contiene:
- El ejecutable principal
- Todas las bibliotecas necesarias
- Los modelos YOLO (.pt)
- Los recursos y archivos de configuración

## Opciones Adicionales

### Crear un Ejecutable de Un Solo Archivo

Si prefieres un único archivo .exe (tarda más en iniciar):

```powershell
pyinstaller --onefile --name ContadorDeCarros main.py --add-data "recursos;recursos" --add-data "*.pt;." --add-data "usuarios.json;." --hidden-import=ultralytics
```

### Agregar un Icono

1. Consigue un archivo .ico
2. Edita el archivo `contador_de_carros.spec`
3. Cambia la línea `icon=None` por `icon='tu_icono.ico'`
4. Reconstruye con `pyinstaller contador_de_carros.spec --clean`

### Ocultar la Consola

Si quieres que no se muestre la ventana de consola:
1. Edita `contador_de_carros.spec`
2. Cambia `console=True` por `console=False`
3. Reconstruye

## Solución de Problemas

### Error: Módulo no encontrado

Si el ejecutable falla por falta de un módulo, agrégalo a `hiddenimports` en el archivo .spec:

```python
hiddenimports=[
    'cv2',
    'numpy',
    # ... otros módulos
    'tu_modulo_faltante',
],
```

### El ejecutable es muy grande

Esto es normal. Los proyectos con PyTorch y ultralytics generan ejecutables grandes (500MB - 2GB). Puedes:
- Usar UPX para comprimir (ya está habilitado en el .spec)
- Distribuir con un instalador
- Considerar distribuir como aplicación Python en lugar de .exe

### Error con archivos de recursos

Asegúrate de que todos los archivos necesarios estén en la sección `datas` del archivo .spec.

## Distribución

Para distribuir tu aplicación:

1. **Opción A - Carpeta ZIP**: Comprime toda la carpeta `dist/ContadorDeCarros/` en un ZIP
2. **Opción B - Instalador**: Usa herramientas como Inno Setup o NSIS para crear un instalador profesional

## Notas Importantes

- El primer inicio del ejecutable puede ser lento (carga de modelos YOLO)
- Asegúrate de que los usuarios tengan los modelos .pt en la misma carpeta
- Los videos deben estar en la carpeta `recursos/videos/`
- Los resultados se guardarán en la carpeta `resultados/`
