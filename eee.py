"""
Launcher with login for Contador de Carros
- Shows a Tkinter login (from login.py)
- Then starts the detector flow from count_cars.py

Usage:
  python run_with_login.py                # opens login, then file selector
  python run_with_login.py --video PATH   # opens login, then uses the given video
  python run_with_login.py --no-gui       # opens login, then uses default video path
"""
import os
import sys
import argparse

try:
    from login import show_login
except Exception as e:
    print("✗ No se pudo cargar el módulo de login:", e)
    sys.exit(1)

import cv2

# Reutilizamos funciones del módulo principal
try:
    import count_cars
except Exception as e:
    print("✗ No se pudo importar count_cars:", e)
    sys.exit(1)


def main() -> int:
    # 1) Login
    user = show_login()
    if not user:
        print("\n✗ Inicio de sesión cancelado")
        return 0
    print(f"\n✓ Usuario autenticado: {user['username']} (rol: {user.get('role','user')})")

    # 2) CLI para opciones de video similares a count_cars
    parser = argparse.ArgumentParser(description='Lanzador con login - Contador de Vehículos')
    parser.add_argument('--video', '-v', type=str, help='Ruta al archivo de video')
    parser.add_argument('--no-gui', action='store_true', help='No usar selector de archivos GUI, usar video por defecto')
    args = parser.parse_args()

    # 3) Selección/Apertura de video (reutilizando helpers)
    video_path = None
    if args.video:
        video_path = args.video
        if not os.path.exists(video_path):
            print(f"\n✗ Error: El archivo '{video_path}' no existe")
            return 1
        print(f"\n✓ Usando video: {video_path}")
    elif args.no_gui:
        video_path = "recursos/videos/Video_int1.mp4"
        if not os.path.exists(video_path):
            print(f"\n✗ Error: Video por defecto no encontrado: {video_path}")
            return 1
        print(f"\n✓ Usando video por defecto: {video_path}")
    else:
        video_path = count_cars.select_video_file()
        if not video_path:
            print("\n✗ Operación cancelada por el usuario")
            return 0

    if not os.path.exists(video_path):
        print(f"\n✗ Error: No se puede acceder al archivo: {video_path}")
        return 1

    print("\n🎬 Iniciando análisis del video...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"\n✗ Error: No se pudo abrir el video: {video_path}")
        print("   Verifica que el formato sea compatible (MP4, AVI, MOV, etc.)")
        return 1

    # Info del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\n📊 Información del video:")
    print(f"   • Resolución: {width}x{height}")
    print(f"   • FPS: {fps:.2f}")
    print(f"   • Frames totales: {frame_count}")
    print(f"   • Duración: {frame_count/fps:.2f} segundos" if fps > 0 else "")

    # 4) Lanzar detector
    count_cars.detector(cap, video_path=video_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
