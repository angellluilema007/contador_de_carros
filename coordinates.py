import cv2
import numpy as np

class Coordinates:
    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {video_path}")
        
        # Variables de control
        self.paused = False
        self.speed = 1.0
        self.scale = 1.0
        self.points = []
        
        # Crear ventana y configurar callback del mouse
        cv2.namedWindow("Frame")
        cv2.setMouseCallback("Frame", self.print_coordinates)
        
        print("\nVideo cargado correctamente")
        print("\nControles:")
        print("Click izquierdo - Capturar coordenadas")
        print("p - Pausar/Reanudar")
        print("+ - Aumentar velocidad")
        print("- - Reducir velocidad")
        print("w - Aumentar tamaño")
        print("s - Reducir tamaño")
        print("r - Resetear ajustes")
        print("c - Limpiar coordenadas")
        print("q - Salir")
        
        self.video()

    def print_coordinates(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Ajustar coordenadas según la escala
            real_x = int(x / self.scale)
            real_y = int(y / self.scale)
            self.points.append([real_x, real_y])
            print(f"[{real_x}, {real_y}],")

    def draw_points(self, frame):
        # Dibujar puntos y líneas capturados
        for i, point in enumerate(self.points):
            x, y = int(point[0] * self.scale), int(point[1] * self.scale)
            cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
            cv2.putText(frame, f"P{i+1}", (x+10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
        if len(self.points) > 1:
            points_scaled = [[int(p[0] * self.scale), int(p[1] * self.scale)] 
                           for p in self.points]
            pts = np.array(points_scaled, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

    def video(self):
        while True:
            if not self.paused:
                status, frame = self.cap.read()
                if not status:
                    # Reiniciar video cuando llega al final
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                # Ajustar tamaño
                if self.scale != 1.0:
                    frame = cv2.resize(frame, None, 
                                    fx=self.scale, fy=self.scale, 
                                    interpolation=cv2.INTER_AREA)

                # Dibujar puntos y líneas
                self.draw_points(frame)

                # Mostrar información
                cv2.putText(frame, f"Velocidad: x{self.speed:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Escala: x{self.scale:.1f}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imshow("Frame", frame)

            # Control de teclas
            key = cv2.waitKey(int(10/self.speed)) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('p'):
                self.paused = not self.paused
                status = "PAUSADO" if self.paused else "REPRODUCIENDO"
                print(f"\nVideo {status}")
            elif key == ord('+') or key == ord('='): # Aumentar velocidad
                self.speed = min(self.speed + 0.2, 4.0)
                print(f"\nVelocidad: x{self.speed:.1f}")
            elif key == ord('-'): # Reducir velocidad
                self.speed = max(self.speed - 0.2, 0.2)
                print(f"\nVelocidad: x{self.speed:.1f}")
            elif key == ord('w'): # Aumentar tamaño
                self.scale = min(self.scale + 0.1, 2.0)
                print(f"\nEscala: x{self.scale:.1f}")
            elif key == ord('s'): # Reducir tamaño
                self.scale = max(self.scale - 0.1, 0.5)
                print(f"\nEscala: x{self.scale:.1f}")
            elif key == ord('r'): # Resetear ajustes
                self.speed = 1.0
                self.scale = 1.0
                print("\nAjustes restaurados")
            elif key == ord('c'): # Limpiar coordenadas
                self.points = []
                print("\nCoordenadas limpiadas")

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    import argparse
    import os

    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Seleccionar coordenadas en video')
    parser.add_argument('--video', type=str, default='recursos/videos/canar.mp4',
                      help='Ruta al archivo de video')
    args = parser.parse_args()

    # Verificar que el archivo existe
    if not os.path.exists(args.video):
        print(f"\nError: No se encuentra el archivo: {args.video}")
        print("\nVideos disponibles en recursos/videos:")
        try:
            videos = os.listdir('recursos/videos')
            for video in videos:
                print(f"- {video}")
        except:
            print("No se puede acceder al directorio recursos/videos")
        print("\nUso: python coordinates.py --video recursos/videos/canar.mp4")
        exit(1)

    print(f"\nAbriendo video: {args.video}")
    print("- Click izquierdo para capturar coordenadas")
    print("- Presiona 'q' para salir")
    
    c = Coordinates(args.video)