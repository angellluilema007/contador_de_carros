import cv2
import numpy as np

class FuturisticHUD:
    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {video_path}")

        self.video_path = video_path
        self.window_width = 1080
        self.window_height = 900
        self.paused = False
        self.speed = 1.0
        self.points = []
        # Estados de mapeo para transformar clics a coordenadas del frame original
        self._raw_w = 0
        self._raw_h = 0
        self._scale = 1.0
        self._x_off = 0
        self._y_off = 0
        self._target_w = int(self.window_width * 0.7)

        # Colores modernos (RGB en formato BGR para OpenCV)
        self.accent_color = (255, 170, 0)      # Cyan brillante (0, 170, 255 en RGB)
        self.warning_color = (0, 165, 255)     # Naranja (255, 165, 0 en RGB)
        self.success_color = (0, 255, 128)     # Verde brillante (128, 255, 0 en RGB)
        self.dark_bg = (15, 15, 20)            # Gris oscuro moderno
        self.panel_bg = (25, 25, 35)           # Panel ligeramente más claro

        # Posición de botones mejorada (más espaciados y grandes)
        self.button_rect = (20, 60, 180, 110)       # Reiniciar (x1, y1, x2, y2)
        self.next_button_rect = (20, 120, 180, 170) # Siguiente (x1, y1, x2, y2)

        cv2.namedWindow("HUD Futurista", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("HUD Futurista", self.window_width, self.window_height)
        cv2.setMouseCallback("HUD Futurista", self.mouse_event)

        print("\n💠 Interfaz Futurista Activa 💠")
        print("Click: Capturar | p: Pausa | q: Salir | c: Limpiar | Botón: Reiniciar Puntos\n")

        self.run()

    def mouse_event(self, event, x, y, flags, params):
        # Click sobre el video para agregar puntos
        if event == cv2.EVENT_LBUTTONDOWN:
            # Ignorar clics en el panel derecho
            if x >= self._target_w:
                return

            # Botones en el canvas de video (coordenadas del canvas)
            bx1, by1, bx2, by2 = self.button_rect
            nx1, ny1, nx2, ny2 = self.next_button_rect
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                self.points.clear()
                print("🔄 Puntos reiniciados.")
                return
            if nx1 <= x <= nx2 and ny1 <= y <= ny2:
                if len(self.points) < 3:
                    print("⚠️ Selecciona al menos 3 puntos para formar un polígono antes de continuar.")
                else:
                    self.finish_and_launch()
                return

            # Convertir clic a coordenadas del frame original (deshacer letterboxing y escala)
            if self._scale <= 0 or self._raw_w == 0 or self._raw_h == 0:
                return  # Aún no tenemos mapeo válido
            x_rel = x - self._x_off
            y_rel = y - self._y_off
            new_w = int(self._raw_w * self._scale)
            new_h = int(self._raw_h * self._scale)
            if x_rel < 0 or y_rel < 0 or x_rel >= new_w or y_rel >= new_h:
                # Clic fuera del área de video
                return
            xr = int(round(x_rel / self._scale))
            yr = int(round(y_rel / self._scale))
            xr = max(0, min(self._raw_w - 1, xr))
            yr = max(0, min(self._raw_h - 1, yr))
            self.points.append((xr, yr))
            print(f"P{len(self.points)} = ({xr}, {yr})")

    def draw_video(self, frame):
        """Renderiza el video (70% del ancho total)"""
        h, w = frame.shape[:2]
        target_w = int(self.window_width * 0.7)
        target_h = self.window_height
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        video_canvas = np.full((self.window_height, target_w, 3), self.dark_bg, dtype=np.uint8)

        x_off = (target_w - new_w) // 2
        y_off = (self.window_height - new_h) // 2
        video_canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized

        # Guardar estado de mapeo para clics
        self._raw_w, self._raw_h = w, h
        self._scale = scale
        self._x_off, self._y_off = x_off, y_off
        self._target_w = target_w

        # Marco con color moderno
        cv2.rectangle(video_canvas, (10, 10), (target_w - 10, self.window_height - 10),
                      self.accent_color, 3)
        cv2.putText(video_canvas, "SELECCION DE ZONA", (30, 45),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, self.accent_color, 2)

        # Botón "Reiniciar" mejorado con fondo y sombra
        bx1, by1, bx2, by2 = self.button_rect
        # Sombra
        cv2.rectangle(video_canvas, (bx1+3, by1+3), (bx2+3, by2+3), (0, 0, 0), -1)
        # Fondo del botón
        cv2.rectangle(video_canvas, (bx1, by1), (bx2, by2), (40, 40, 60), -1)
        # Borde
        cv2.rectangle(video_canvas, (bx1, by1), (bx2, by2), self.warning_color, 3)
        # Texto centrado
        cv2.putText(video_canvas, "REINICIAR", (bx1 + 25, by1 + 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.warning_color, 2)

        # Botón "Siguiente" mejorado
        nx1, ny1, nx2, ny2 = self.next_button_rect
        # Sombra
        cv2.rectangle(video_canvas, (nx1+3, ny1+3), (nx2+3, ny2+3), (0, 0, 0), -1)
        # Fondo (más brillante si hay suficientes puntos)
        btn_enabled = len(self.points) >= 3
        bg_color = (50, 70, 50) if btn_enabled else (30, 30, 40)
        cv2.rectangle(video_canvas, (nx1, ny1), (nx2, ny2), bg_color, -1)
        # Borde
        border_color = self.success_color if btn_enabled else (80, 80, 100)
        cv2.rectangle(video_canvas, (nx1, ny1), (nx2, ny2), border_color, 3)
        # Texto centrado
        text_color = self.success_color if btn_enabled else (120, 120, 140)
        cv2.putText(video_canvas, "SIGUIENTE", (nx1 + 20, ny1 + 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color, 2)

        # Dibujar puntos (convertir desde coords RAW a canvas)
        disp_pts = []
        for i, (xr, yr) in enumerate(self.points):
            xd = int(self._x_off + xr * self._scale)
            yd = int(self._y_off + yr * self._scale)
            disp_pts.append((xd, yd))
            # Círculo con borde
            cv2.circle(video_canvas, (xd, yd), 8, self.accent_color, -1)
            cv2.circle(video_canvas, (xd, yd), 8, (255, 255, 255), 2)
            # Etiqueta con fondo
            label = f"P{i+1}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(video_canvas, (xd + 12, yd - 25), (xd + 12 + text_size[0] + 8, yd - 5), (0, 0, 0), -1)
            cv2.putText(video_canvas, label, (xd + 16, yd - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.accent_color, 2)
        if len(disp_pts) > 1:
            pts = np.array(disp_pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(video_canvas, [pts], True, self.accent_color, 3)

        return video_canvas

    def draw_panel(self):
        """Panel derecho compacto (30%)"""
        panel_w = int(self.window_width * 0.3)
        panel = np.full((self.window_height, panel_w, 3), self.panel_bg, dtype=np.uint8)

        # Marco del panel
        cv2.rectangle(panel, (10, 10), (panel_w - 10, self.window_height - 10),
                      self.accent_color, 3)
        
        # Título del panel
        cv2.putText(panel, "COORDENADAS", (25, 45),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, self.accent_color, 2)

        # Contador de puntos
        points_text = f"Puntos: {len(self.points)}"
        cv2.putText(panel, points_text, (25, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.success_color, 1)

        # Tabla de coordenadas
        y = 110
        cv2.putText(panel, "Pto", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.accent_color, 2)
        cv2.putText(panel, "X", (90, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.accent_color, 2)
        cv2.putText(panel, "Y", (170, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.accent_color, 2)
        cv2.line(panel, (20, y + 8), (panel_w - 20, y + 8), self.accent_color, 2)
        y += 35

        # Mostrar hasta 12 puntos recientes
        for i, (x, yv) in enumerate(self.points[-12:]):
            row_color = self.success_color if (i % 2 == 0) else (180, 220, 255)
            cv2.putText(panel, f"P{i+1}", (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, row_color, 1)
            cv2.putText(panel, f"{x}", (90, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, row_color, 1)
            cv2.putText(panel, f"{yv}", (170, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, row_color, 1)
            y += 28

        # Información adicional en la parte inferior
        bottom_y = self.window_height - 100
        cv2.line(panel, (20, bottom_y), (panel_w - 20, bottom_y), self.accent_color, 1)
        
        # Velocidad
        cv2.putText(panel, f"Velocidad: x{self.speed:.1f}",
                    (25, bottom_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.warning_color, 1)
        
        # Estado
        if len(self.points) >= 3:
            status_text = "Listo para continuar"
            status_color = self.success_color
        else:
            status_text = f"Agrega {3-len(self.points)} punto(s) mas"
            status_color = self.warning_color
        cv2.putText(panel, status_text, (25, bottom_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1)
        
        return panel

    def run(self):
        while True:
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                video = self.draw_video(frame)
                panel = self.draw_panel()
                hud = np.hstack((video, panel))
                cv2.imshow("HUD Futurista", hud)

            key = cv2.waitKey(int(10/self.speed)) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                self.paused = not self.paused
            elif key == ord('+'):
                self.speed = min(self.speed + 0.2, 3.0)
            elif key == ord('-'):
                self.speed = max(self.speed - 0.2, 0.2)
            elif key == ord('c'):
                self.points.clear()

        self.cap.release()
        cv2.destroyAllWindows()

    def finish_and_launch(self):
        """Guarda los puntos seleccionados y lanza count_cars.py con esos puntos y el mismo video."""
        import json, os, subprocess, sys
        try:
            app_dir = os.path.dirname(os.path.abspath(__file__))
            coords_dir = os.path.join(app_dir, 'recursos')
            os.makedirs(coords_dir, exist_ok=True)
            coords_path = os.path.join(coords_dir, 'coords_selected.json')

            payload = {
                'video': self.video_path,
                'points': [(int(x), int(y)) for (x, y) in self.points]
            }
            with open(coords_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            print(f"💾 Coordenadas guardadas en: {coords_path}")
            print("🚀 Iniciando conteo de vehículos con count_cars.py ...")

            # Cerrar esta ventana antes de abrir el conteo
            self.cap.release()
            cv2.destroyAllWindows()

            count_path = os.path.join(app_dir, 'count_cars.py')
            cmd = [sys.executable, count_path, '--video', self.video_path, '--coords', coords_path]
            subprocess.run(cmd)
        except Exception as e:
            print(f"Error al continuar al conteo: {e}")


if __name__ == '__main__':
    import argparse, os, json
    parser = argparse.ArgumentParser(description='HUD Futurista Azul con Botón Reiniciar')
    parser.add_argument('--video', type=str, default='', help='Ruta del video a usar')
    args = parser.parse_args()

    video_path = args.video
    
    # Si no se pasó video por CLI, buscar en video_selected.json (viene de paravideo.py)
    if not video_path:
        try:
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recursos', 'video_selected.json')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    video_path = data.get('video', '')
                    print(f"Video cargado desde paravideo.py: {video_path}")
        except Exception as e:
            print(f"No se pudo cargar video desde config: {e}")
    
    # Fallback al video por defecto
    if not video_path:
        video_path = 'recursos/videos/Video_int1.mp4'
    
    if not os.path.exists(video_path):
        print(f"Error: No se encuentra el archivo: {video_path}")
        exit(1)

    print(f"Cargando video en coordinates.py: {video_path}")
    FuturisticHUD(video_path)
