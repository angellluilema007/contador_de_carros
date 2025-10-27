import cv2
from matplotlib.path import Path
import torch
import numpy as np
import time

# ======== CONFIGURACIÓN HUD ========
ACCENT_COLOR = (0, 255, 0)     # Color principal (amarillo-naranja)
HIGHLIGHT_COLOR = (0, 0, 0)   # Color de resaltado
WARNING_COLOR = (0, 100, 255)     # Color de advertencia

def draw_hex(frame, center, size, color, thickness=2):
    """Dibuja un hexágono"""
    points = []
    for i in range(6):
        angle_deg = 60 * i - 30
        angle_rad = np.pi / 180 * angle_deg
        points.append([
            center[0] + size * np.cos(angle_rad),
            center[1] + size * np.sin(angle_rad)
        ])
    points = np.array(points, dtype=np.int32)
    cv2.polylines(frame, [points], True, color, thickness, cv2.LINE_AA)
    return points

def draw_hud_element(frame, x, y, w, h, title, value, frame_count):
    """Dibuja un elemento del HUD con hexágonos y efectos"""
    # Fondo semitransparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # Hexágonos decorativos
    pulse = 0.5 + 0.5 * np.sin(frame_count * 0.1)
    hex_color = tuple(int(c * pulse) for c in ACCENT_COLOR)
    draw_hex(frame, (x + 30, y + h//2), 20, hex_color, 2)
    draw_hex(frame, (x + w - 30, y + h//2), 20, hex_color, 2)
    
    # Líneas dinámicas
    line_length = int(w * (0.7 + 0.3 * pulse))
    cv2.line(frame, (x + 50, y + h - 10), (x + line_length - 50, y + h - 10), 
             hex_color, 2, cv2.LINE_AA)
    
    # Texto con efecto de brillo
    cv2.putText(frame, f"{title}: {value}", (x + 60, y + h//2 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, f"{title}: {value}", (x + 60, y + h//2 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, hex_color, 2, cv2.LINE_AA)

def add_hud_effects(frame):
    """Añade efectos globales al HUD"""
    # Efecto de escaneo
    scan_line = int(frame.shape[0] * (0.5 + 0.5 * np.sin(time.time())))
    overlay = frame.copy()
    cv2.line(overlay, (0, scan_line), (frame.shape[1], scan_line),
             HIGHLIGHT_COLOR, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame)
    
    # Vignette effect-para poner dolor al fondo de pantalla
    #rows, cols = frame.shape[:2]
    #kernel_x = cv2.getGaussianKernel(cols, cols//4)
    #kernel_y = cv2.getGaussianKernel(rows, rows//4)
    #kernel = kernel_y * kernel_x.T
    #mask = 255 * kernel / np.linalg.norm(kernel)
    #for i in range(3):
    #    frame[:,:,i] = frame[:,:,i] * mask

# ======== ZONAS DE CONTEO POR CARRILES Y GIROS ========
LANES = {
    # Carriles del primer brazo (directo)
    "Brazo 1 - Directo 1": np.array([  # Carril derecho directo
        [1168, 1024],
        [1448, 1024],
        [1422, 964],
        [1184, 958],
    ]),
    "Brazo 1 - Directo 2": np.array([  # Carril central directo
        [0,0],
        [0,0],
        [0,0],
        [0,0],
    ]),
    
    # Giros del primer brazo
    "Brazo 1 - Giro Derecha": np.array([  # Giro a la derecha
        [0,0],
        [0,0],
        [0,0],
        [0,0],
    ]),
    "Brazo 1 - Giro Izquierda": np.array([  # Giro a la izquierda
        [0,0],
        [0,0],
        [0,0],
        [0,0],
    ]),
    
    # Carriles del segundo brazo (directo)
    "Brazo 2 - Directo 1": np.array([  # Carril derecho directo
        [1436, 976],
        [1650, 962],
        [1568, 902],
        [1396, 908],
    ]),
    "Brazo 2 - Directo 2": np.array([  # Carril central directo
        [0,0],
        [0,0],
        [0,0],
        [0,0],
    ]),
    
    # Giros del segundo brazo
    "Brazo 2 - Giro Derecha": np.array([  # Giro a la derecha
        [0,0],
        [0,0],
        [0,0],
        [0,0],
    ]),
    "Brazo 2 - Giro Izquierda": np.array([  # Giro a la izquierda
        [0,0],
        [0,0],
        [0,0],
        [0,0],
    ])
}


def get_center(bbox):
    return ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)

def load_model():
    model = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True)
    return model

def get_bboxes(preds):
    df = preds.pandas().xyxy[0]
    df = df[df["confidence"] >= 0.4]
    df = df[df["name"] == "car"]
    return df[["xmin", "ymin", "xmax", "ymax"]].values.astype(int)

def is_valid_detection(xc, yc, zone):
    """Comprueba si el punto (xc,yc) está dentro del polígono 'zone'.
    'zone' debe ser un array Nx2 con coordenadas en el mismo sistema que xc,yc.
    """
    return Path(zone).contains_point((xc, yc))

def draw_hud_panel(frame, x, y, w, h, color=(0, 255, 255)):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

#def draw_direction_arrow(frame, start_point, end_point, color, thickness=3, arrow_size=20):
 #   """Dibuja una flecha para indicar el sentido de la vía"""
    # Dibujar la línea principal
  #  cv2.arrowedLine(frame, 
   #                 tuple(start_point.astype(int)), 
    #                tuple(end_point.astype(int)), 
     #               color, thickness, cv2.LINE_AA, tipLength=0.3)
    
    # Efecto de brillo
    #cv2.arrowedLine(frame, 
     #               tuple(start_point.astype(int)), 
      #              tuple(end_point.astype(int)), 
       #             tuple(min(255, c + 50) for c in color), 
        #            thickness//2, cv2.LINE_AA, tipLength=0.2)

# Definir los puntos de las flechas de dirección para cada carril
#DIRECTION_ARROWS = {
    # Flechas para movimientos directos del Brazo 1
 #   "Brazo 1 - Directo 1": {
  #      "start": np.array([1200, 900]),
   #     "end": np.array([1400, 900]),
    #    "text": "→"
    #},
    #"Brazo 1 - Directo 2": {
     #   "start": np.array([1300, 850]),
      #  "end": np.array([1500, 850]),
       # "text": "→"
    #},
    # Flechas para giros del Brazo 1
    #"Brazo 1 - Giro Derecha": {
    #    "start": np.array([1150, 880]),
     #   "end": np.array([1200, 950]),
      #  "text": "↘"
    #},
    #"Brazo 1 - Giro Izquierda": {
     #   "start": np.array([1350, 880]),
      #  "end": np.array([1300, 950]),
       # "text": "↙"
    #},
    # Flechas para movimientos directos del Brazo 2
    #"Brazo 2 - Directo 1": {
     #   "start": np.array([900, 850]),
      #  "end": np.array([700, 850]),
       # "text": "←"
   # },
    #"Brazo 2 - Directo 2": {
     #   "start": np.array([800, 750]),
      #  "end": np.array([600, 750]),
       # "text": "←"
    #},
    # Flechas para giros del Brazo 2
    #"Brazo 2 - Giro Derecha": {
     #   "start": np.array([750, 800]),
      #  "end": np.array([800, 900]),
       # "text": "↘"
    #},
    #"Brazo 2 - Giro Izquierda": {
     #   "start": np.array([950, 800]),
      #  "end": np.array([900, 900]),
       # "text": "↙"
    #}
#}

def detector(cap):
    model = load_model()
    frame_count = 0
    # Inicializar contadores por carril
    lane_counts = {lane: 0 for lane in LANES.keys()}
    total_count = 0
    tracked_cars = {}
    next_id = 1
    
    # Variables de control de video
    paused = False
    speed = 1.0  # Factor de velocidad
    scale = 1.0  # Factor de escala
    show_info = True  # Mostrar información de ajustes

    print("\nControles del video:")
    print("p - Pausar/Reanudar")
    print("+ - Aumentar velocidad")
    print("- - Reducir velocidad")
    print("w - Aumentar tamaño")
    print("s - Reducir tamaño")
    print("i - Mostrar/Ocultar información")
    print("r - Resetear ajustes")
    print("q - Salir")

    while cap.isOpened():
        # Leer frame (si estamos en pausa mantenemos el último frame)
        if not paused:
            status, frame = cap.read()
            if not status:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reiniciar video
                continue

            # Ajustar tamaño del frame si es necesario
            if scale != 1.0:
                frame = cv2.resize(frame, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA)

            # Ejecutar detector sobre el frame (las coordenadas devueltas
            # estarán en el sistema de coordenadas del frame actual)
            preds = model(frame)
            bboxes = get_bboxes(preds)

        # Calcular la zona de conteo escalada para este frame.
        # Normalizamos ZONE respecto al tamaño original del video al iniciar
        # detector; si no hay información de tamaño usaremos la resolución
        # actual del frame.
        try:
            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        except Exception:
            orig_w, orig_h = frame.shape[1], frame.shape[0]

        if orig_w <= 0 or orig_h <= 0:
            orig_w, orig_h = frame.shape[1], frame.shape[0]

        # Calcular zonas normalizadas y escaladas para cada carril
        current_lanes = {}
        for lane_name, lane_points in LANES.items():
            # Normalizar coordenadas (valores en [0,1])
            lane_norm = lane_points.astype(float) / np.array([orig_w, orig_h])
            # Escalar al tamaño actual del frame
            current_lanes[lane_name] = (lane_norm * np.array([frame.shape[1], frame.shape[0]])).astype(int)
        current_positions = {}

        for box in bboxes:
            xc, yc = get_center(box)
            matched = False

            for car_id, car_info in tracked_cars.items():
                prev_x, prev_y = car_info['pos']
                if abs(xc - prev_x) < 50 and abs(yc - prev_y) < 50:
                    # Verificar en qué carril está el vehículo
                    in_lanes = {}
                    counted_now = False
                    
                    for lane_name, lane_zone in current_lanes.items():
                        is_in_lane = is_valid_detection(xc, yc, lane_zone)
                        was_counted = car_info.get(f'counted_{lane_name}', False)
                        
                        if is_in_lane and not was_counted:
                            lane_counts[lane_name] += 1
                            counted_now = True
                            total_count += 1
                        
                        in_lanes[lane_name] = is_in_lane
                        in_lanes[f'counted_{lane_name}'] = is_in_lane or was_counted

                    current_positions[car_id] = {
                        'pos': (xc, yc),
                        **in_lanes  # Incluye información de todos los carriles
                    }
                    
                    matched = True
                    break

            if not matched:
                # Nuevo vehículo detectado
                in_lanes = {}
                for lane_name, lane_zone in current_lanes.items():
                    is_in_lane = is_valid_detection(xc, yc, lane_zone)
                    in_lanes[lane_name] = is_in_lane
                    in_lanes[f'counted_{lane_name}'] = is_in_lane
                    if is_in_lane:
                        lane_counts[lane_name] += 1
                        total_count += 1

                current_positions[next_id] = {
                    'pos': (xc, yc),
                    **in_lanes
                }
                next_id += 1

            # Rectángulo neón azul
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 2)
            # Punto central
            cv2.circle(frame, (xc, yc), 6, (0, 255, 255), -1)

        tracked_cars = current_positions
        frame_count += 1

        # Aplicar efecto de fondo oscuro con vignette
        add_hud_effects(frame)

        # ===== ELEMENTOS HUD =====
        # Panel Total Cars
        draw_hud_element(frame, 50, 40, 400, 80, "TOTAL", total_count, frame_count)
        
        # Paneles por carril y giros
        y_offset = 140
        lane_colors = {
            # Movimientos directos Brazo 1 (tonos azules)
            "Brazo 1 - Directo 1": (255, 128, 0),    # Azul claro
            "Brazo 1 - Directo 2": (255, 191, 0),    # Azul medio
            # Giros Brazo 1 (tonos verdes)
            "Brazo 1 - Giro Derecha": (0, 255, 0),    # Verde brillante
            "Brazo 1 - Giro Izquierda": (0, 255, 128), # Verde azulado
            
            # Movimientos directos Brazo 2 (tonos rojos)
            "Brazo 2 - Directo 1": (0, 0, 255),      # Rojo brillante
            "Brazo 2 - Directo 2": (0, 0, 192),      # Rojo oscuro
            # Giros Brazo 2 (tonos morados)
            "Brazo 2 - Giro Derecha": (255, 0, 255),  # Magenta
            "Brazo 2 - Giro Izquierda": (192, 0, 255) # Púrpura
        }
        
        # Organizar los movimientos por brazo y tipo
        arm_1_direct = {k: v for k, v in lane_counts.items() if k.startswith("Brazo 1 - Directo")}
        arm_1_turns = {k: v for k, v in lane_counts.items() if k.startswith("Brazo 1 - Giro")}
        arm_2_direct = {k: v for k, v in lane_counts.items() if k.startswith("Brazo 2 - Directo")}
        arm_2_turns = {k: v for k, v in lane_counts.items() if k.startswith("Brazo 2 - Giro")}
        
        # Calcular totales por brazo y tipo de movimiento
        total_arm_1 = sum(arm_1_direct.values()) + sum(arm_1_turns.values())
        total_arm_2 = sum(arm_2_direct.values()) + sum(arm_2_turns.values())
        total_direct_1 = sum(arm_1_direct.values())
        total_turns_1 = sum(arm_1_turns.values())
        total_direct_2 = sum(arm_2_direct.values())
        total_turns_2 = sum(arm_2_turns.values())
        
        # === BRAZO 1 ===
        # Total Brazo 1
        draw_hud_element(frame, 50, y_offset, 400, 80, "BRAZO 1 - TOTAL", total_arm_1, frame_count)
        y_offset += 100
        
        # Subtotal movimientos directos Brazo 1
        #draw_hud_element(frame, 50, y_offset, 400, 80, "BRAZO 1 - DIRECTOS", total_direct_1, frame_count)
        #y_offset += 100
        
        # Movimientos directos Brazo 1
        #for lane_name, count in arm_1_direct.items():
         #   draw_hud_element(frame, 50, y_offset, 400, 80, lane_name, count, frame_count + np.pi)
         #   y_offset += 100
        
        # Subtotal giros Brazo 1
        #draw_hud_element(frame, 50, y_offset, 400, 80, "BRAZO 1 - GIROS", total_turns_1, frame_count)
        #y_offset += 100
        
        # Giros Brazo 1
        #for lane_name, count in arm_1_turns.items():
         #   draw_hud_element(frame, 50, y_offset, 400, 80, lane_name, count, frame_count + np.pi)
          #  y_offset += 100
            
        y_offset += 20  # Espacio entre brazos
        
        # === BRAZO 2 ===
        # Total Brazo 2
        draw_hud_element(frame, 50, y_offset, 400, 80, "BRAZO 2 - TOTAL", total_arm_2, frame_count)
        y_offset += 100
        
        # Subtotal movimientos directos Brazo 2
        #draw_hud_element(frame, 50, y_offset, 400, 80, "BRAZO 2 - DIRECTOS", total_direct_2, frame_count)
        #y_offset += 100
        
        # Movimientos directos Brazo 2
        #for lane_name, count in arm_2_direct.items():
         #   draw_hud_element(frame, 50, y_offset, 400, 80, lane_name, count, frame_count + np.pi)
          #  y_offset += 100
        
        # Subtotal giros Brazo 2
        #draw_hud_element(frame, 50, y_offset, 400, 80, "BRAZO 2 - GIROS", total_turns_2, frame_count)
        #y_offset += 100
        
        # Giros Brazo 2
        #for lane_name, count in arm_2_turns.items():
         #   draw_hud_element(frame, 50, y_offset, 400, 80, lane_name, count, frame_count + np.pi)
          #  y_offset += 100
            
        # Dibujar zonas de detección para cada carril
        for lane_name, lane_zone in current_lanes.items():
            pts = lane_zone.reshape((-1, 1, 2)).astype(np.int32)
            color = lane_colors.get(lane_name, HIGHLIGHT_COLOR)
            
            # Efecto neón más intenso
            cv2.polylines(frame, [pts], True, color, 4, cv2.LINE_AA)
            # Efecto de brillo interior
            interior_color = tuple(min(255, c + 50) for c in color)
            cv2.polylines(frame, [pts], True, interior_color, 2, cv2.LINE_AA)
            
            # Añadir etiqueta del carril con fondo semi-transparente
            label_pos = pts.mean(axis=0).astype(int)[0]
            text_size = cv2.getTextSize(lane_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            # Fondo semi-transparente para el texto
            bg_pts = np.array([
                [label_pos[0] - 5, label_pos[1] - text_size[1] - 5],
                [label_pos[0] + text_size[0] + 5, label_pos[1] - text_size[1] - 5],
                [label_pos[0] + text_size[0] + 5, label_pos[1] + 5],
                [label_pos[0] - 5, label_pos[1] + 5]
            ], np.int32)
            
            overlay = frame.copy()
            cv2.fillPoly(overlay, [bg_pts], (0, 0, 0))
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            
            # Texto con borde negro para mejor visibilidad
            cv2.putText(frame, lane_name, 
                       (label_pos[0], label_pos[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, lane_name, 
                       (label_pos[0], label_pos[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
            
            # Dibujar flecha de dirección si existe para este carril
            #if lane_name in DIRECTION_ARROWS:
             #   arrow_info = DIRECTION_ARROWS[lane_name]
              #  draw_direction_arrow(frame, arrow_info["start"], arrow_info["end"], color)
                
                # Agregar símbolo de dirección
               # arrow_center = (arrow_info["start"] + arrow_info["end"]) // 2
                #cv2.putText(frame, arrow_info["text"],
                 #          tuple(arrow_center.astype(int)),
                  #         cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4, cv2.LINE_AA)
                #cv2.putText(frame, arrow_info["text"],
                 #          tuple(arrow_center.astype(int)),
                  #         cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2, cv2.LINE_AA)
        
        # Hexágonos decorativos en las esquinas
        draw_hex(frame, (30, 30), 25, ACCENT_COLOR, 2)
        draw_hex(frame, (frame.shape[1] - 30, 30), 25, ACCENT_COLOR, 2)
        
        # Tiempo y FPS
        fps_text = f"FPS: {int(1000/max(1, cv2.getTickFrequency()/(cv2.getTickCount())))}"
        time_text = time.strftime("%H:%M:%S")
        cv2.putText(frame, fps_text, (frame.shape[1] - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, ACCENT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(frame, time_text, (frame.shape[1] - 150, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, ACCENT_COLOR, 2, cv2.LINE_AA)

        # Mostrar información de ajustes
        if show_info:
            info_color = (0, 255, 255)
            cv2.putText(frame, f"Velocidad: x{speed:.1f}", (10, frame.shape[0] - 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, info_color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"Escala: x{scale:.1f}", (10, frame.shape[0] - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, info_color, 2, cv2.LINE_AA)
            status_text = "PAUSADO" if paused else "REPRODUCIENDO"
            cv2.putText(frame, f"Estado: {status_text}", (10, frame.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, info_color, 2, cv2.LINE_AA)

        #Fondo más oscuro para el efecto global-fondo de pantalla
        overlay_bg = frame.copy()
        cv2.rectangle(overlay_bg, (0, 0), (frame.shape[1], frame.shape[0]),
                     (0, 20, 50), -1)
        cv2.addWeighted(overlay_bg, 0.15 , frame, 0.85 , 0, frame)  #para poner el color de fondo y transparencia

        cv2.imshow("FUTURISTIC TRAFFIC HUD", frame)

        # Control de teclas
        delay = int(25/speed) if not paused else 0
        key = cv2.waitKey(delay) & 0xFF
        
        if key == ord('q'):
            print("\nSaliendo...")
            break
        elif key == ord('p'):
            paused = not paused
            status = "PAUSADO" if paused else "REPRODUCIENDO"
            print(f"\nVideo {status}")
        elif key == ord('+') or key == ord('='): # Aumentar velocidad
            speed = min(speed + 0.2, 4.0)
            print(f"\nVelocidad: x{speed:.1f}")
        elif key == ord('-'): # Reducir velocidad
            speed = max(speed - 0.2, 0.2)
            print(f"\nVelocidad: x{speed:.1f}")
        elif key == ord('w'): # Aumentar tamaño
            scale = min(scale + 0.1, 2.0)
            print(f"\nEscala: x{scale:.1f}")
        elif key == ord('s'): # Reducir tamaño
            scale = max(scale - 0.1, 0.5)
            print(f"\nEscala: x{scale:.1f}")
        elif key == ord('i'): # Mostrar/ocultar información
            show_info = not show_info
            print("\nInformación", "visible" if show_info else "oculta")
        elif key == ord('r'): # Resetear ajustes
            speed = 1.0
            scale = 1.0
            show_info = True
            print("\nAjustes restaurados")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    cap = cv2.VideoCapture("recursos/videos/canar.mp4")
    detector(cap)
