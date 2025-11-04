import cv2
from matplotlib.path import Path
import torch
import numpy as np
import time
import importlib
import sys
from sort import Sort  # Tracker SORT para tracking persistente
try:
    # Windows: obtener tamaño de pantalla para posicionar ventanas
    from ctypes import windll
except Exception:
    windll = None

# ======== CONFIGURACIÓN HUD ========
# Paleta visual (BGR)
# Fondo: #0A0A0F -> (15,10,10) en BGR
# Acentos: #00FFFF (cian neón) -> (255,255,0), #FF00FF (magenta) -> (255,0,255)
# Brillo: #00AEEF (glow) -> (239,174,0) y blanco suave
ACCENT_COLOR = (255, 255, 0)       # Cian neón (BGR)
HIGHLIGHT_COLOR = (255, 0, 255)    # Magenta (BGR)
WARNING_COLOR = (0, 100, 255)      # Naranja/amber (BGR), se mantiene para avisos

# Tamaños por ventana (ancho, alto)
HUD_WINDOW_SIZE = (1280, 900)
RESULTS_PANEL_SIZE = (420, 900)

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
    """Dibuja una tarjeta HUD estilo glassmorphism con bordes redondeados, glow y tipografía legible."""
    # Colores (BGR)
    bg_glass = (30, 30, 45)           # vidrio oscuro
    glow_cyan = (255, 255, 0)         # cian
    glow_magenta = (255, 0, 255)      # magenta
    glow_soft = (239, 174, 0)         # azul eléctrico tenue (00AEEF)
    white_soft = (250, 245, 245)      # blanco suave

    # Helpers locales
    def rounded_rect(img, pt1, pt2, radius, color, thickness=-1):
        x1, y1 = pt1
        x2, y2 = pt2
        r = int(max(0, min(radius, min(abs(x2-x1), abs(y2-y1))//2)))
        # Centro + bordes
        if thickness < 0:
            # Relleno
            cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, -1, cv2.LINE_AA)
            cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, -1, cv2.LINE_AA)
            cv2.ellipse(img, (x1+r, y1+r), (r, r), 180, 0, 90, color, -1, cv2.LINE_AA)
            cv2.ellipse(img, (x2-r, y1+r), (r, r), 270, 0, 90, color, -1, cv2.LINE_AA)
            cv2.ellipse(img, (x1+r, y2-r), (r, r), 90, 0, 90, color, -1, cv2.LINE_AA)
            cv2.ellipse(img, (x2-r, y2-r), (r, r), 0, 0, 90, color, -1, cv2.LINE_AA)
        else:
            # Borde
            cv2.line(img, (x1+r, y1), (x2-r, y1), color, thickness, cv2.LINE_AA)
            cv2.line(img, (x1+r, y2), (x2-r, y2), color, thickness, cv2.LINE_AA)
            cv2.line(img, (x1, y1+r), (x1, y2-r), color, thickness, cv2.LINE_AA)
            cv2.line(img, (x2, y1+r), (x2, y2-r), color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x1+r, y1+r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x2-r, y1+r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x1+r, y2-r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x2-r, y2-r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)

    # 1) Capa de vidrio translúcido
    overlay = frame.copy()
    rounded_rect(overlay, (x, y), (x + w, y + h), radius=16, color=bg_glass, thickness=-1)
    cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)

    # 2) Borde neón dual (cian/magenta) + brillo suave
    border = frame.copy()
    rounded_rect(border, (x, y), (x + w, y + h), 16, glow_cyan, thickness=2)
    rounded_rect(border, (x+1, y+1), (x + w - 1, y + h - 1), 16, glow_magenta, thickness=1)
    cv2.addWeighted(border, 0.6, frame, 0.4, 0, frame)

    # 3) Reflejo sutil (gloss) en el borde superior izquierdo
    gloss = frame.copy()
    cv2.ellipse(gloss, (x + int(w*0.25), y + 8), (int(w*0.35), 10), 0, 0, 180, white_soft, -1, cv2.LINE_AA)
    cv2.addWeighted(gloss, 0.06, frame, 0.94, 0, frame)

    # 4) Barrido neón interno (scan) dentro de la tarjeta
    sweep = frame.copy()
    phase = (frame_count * 4) % (w + 120)
    sx = x - 60 + phase
    cv2.rectangle(sweep, (sx-6, y+4), (sx+6, y + h - 4), glow_soft, -1)
    cv2.addWeighted(sweep, 0.12, frame, 0.88, 0, frame)

    # 5) Pequeños hexágonos decorativos con pulso
    pulse = 0.6 + 0.4 * np.sin(frame_count * 0.08)
    hex_color = (int(glow_cyan[0]*pulse), int(glow_cyan[1]*pulse), int(glow_cyan[2]*pulse))
    draw_hex(frame, (x + 26, y + h//2), 18, hex_color, 2)
    draw_hex(frame, (x + w - 26, y + h//2), 18, (int(glow_magenta[0]*pulse), int(glow_magenta[1]*pulse), int(glow_magenta[2]*pulse)), 2)

    # 6) Tipografía: título grande y valor destacado
    value_str = f"{value}" if value is not None else ""
    title_pos = (x + 58, y + int(h*0.45))
    value_pos = (x + 58, y + int(h*0.80))
    # Título con sombra
    cv2.putText(frame, str(title), (title_pos[0]+1, title_pos[1]+1), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, str(title), title_pos, cv2.FONT_HERSHEY_DUPLEX, 0.9, white_soft, 1, cv2.LINE_AA)
    # Valor con énfasis
    cv2.putText(frame, value_str, (value_pos[0]+2, value_pos[1]+2), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, value_str, value_pos, cv2.FONT_HERSHEY_DUPLEX, 1.2, glow_cyan, 2, cv2.LINE_AA)

    # 7) Línea inferior luminosa (sección)
    line = frame.copy()
    cv2.line(line, (x + 50, y + h - 10), (x + w - 50, y + h - 10), glow_soft, 2, cv2.LINE_AA)
    cv2.addWeighted(line, 0.5, frame, 0.5, 0, frame)

def add_hud_effects(frame):
    """Efectos globales: tono de fondo #0A0A0F, barrido neón tenue y bordes con ligera viñeta."""
    h, w = frame.shape[:2]

    # 1) Tinte de fondo (ambient) para un look unificado
    ambient = frame.copy()
    cv2.rectangle(ambient, (0, 0), (w, h), (15, 10, 10), -1)  # #0A0A0F en BGR
    cv2.addWeighted(ambient, 0.20, frame, 0.80, 0, frame)

    # 2) Barrido neón horizontal sutil recorriendo el HUD
    t = time.time()
    y_center = int((np.sin(t * 0.8) * 0.5 + 0.5) * (h - 1))
    band_h = max(10, h // 28)

    sweep = frame.copy()
    cv2.rectangle(sweep, (0, max(0, y_center - band_h)), (w, min(h - 1, y_center + band_h)), (239, 174, 0), -1)  # glow suave
    cv2.addWeighted(sweep, 0.08, frame, 0.92, 0, frame)

    # 3) Viñeta ligera con líneas superpuestas (rendimiento-friendly)
    vignette = frame.copy()
    edge_cols = [
        ((0, 0, 0), 1),
        ((10, 10, 16), 1),
        ((20, 20, 28), 1)
    ]
    for i, (col, th) in enumerate(edge_cols):
        cv2.rectangle(vignette, (i, i), (w - 1 - i, h - 1 - i), col, th, cv2.LINE_AA)
    cv2.addWeighted(vignette, 0.18, frame, 0.82, 0, frame)

# ======== ZONAS DE CONTEO POR CARRILES Y GIROS ========
LANES = {
    # Carriles del primer brazo (directo)
    "Brazo 1 - Directo 1": np.array([  # Carril derecho directo
        [506, 1062],
        [1162, 890],
        [1378, 984],
        [634, 1294],
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
        [0,0],
        [0,0],
        [0,0],
        [0,0],
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
    """Carga el modelo YOLOv8 para detección de vehículos"""
    try:
        from ultralytics import YOLO
        import torch
        print("Cargando modelo YOLOv8 Medium (más preciso)...")
        # Usar yolov8m para mejor precisión (balance entre velocidad y precisión)
        model = YOLO('yolov8m.pt')  # yolov8n < yolov8s < yolov8m < yolov8l < yolov8x
        
        # Configurar para usar menos memoria
        model.overrides['verbose'] = False
        model.overrides['half'] = torch.cuda.is_available()  # FP16 si hay GPU
        
        print("Modelo YOLOv8 Medium cargado exitosamente")
        print(f"Tipo de modelo: {type(model)}")
        print(f"Usando FP16: {model.overrides['half']}")
        return model
    except Exception as e:
        print(f"Error al cargar el modelo YOLOv8: {e}")
        print("El modelo se descargará automáticamente en la primera ejecución")
        from ultralytics import YOLO
        model = YOLO('yolov8m.pt')
        return model

def get_bboxes(preds):
    """Extrae bounding boxes de las predicciones de YOLOv8 con tipos de vehículos y filtros mejorados"""
    detections = []
    
    # Clases de vehículos en COCO dataset (YOLOv8 usa las mismas clases)
    VEHICLE_CLASSES = {
        1: 'bicicleta',      # bicycle
        2: 'auto',           # car
        3: 'motocicleta',    # motorcycle
        5: 'bus',            # bus
        7: 'camion',         # truck
    }
    
    # Tamaños mínimos por tipo de vehículo (ancho x alto en píxeles)
    MIN_SIZES = {
        'bicicleta': (15, 20),
        'motocicleta': (20, 25),
        'auto': (36, 36),
        'bus': (56, 56),
        'camion': (46, 46)
    }
    
    # YOLOv8 devuelve una lista de resultados
    if len(preds) > 0:
        result = preds[0]  # Primer frame
        boxes = result.boxes  # Obtener boxes
        
        for box in boxes:
            # Filtrar por confianza y clase de vehículo
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # Threshold de confianza más estricto
            if conf >= 0.35 and cls in VEHICLE_CLASSES:
                # Obtener coordenadas [x1, y1, x2, y2]
                coords = box.xyxy[0].cpu().numpy().astype(int)
                vehicle_type = VEHICLE_CLASSES[cls]
                
                # Calcular dimensiones del bbox
                width = coords[2] - coords[0]
                height = coords[3] - coords[1]
                
                # Filtrar por tamaño mínimo
                min_w, min_h = MIN_SIZES.get(vehicle_type, (30, 30))
                if width >= min_w and height >= min_h:
                    # Calcular relación de aspecto
                    aspect_ratio = width / max(height, 1)
                    
                    # Filtrar aspectos extraños (muy alargados o muy anchos)
                    if 0.25 <= aspect_ratio <= 4.0:
                        detections.append({
                            'bbox': coords,
                            'type': vehicle_type,
                            'conf': conf,
                            'class_id': cls,
                            'width': width,
                            'height': height
                        })
    
    return detections

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

def render_results_panel(vehicle_type_counts, total_arm_1, total_arm_2, lane_counts, timestamp_text="", size=(420, 720)):
    """Panel lateral estilo dashboard moderno con glassmorphism, líneas brillantes y secciones claras."""
    width, height = size
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (15, 10, 10)  # Fondo #0A0A0F

    def rounded_rect(img, pt1, pt2, radius, color, thickness=-1):
        x1, y1 = pt1
        x2, y2 = pt2
        r = int(max(0, min(radius, min(abs(x2-x1), abs(y2-y1))//2)))
        if thickness < 0:
            cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, -1, cv2.LINE_AA)
            cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, -1, cv2.LINE_AA)
            cv2.ellipse(img, (x1+r, y1+r), (r, r), 180, 0, 90, color, -1, cv2.LINE_AA)
            cv2.ellipse(img, (x2-r, y1+r), (r, r), 270, 0, 90, color, -1, cv2.LINE_AA)
            cv2.ellipse(img, (x1+r, y2-r), (r, r), 90, 0, 90, color, -1, cv2.LINE_AA)
            cv2.ellipse(img, (x2-r, y2-r), (r, r), 0, 0, 90, color, -1, cv2.LINE_AA)
        else:
            cv2.line(img, (x1+r, y1), (x2-r, y1), color, thickness, cv2.LINE_AA)
            cv2.line(img, (x1+r, y2), (x2-r, y2), color, thickness, cv2.LINE_AA)
            cv2.line(img, (x1, y1+r), (x1, y2-r), color, thickness, cv2.LINE_AA)
            cv2.line(img, (x2, y1+r), (x2, y2-r), color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x1+r, y1+r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x2-r, y1+r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x1+r, y2-r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x2-r, y2-r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)

    # Header glass
    header_h = 64
    head = panel.copy()
    rounded_rect(head, (10, 8), (width - 10, header_h), 14, (30, 30, 45), -1)
    cv2.addWeighted(head, 0.35, panel, 0.65, 0, panel)
    # Líneas brillantes
    cv2.line(panel, (12, 10), (width - 12, 10), (255, 255, 0), 2, cv2.LINE_AA)
    cv2.line(panel, (12, header_h), (width - 12, header_h), (239, 174, 0), 1, cv2.LINE_AA)

    title = "TRAFFIC ANALYTICS"
    cv2.putText(panel, title, (20, 42), cv2.FONT_HERSHEY_DUPLEX, 0.85, (250, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(panel, title, (20, 42), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 0), 2, cv2.LINE_AA)
    if timestamp_text:
        ts_size = cv2.getTextSize(timestamp_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.putText(panel, timestamp_text, (width - ts_size[0] - 18, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 220, 240), 1, cv2.LINE_AA)

    y = header_h + 18
    # Totales
    cv2.putText(panel, "TOTAL VEHICULOS", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200, 220, 255), 1, cv2.LINE_AA)
    total_all = int(total_arm_1) + int(total_arm_2)
    total_w = cv2.getTextSize(str(total_all), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0][0]
    cv2.putText(panel, str(total_all), (width - 24 - total_w, y + 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(panel, (18, y + 10), (width - 18, y + 10), (239, 174, 0), 1, cv2.LINE_AA)

    # Detalle por brazo
    y += 36
    entries = [("Brazo 1", int(total_arm_1), (255, 255, 0)), ("Brazo 2", int(total_arm_2), (255, 0, 255))]
    for name, val, col in entries:
        cv2.circle(panel, (28, y), 7, col, -1)
        cv2.putText(panel, name, (46, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210, 220, 240), 1, cv2.LINE_AA)
        v_text = str(val)
        v_w = cv2.getTextSize(v_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]
        cv2.putText(panel, v_text, (width - 24 - v_w, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y += 28

    y += 8
    cv2.line(panel, (18, y), (width - 18, y), (60, 65, 85), 1, cv2.LINE_AA)
    y += 26

    # Individual Status (por carril)
    cv2.putText(panel, "INDIVIDUAL STATUS", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 255), 1, cv2.LINE_AA)
    y += 18

    # Colores fijos por carril
    def color_for_lane(name: str):
        base = sum(ord(c) for c in name) % 180
        palette = [
            (255, 255, 0),  # cian
            (255, 0, 255),  # magenta
            (200, 200, 255),
            (180, 105, 255),
            (60, 220, 255),
            (0, 255, 180),
            (255, 140, 60),
        ]
        return palette[base % len(palette)]

    sorted_lanes = sorted(list(lane_counts.items()), key=lambda kv: (0 if kv[0].startswith('Brazo 1') else 1, kv[0]))
    for lane_name, count in sorted_lanes:
        col = color_for_lane(lane_name)
        cv2.circle(panel, (28, y), 6, col, -1)
        cv2.putText(panel, lane_name, (46, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 220, 240), 1, cv2.LINE_AA)
        status = "ACTIVE" if int(count) > 0 else "IDLE"
        status_col = (0, 220, 0) if status == "ACTIVE" else (140, 150, 165)
        s_w = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
        cv2.putText(panel, status, (width - 24 - s_w, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_col, 1, cv2.LINE_AA)
        # barra
        bar_x0 = 30
        bar_w = width - 44 - 70
        bar_h = 6
        cv2.rectangle(panel, (bar_x0, y + 10), (bar_x0 + bar_w, y + 10 + bar_h), (45, 50, 70), -1)
        arm_total = total_arm_1 if lane_name.startswith('Brazo 1') else total_arm_2
        frac = 0 if int(arm_total) == 0 else min(1.0, float(count) / float(arm_total))
        cv2.rectangle(panel, (bar_x0, y + 10), (bar_x0 + int(frac * bar_w), y + 10 + bar_h), col, -1)
        y += 28
        if y > height - 260:
            break

    # Sección tipos de vehículo
    cv2.line(panel, (18, height - 228), (width - 18, height - 228), (60, 65, 85), 1, cv2.LINE_AA)
    cv2.putText(panel, "TIPOS DE VEHICULO", (20, height - 200), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 220, 255), 1, cv2.LINE_AA)

    type_order = ['auto', 'bus', 'camion', 'motocicleta', 'bicicleta']
    type_colors = {
        'auto': (255, 255, 0),
        'bus': (239, 174, 0),
        'camion': (0, 0, 255),
        'motocicleta': (255, 0, 255),
        'bicicleta': (0, 255, 0)
    }
    values = [int(vehicle_type_counts.get(k, 0)) for k in type_order]
    max_val = max([1] + values)
    bx, by = 24, height - 176
    mini_w, mini_h, gap = (width - 48), 10, 8
    for i, k in enumerate(type_order):
        v = int(vehicle_type_counts.get(k, 0))
        frac = 0 if max_val == 0 else v / max_val
        y_row = by + i * (mini_h + gap)
        cv2.putText(panel, k.upper(), (bx, y_row - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (185, 205, 235), 1, cv2.LINE_AA)
        cv2.rectangle(panel, (bx, y_row + 6), (bx + mini_w, y_row + 6 + mini_h), (50, 55, 75), -1)
        cv2.rectangle(panel, (bx, y_row + 6), (bx + int(frac * mini_w), y_row + 6 + mini_h), type_colors.get(k, (200, 200, 200)), -1)

    # Botón de exportar (mantener geometría usada por el callback)
    btn_x = 24
    btn_y = height - 60
    btn_w = width - 48
    btn_h = 35

    button = panel.copy()
    rounded_rect(button, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), 10, (30, 60, 60), -1)
    cv2.addWeighted(button, 0.55, panel, 0.45, 0, panel)
    cv2.rectangle(panel, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), (255, 255, 0), 2, cv2.LINE_AA)

    btn_text = "EXPORTAR RESULTADOS"
    text_size = cv2.getTextSize(btn_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = btn_x + (btn_w - text_size[0]) // 2
    text_y = btn_y + (btn_h + text_size[1]) // 2
    cv2.putText(panel, btn_text, (text_x+1, text_y+1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(panel, btn_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (250, 245, 245), 2, cv2.LINE_AA)

    # Pie
    cv2.putText(panel, "Actualizado en tiempo real", (18, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 205, 255), 1, cv2.LINE_AA)
    return panel

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
    # Optimizaciones de OpenCV para mejor rendimiento
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)  # Ajustar según núcleos disponibles
    
    # Cargar el modelo YOLO
    model = load_model()
    
    # Verificar que el modelo sea callable
    if not callable(model):
        raise TypeError(f"El modelo no es callable. Tipo: {type(model)}")
    
    frame_count = 0
    # Inicializar contadores por carril
    lane_counts = {lane: 0 for lane in LANES.keys()}
    total_count = 0
    
    # Inicializar SORT tracker
    tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
    
    # Diccionario para almacenar información de vehículos rastreados
    tracked_vehicles = {}  # {track_id: {'type': ..., 'counted_lanes': set(), 'last_seen': frame_count}}
    
    # Parámetros de optimización
    MAX_TRACKED_VEHICLES = 200  # Límite máximo de vehículos en memoria
    CLEANUP_INTERVAL = 100  # Limpiar cada N frames
    
    # Contadores por tipo de vehículo
    vehicle_type_counts = {
        'auto': 0,
        'bus': 0,
        'camion': 0,
        'motocicleta': 0,
        'bicicleta': 0
    }
    
    # Variables de control de video
    paused = False
    speed = 1.0  # Factor de velocidad
    show_info = True  # Mostrar información de ajustes
    end_reached = False  # Fin de video alcanzado
    last_frame_raw = None  # Último frame válido para mostrar al finalizar

    # Variables para funcionalidad de botones
    button_rects = []
    export_button_rect = None
    
    # Función para exportar resultados
    def export_results():
        import pandas as pd
        from datetime import datetime
        import os
        import tkinter as tk
        from tkinter import filedialog
        
        # Crear ventana oculta para el diálogo
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        # Abrir diálogo para seleccionar carpeta
        save_dir = filedialog.askdirectory(
            title="Seleccionar carpeta para guardar resultados",
            initialdir=os.getcwd()
        )
        
        root.destroy()
        
        # Si el usuario cancela, no hacer nada
        if not save_dir:
            print("\n❌ Exportación cancelada")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Crear subdirectorio con timestamp dentro de la carpeta seleccionada
        output_dir = os.path.join(save_dir, f'resultados_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)
        
        # Exportar conteo por tipo de vehículo
        df_types = pd.DataFrame(list(vehicle_type_counts.items()), columns=['Tipo', 'Cantidad'])
        filename_types = os.path.join(output_dir, 'conteo_por_tipo.csv')
        df_types.to_csv(filename_types, index=False, encoding='utf-8-sig')
        
        # Exportar conteo por carril
        df_lanes = pd.DataFrame(list(lane_counts.items()), columns=['Carril', 'Cantidad'])
        filename_lanes = os.path.join(output_dir, 'conteo_por_carril.csv')
        df_lanes.to_csv(filename_lanes, index=False, encoding='utf-8-sig')
        
        # Exportar resumen general
        total_arm_1 = sum(v for k, v in lane_counts.items() if k.startswith('Brazo 1'))
        total_arm_2 = sum(v for k, v in lane_counts.items() if k.startswith('Brazo 2'))
        
        df_summary = pd.DataFrame({
            'Concepto': ['Total General', 'Total Brazo 1', 'Total Brazo 2'],
            'Cantidad': [total_count, total_arm_1, total_arm_2]
        })
        filename_summary = os.path.join(output_dir, 'resumen_general.csv')
        df_summary.to_csv(filename_summary, index=False, encoding='utf-8-sig')
        
        print(f"\n✅ Resultados exportados exitosamente en:")
        print(f"   📁 {output_dir}")
        print(f"   - conteo_por_tipo.csv")
        print(f"   - conteo_por_carril.csv")
        print(f"   - resumen_general.csv")
    
    # Callback para clicks del mouse en ventana HUD
    def mouse_callback(event, x, y, flags, param):
        nonlocal paused, end_reached, last_frame_raw
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, rect in enumerate(button_rects):
                if rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                    if i == 0:  # PLAY
                        # Si ya terminó el video, reiniciar desde el inicio
                        if end_reached:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            end_reached = False
                            last_frame_raw = None
                        paused = False
                        print("\n▶ Video REPRODUCIENDO")
                    elif i == 1:  # PAUSE
                        paused = True
                        print("\n⏸ Video PAUSADO")
                    elif i == 2:  # STOP
                        print("\n⏹ DETENIENDO...")
                        cap.release()
                        cv2.destroyAllWindows()
                        sys.exit(0)
    
    # Callback para clicks del mouse en ventana RESULTADOS
    def mouse_callback_results(event, x, y, flags, param):
        nonlocal export_button_rect
        if event == cv2.EVENT_LBUTTONDOWN:
            if export_button_rect is not None:
                x1, y1, x2, y2 = export_button_rect
                if x1 <= x <= x2 and y1 <= y <= y2:
                    export_results()

    print("\nControles del video:")
    print("p - Pausar/Reanudar")
    print("+ - Aumentar velocidad")
    print("- - Reducir velocidad")
    print("i - Mostrar/Ocultar información")
    print("r - Resetear ajustes")
    print("q - Salir")
    print("\nTambién puedes usar los botones PLAY, PAUSE y STOP con el mouse")

    # Preparar ventanas con tamaño por ventana
    cv2.namedWindow("FUTURISTIC TRAFFIC HUD", cv2.WINDOW_NORMAL)
    cv2.namedWindow("RESULTADOS", cv2.WINDOW_NORMAL)
    
    # Configurar callback del mouse
    cv2.setMouseCallback("FUTURISTIC TRAFFIC HUD", mouse_callback)
    cv2.setMouseCallback("RESULTADOS", mouse_callback_results)
    
    # Establecer tamaños independientes
    HUD_WIDTH, HUD_HEIGHT = HUD_WINDOW_SIZE
    RES_WIDTH, RES_HEIGHT = RESULTS_PANEL_SIZE
    cv2.resizeWindow("FUTURISTIC TRAFFIC HUD", HUD_WIDTH, HUD_HEIGHT)
    cv2.resizeWindow("RESULTADOS", RES_WIDTH, RES_HEIGHT)
    
    # Definir coordenadas del botón de exportar en el panel de resultados
    export_button_rect = (24, RES_HEIGHT - 60, RES_WIDTH - 24, RES_HEIGHT - 25)
    
    # Obtener información del video para el cronómetro
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    total_duration_seconds = total_frames / fps_video if fps_video > 0 else 0
    
    # Variable para posicionar ventanas una vez
    windows_positioned = False
    
    # Helper para obtener tamaño de pantalla
    def _get_screen_size():
        # Por defecto, intentar un tamaño de pantalla común
        screen_w, screen_h = 1920, 1080
        try:
            if windll is not None and sys.platform.startswith('win'):
                user32 = windll.user32
                user32.SetProcessDPIAware()
                screen_w = user32.GetSystemMetrics(0)
                screen_h = user32.GetSystemMetrics(1)
        except Exception:
            pass
        return screen_w, screen_h

    # Logs de rendimiento
    import time
    import gc
    last_time = time.time()
    avg_fps = 0
    frame_times = []
    gc_counter = 0  # Contador para garbage collection periódico
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if gpu_available else 'CPU'
    except Exception:
        gpu_available = False
        device_name = 'CPU'
    while cap.isOpened():
        t0 = time.time()
        # Leer frame (si estamos en pausa mantenemos el último frame)
        if not paused:
            status, frame_raw = cap.read()
            if not status:
                # Fin del video: pausar y permitir exportación
                paused = True
                end_reached = True
                # Mantener el último frame para mostrar
                if last_frame_raw is None:
                    last_frame_raw = np.zeros((HUD_HEIGHT, HUD_WIDTH, 3), dtype=np.uint8)
                # Usar el último frame para seguir renderizando
                frame_raw = last_frame_raw
            else:
                last_frame_raw = frame_raw

            # Posicionar ventanas lado a lado (solo una vez)
            if not windows_positioned:
                HUD_WIDTH, HUD_HEIGHT = HUD_WINDOW_SIZE
                # Colocar ambas ventanas una al lado de la otra con un pequeño espacio
                cv2.moveWindow("FUTURISTIC TRAFFIC HUD", 10, 10)
                cv2.moveWindow("RESULTADOS", HUD_WIDTH + 20, 10)
                windows_positioned = True

            # Ejecutar detector sobre el frame RAW ORIGINAL (mejor precisión)
            if not end_reached:
                # Usar torch.no_grad() para reducir uso de memoria y mejorar velocidad
                import torch
                with torch.no_grad():
                    # Configurar para máxima eficiencia
                    preds = model(frame_raw, verbose=False, stream=False, half=torch.cuda.is_available())
                bboxes = get_bboxes(preds)
                # Liberar predicciones inmediatamente
                del preds
            else:
                bboxes = []

        # Tamaños de resolución
        raw_h, raw_w = frame_raw.shape[0], frame_raw.shape[1]
        HUD_WIDTH, HUD_HEIGHT = HUD_WINDOW_SIZE
        
        # Frame redimensionado para HUD (solo para visualización)
        frame = cv2.resize(frame_raw, (HUD_WIDTH, HUD_HEIGHT), interpolation=cv2.INTER_AREA)

        # Calcular zonas: LANES está definido en coordenadas del video original
        # Para conteo usamos coordenadas del frame raw original
        # Para dibujo escalamos al tamaño del HUD
        lanes_count = {}
        lanes_draw = {}
        for lane_name, lane_points in LANES.items():
            # Las zonas ya están en coords del video original, usarlas directamente para conteo
            lanes_count[lane_name] = lane_points
            # Para dibujo: escalar a HUD
            scale_x = HUD_WIDTH / float(raw_w)
            scale_y = HUD_HEIGHT / float(raw_h)
            lanes_draw[lane_name] = (lane_points * np.array([scale_x, scale_y])).astype(int)

        # Preparar detecciones para SORT: [x1, y1, x2, y2, conf]
        detections_for_sort = np.array([[d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3], d['conf']] 
                                         for d in bboxes]) if len(bboxes) > 0 else np.empty((0, 5))
        
        # Actualizar tracker SORT
        tracked_objects = tracker.update(detections_for_sort)
        
        # Procesar objetos rastreados
        for trk in tracked_objects:
            x1, y1, x2, y2, track_id = trk
            track_id = int(track_id)
            box = [int(x1), int(y1), int(x2), int(y2)]
            xc, yc = get_center(box)
            
            # Buscar el tipo de vehículo de la detección más cercana
            vehicle_type = 'auto'  # Por defecto
            conf = 0.5
            if len(bboxes) > 0:
                # Encontrar la detección más cercana al track
                min_dist = float('inf')
                for detection in bboxes:
                    det_box = detection['bbox']
                    det_xc, det_yc = get_center(det_box)
                    dist = np.sqrt((xc - det_xc)**2 + (yc - det_yc)**2)
                    if dist < min_dist:
                        min_dist = dist
                        vehicle_type = detection['type']
                        conf = detection['conf']
            
            # Inicializar info del vehículo si es nuevo
            if track_id not in tracked_vehicles:
                tracked_vehicles[track_id] = {
                    'type': vehicle_type,
                    'counted_lanes': set(),
                    'first_seen': frame_count,
                    'last_seen': frame_count
                }
            
            # Actualizar tipo y último frame visto
            tracked_vehicles[track_id]['type'] = vehicle_type
            tracked_vehicles[track_id]['last_seen'] = frame_count
            
            # Verificar en qué carril está y si debe contarse
            for lane_name, lane_zone in lanes_count.items():
                is_in_lane = is_valid_detection(xc, yc, lane_zone)
                
                if is_in_lane and lane_name not in tracked_vehicles[track_id]['counted_lanes']:
                    # Contar solo si no se ha contado en este carril antes
                    lane_counts[lane_name] += 1
                    total_count += 1
                    tracked_vehicles[track_id]['counted_lanes'].add(lane_name)
                    
                    # Contar por tipo de vehículo
                    if vehicle_type in vehicle_type_counts:
                        vehicle_type_counts[vehicle_type] += 1

            # Colores por tipo de vehículo
            type_colors = {
                'auto': (255, 255, 0),      # Amarillo
                'bus': (0, 165, 255),       # Naranja
                'camion': (0, 0, 255),      # Rojo
                'motocicleta': (255, 0, 255), # Magenta
                'bicicleta': (0, 255, 0)    # Verde
            }
            color = type_colors.get(vehicle_type, (255, 255, 255))
            
            # Escalar bbox y centro desde RAW a HUD para dibujar
            scale_x = HUD_WIDTH / float(raw_w)
            scale_y = HUD_HEIGHT / float(raw_h)
            box_disp = [int(box[0] * scale_x), int(box[1] * scale_y), 
                       int(box[2] * scale_x), int(box[3] * scale_y)]
            xc_disp, yc_disp = int(xc * scale_x), int(yc * scale_y)
            
            # Rectángulo con color según tipo (en HUD)
            cv2.rectangle(frame, (box_disp[0], box_disp[1]), (box_disp[2], box_disp[3]), color, 2)
            # Punto central
            cv2.circle(frame, (xc_disp, yc_disp), 6, color, -1)
            
            # Etiqueta del tipo de vehículo con ID de tracking
            label = f"ID:{track_id} {vehicle_type} {conf:.2f}"
            cv2.putText(frame, label, (box_disp[0], box_disp[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        
        frame_count += 1
        
        # Limpieza periódica de tracked_vehicles para evitar fugas de memoria
        if frame_count % CLEANUP_INTERVAL == 0:
            # Eliminar vehículos que no se han visto en los últimos N frames
            stale_ids = [tid for tid, info in tracked_vehicles.items() 
                        if frame_count - info['last_seen'] > 100]
            for tid in stale_ids:
                del tracked_vehicles[tid]
            
            # Si aún hay demasiados vehículos, mantener solo los más recientes
            if len(tracked_vehicles) > MAX_TRACKED_VEHICLES:
                sorted_vehicles = sorted(tracked_vehicles.items(), 
                                       key=lambda x: x[1]['last_seen'], 
                                       reverse=True)
                tracked_vehicles = dict(sorted_vehicles[:MAX_TRACKED_VEHICLES])
            
            # Garbage collection periódico cada 500 frames (~16 segundos a 30fps)
            gc_counter += 1
            if gc_counter >= 5:  # Cada 500 frames
                gc.collect()
                gc_counter = 0
        
        # Logs de rendimiento
        t1 = time.time()
        frame_time = (t1 - t0) * 1000  # ms
        frame_times.append(frame_time)
        if len(frame_times) > 60:
            frame_times.pop(0)
        avg_fps = 1000 / (sum(frame_times) / len(frame_times)) if frame_times else 0
        
        # Solo mostrar logs cada 30 frames para no saturar consola
        if frame_count % 30 == 0:
            print(f"[Rendimiento] Frame {frame_count}: {frame_time:.1f} ms | FPS: {avg_fps:.1f} | Dispositivo: {device_name}")

        # Aplicar efecto de fondo oscuro con vignette
        add_hud_effects(frame)

        # ===== ELEMENTOS HUD =====
        # Panel Total Cars con información de precisión
        draw_hud_element(frame, 50, 40, 400, 80, "TOTAL", total_count, frame_count)
        
        # Mostrar información del modelo y precisión
        model_info = "YOLOv8-Medium | Conf: 0.35+ | Optimizado"
        cv2.putText(frame, model_info, (50, 135),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        
        # === PANEL DE TIPOS DE VEHÍCULOS ===
        # Posición en el lado derecho de la pantalla
        vehicle_panel_x = frame.shape[1] - 470
        vehicle_panel_y = 80
        
        # Título del panel
        draw_hud_element(frame, vehicle_panel_x, vehicle_panel_y, 400, 80, 
                        "TIPOS DE VEHICULOS", "", frame_count)
        vehicle_panel_y += 100
        
        # Conteos por tipo con iconos y colores
        type_colors_hud = {
            'auto': (255, 255, 0),
            'bus': (0, 165, 255),
            'camion': (0, 0, 255),
            'motocicleta': (255, 0, 255),
            'bicicleta': (0, 255, 0)
        }
        
        for vtype, count in vehicle_type_counts.items():
            if count > 0:  # Solo mostrar tipos con detecciones
                # Dibujar panel pequeño para cada tipo
                overlay = frame.copy()
                cv2.rectangle(overlay, 
                            (vehicle_panel_x, vehicle_panel_y), 
                            (vehicle_panel_x + 400, vehicle_panel_y + 50), 
                            (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                
                color = type_colors_hud.get(vtype, (255, 255, 255))
                
                # Circulo de color
                cv2.circle(frame, (vehicle_panel_x + 30, vehicle_panel_y + 25), 15, color, -1)
                
                # Texto
                text = f"{vtype.upper()}: {count}"
                cv2.putText(frame, text, 
                          (vehicle_panel_x + 60, vehicle_panel_y + 35),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                
                vehicle_panel_y += 60
        
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
        for lane_name, lane_zone in lanes_draw.items():
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

        # Construir panel de resultados (panel analítico estilo sidebar)
        try:
            panel_img = render_results_panel(
                vehicle_type_counts,
                total_arm_1,
                total_arm_2,
                lane_counts,
                timestamp_text=time.strftime("%Y-%m-%d  %H:%M:%S"),
                size=RESULTS_PANEL_SIZE
            )
            cv2.imshow("RESULTADOS", panel_img)
            # Posicionamiento ya se maneja arriba en el primer frame
        except Exception:
            # Si aún no se han calculado totales, omitir hasta el siguiente frame
            pass

        # Mostrar información de ajustes
        if show_info:
            info_color = (0, 255, 255)
            cv2.putText(frame, f"Velocidad: x{speed:.1f}", (10, frame.shape[0] - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, info_color, 2, cv2.LINE_AA)
            status_text = "FINALIZADO" if end_reached else ("PAUSADO" if paused else "REPRODUCIENDO")
            cv2.putText(frame, f"Estado: {status_text}", (10, frame.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, info_color, 2, cv2.LINE_AA)

        #Fondo más oscuro para el efecto global-fondo de pantalla
        overlay_bg = frame.copy()
        cv2.rectangle(overlay_bg, (0, 0), (frame.shape[1], frame.shape[0]),
                      (0, 20, 50), -1)
        cv2.addWeighted(overlay_bg, 0.15, frame, 0.85, 0, frame)  # para poner el color de fondo y transparencia

        # ===== Panel de controles futurista profesional =====
        label_h = 60
        label_w = frame.shape[1]
        label_img = np.zeros((label_h, label_w, 3), dtype=np.uint8)
        
        # Fondo oscuro elegante con gradiente vertical
        for j in range(label_h):
            grad = j / label_h
            color = (
                int(15 * (1 - grad) + 25 * grad),
                int(15 * (1 - grad) + 30 * grad),
                int(20 * (1 - grad) + 40 * grad)
            )
            cv2.line(label_img, (0, j), (label_w, j), color, 1)
        
        # Línea neón superior brillante
        cv2.line(label_img, (0, 0), (label_w, 0), (0, 255, 255), 4)
        cv2.line(label_img, (0, 3), (label_w, 3), (0, 180, 255), 2)
        
        # Calcular tiempo transcurrido y duración total
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        current_time_seconds = current_frame / fps_video if fps_video > 0 else 0
        
        # Formatear tiempos en MM:SS
        def format_time(seconds):
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins:02d}:{secs:02d}"
        
        current_time_str = format_time(current_time_seconds)
        total_time_str = format_time(total_duration_seconds)
        timer_text = f"{current_time_str} / {total_time_str}"
        
        # Mostrar cronómetro en la esquina izquierda
        cv2.putText(label_img, timer_text, (20, 38), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(label_img, timer_text, (20, 38), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        
        # Botones modernos centrados
        btn_width = 150
        btn_height = 35
        btn_gap = 25
        total_btn_width = (btn_width * 3) + (btn_gap * 2)
        start_x = (label_w - total_btn_width) // 2
        btn_y = 13
        
        buttons = [
            {"text": "PLAY", "color": (100, 255, 100), "bg": (30, 80, 30)},
            {"text": "PAUSE", "color": (255, 200, 100), "bg": (80, 60, 30)},
            {"text": "STOP", "color": (255, 100, 100), "bg": (80, 30, 30)}
        ]
        
        # Limpiar lista de rectángulos de botones
        button_rects.clear()
        
        for i, btn in enumerate(buttons):
            x = start_x + i * (btn_width + btn_gap)
            y1 = btn_y
            y2 = btn_y + btn_height
            
            # Guardar las coordenadas del botón (ajustadas para la ventana combinada)
            # Los botones están en la parte inferior (después del frame)
            btn_y_offset = frame.shape[0]  # Offset del video
            button_rects.append((x, btn_y_offset + y1, x + btn_width, btn_y_offset + y2))
            
            # Determinar si el botón está activo
            is_active = False
            if i == 0 and not paused:  # PLAY activo
                is_active = True
            elif i == 1 and paused:  # PAUSE activo
                is_active = True
            
            # Fondo del botón con transparencia (más brillante si está activo)
            overlay = label_img.copy()
            if is_active:
                cv2.rectangle(overlay, (x, y1), (x + btn_width, y2), btn["bg"], -1)
                cv2.addWeighted(overlay, 0.8, label_img, 0.2, 0, label_img)
            else:
                cv2.rectangle(overlay, (x, y1), (x + btn_width, y2), btn["bg"], -1)
                cv2.addWeighted(overlay, 0.5, label_img, 0.5, 0, label_img)
            
            # Borde brillante del botón (más grueso si está activo)
            border_thickness = 3 if is_active else 2
            cv2.rectangle(label_img, (x, y1), (x + btn_width, y2), btn["color"], border_thickness, cv2.LINE_AA)
            
            # Texto del botón centrado
            text_size_btn = cv2.getTextSize(btn["text"], cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x_btn = x + (btn_width - text_size_btn[0]) // 2
            text_y_btn = y1 + (btn_height + text_size_btn[1]) // 2
            
            # Sombra del texto
            cv2.putText(label_img, btn["text"], (text_x_btn+2, text_y_btn+2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            # Texto principal
            cv2.putText(label_img, btn["text"], (text_x_btn, text_y_btn), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, btn["color"], 2, cv2.LINE_AA)
        # Combinar video y label (label abajo)
        combined_img = np.vstack([frame, label_img])
        cv2.imshow("FUTURISTIC TRAFFIC HUD", combined_img)
        
        # Limpiar variables temporales para evitar fugas de memoria
        if 'lanes_count' in locals():
            lanes_count.clear()
        if 'lanes_draw' in locals():
            lanes_draw.clear()
        if 'bboxes' in locals():
            del bboxes
        if 'detections_for_sort' in locals():
            del detections_for_sort
        if 'tracked_objects' in locals():
            del tracked_objects
        # Liberar memoria de overlays
        del label_img
        if 'overlay_bg' in locals():
            del overlay_bg

        # Control de teclas
        # Usar delay mínimo de 10ms incluso cuando está pausado para permitir eventos del mouse
        delay = int(25/speed) if not paused else 10
        key = cv2.waitKey(delay) & 0xFF
        
        if key == ord('q'):
            print("\nSaliendo...")
            break
        elif key == ord('p'):
            # Si estaba finalizado y se quiere reanudar, reiniciar desde el inicio
            if end_reached:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                end_reached = False
            paused = not paused
            status = "PAUSADO" if paused else "REPRODUCIENDO"
            print(f"\nVideo {status}")
        elif key == ord('+') or key == ord('='): # Aumentar velocidad
            speed = min(speed + 0.2, 4.0)
            print(f"\nVelocidad: x{speed:.1f}")
        elif key == ord('-'): # Reducir velocidad
            speed = max(speed - 0.2, 0.2)
            print(f"\nVelocidad: x{speed:.1f}")
        elif key == ord('i'): # Mostrar/ocultar información
            show_info = not show_info
            print("\nInformación", "visible" if show_info else "oculta")
        elif key == ord('r'): # Resetear ajustes
            speed = 1.0
            show_info = True
            print("\nAjustes restaurados")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[Rendimiento final] FPS promedio: {avg_fps:.1f} | Dispositivo: {device_name}")


if __name__ == '__main__':
    import argparse, os, json
    parser = argparse.ArgumentParser(description='Conteo de vehículos con HUD futurista')
    parser.add_argument('--video', type=str, default='recursos/videos/Video_int1.mp4', help='Ruta del video a analizar')
    parser.add_argument('--coords', type=str, default='', help='Ruta a JSON con puntos del polígono (lista de [x,y])')
    args = parser.parse_args()

    # Si se proporcionan coordenadas externas, cargar y reemplazar las zonas
    if args.coords:
        try:
            if os.path.exists(args.coords):
                with open(args.coords, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                pts = data.get('points', data)
                # Normalizar a lista de pares [x,y]
                if isinstance(pts, list) and len(pts) >= 3:
                    norm = []
                    for p in pts:
                        if isinstance(p, (list, tuple)) and len(p) >= 2:
                            norm.append([int(p[0]), int(p[1])])
                    if len(norm) >= 3:
                        LANES.clear()
                        LANES['Zona Seleccionada'] = np.array(norm, dtype=int)
                        print(f"Zonas cargadas desde {args.coords}: {len(norm)} puntos")
                # Si el JSON incluye 'video', preferirlo si no se pasó --video explícito
                if not (args.video and args.video.strip()) and isinstance(data, dict) and 'video' in data:
                    args.video = data['video']
        except Exception as e:
            print(f"No se pudieron cargar coordenadas desde {args.coords}: {e}")

    video_path = args.video or 'recursos/videos/Video_int1.mp4'
    if not os.path.exists(video_path):
        print(f"Advertencia: no se encuentra el video {video_path}, se usará la ruta por defecto si existe.")
    cap = cv2.VideoCapture(video_path)
    detector(cap) 