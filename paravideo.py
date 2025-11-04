import os
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from typing import Any

try:
    import tkintervideo as _tkv
    TkinterVideoCls = getattr(_tkv, 'TkinterVideo', None)
except Exception:
    TkinterVideoCls = None


class VideoPlayer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Reproductor de Video – Prometheo")
        self.geometry("900x560")
        self.configure(bg="#0a0a0f")

        self.use_tkv = TkinterVideoCls is not None
        self.video_cargado = False
        self.is_paused = False
        self.video_path = None  # Guardar ruta del video cargado

        self.app_dir = os.path.dirname(os.path.abspath(__file__))
        self.videos_dir = os.path.join(self.app_dir, "recursos", "videos")
        self.last_dir = self.videos_dir if os.path.isdir(self.videos_dir) else os.path.expanduser("~")

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # --- Encabezado ---
        tk.Label(self, text="🎬 Reproductor Futurista", fg="#00ffaa", bg="#0a0a0f",
                 font=("Consolas", 16, "bold")).pack(pady=(10, 8))

        toolbar = tk.Frame(self, bg="#0a0a0f")
        toolbar.pack(fill="x", padx=12, pady=(0, 8))

        tk.Label(toolbar, text="Video:", fg="#ffffff", bg="#0a0a0f",
                 font=("Consolas", 10, "bold")).pack(side="left", padx=(0, 8))

        self.video_var = tk.StringVar()
        self.cbo_video = ttk.Combobox(toolbar, textvariable=self.video_var, state="readonly", width=60)
        self.cbo_video.pack(side="left", padx=(0, 8))
        self.cbo_video.bind("<<ComboboxSelected>>", self._on_combo_select)
        self._refresh_combo()

        tk.Button(toolbar, text="Examinar…", command=self.cargar_video,
                  bg="#2affd5", fg="#0a0a0f", font=("Consolas", 9, "bold"), width=12).pack(side="left")

        # --- Frame contenedor principal ---
        main_frame = tk.Frame(self, bg="#0a0a0f")
        main_frame.pack(fill="both", expand=True)

        # --- Frame de video (altura fija controlada) ---
        video_container = tk.Frame(main_frame, bg="#000000", height=400)
        video_container.pack(fill="x", padx=20, pady=(0, 10))
        video_container.pack_propagate(False)  # <--- evita que se expanda

        if self.use_tkv:
            self.videoplayer = TkinterVideoCls(master=video_container, scaled=True, pre_load=False)  # type: ignore
            self.videoplayer.pack(fill="both", expand=True)
        else:
            self.video_label = tk.Label(video_container, bg="#000000")
            self.video_label.pack(fill="both", expand=True)
            self.cap: Any = None
            self._job_id = None
            self._frame_image = None
            self._fps = 30.0
            self._delay_ms = 33

        # --- Barra de controles ---
        btn_frame = tk.Frame(self, bg="#0a0a0f")
        btn_frame.pack(side="bottom", fill="x", pady=(5, 12))

        self.btn_open = tk.Button(btn_frame, text="Cargar Video", command=self.cargar_video,
                                  bg="#2affd5", fg="#0a0a0f", font=("Consolas", 10, "bold"), width=15)
        self.btn_open.grid(row=0, column=0, padx=10)

        self.btn_play = tk.Button(btn_frame, text="▶ Reproducir", command=self.play_video,
                                  bg="#00ffaa", fg="#0a0a0f", font=("Consolas", 10, "bold"), width=15)
        self.btn_play.grid(row=0, column=1, padx=10)

        self.btn_pause = tk.Button(btn_frame, text="⏸ Pausar", command=self.pause_video,
                                   bg="#ff00ff", fg="#ffffff", font=("Consolas", 10, "bold"), width=15)
        self.btn_pause.grid(row=0, column=2, padx=10)

        self.btn_stop = tk.Button(btn_frame, text="⏹ Detener", command=self.stop_video,
                                  bg="#ff5555", fg="#ffffff", font=("Consolas", 10, "bold"), width=15)
        self.btn_stop.grid(row=0, column=3, padx=10)

        self.btn_next = tk.Button(btn_frame, text="Siguiente ▶", command=self.siguiente,
                                  bg="#ffaa00", fg="#0a0a0f", font=("Consolas", 10, "bold"), width=15)
        self.btn_next.grid(row=0, column=4, padx=10)

        # Atajos
        self.bind("<space>", self._toggle_play_pause)
        self.bind("<Escape>", self._esc_stop)
        self.bind("<Control-o>", lambda e: self.cargar_video())

    # --- Métodos ---
    def siguiente(self):
        if not self.video_path:
            messagebox.showwarning("Sin video", "Carga un video antes de continuar.")
            return
        
        # Guardar ruta del video en un archivo temporal para coordinates.py
        import json
        config_path = os.path.join(self.app_dir, 'recursos', 'video_selected.json')
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({'video': self.video_path}, f, ensure_ascii=False, indent=2)
        
        print(f"Video seleccionado guardado: {self.video_path}")
        self.on_close()
        
        import subprocess, sys
        ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "coordinates.py")
        if os.path.exists(ruta):
            subprocess.run([sys.executable, ruta, '--video', self.video_path])
        else:
            print(f"No se encontró coordinates.py en {ruta}")

    def _refresh_combo(self):
        try:
            exts = (".mp4", ".avi", ".mkv", ".mov", ".webm", ".mpg")
            if os.path.isdir(self.videos_dir):
                files = [f for f in os.listdir(self.videos_dir) if f.lower().endswith(exts)]
                self.cbo_video["values"] = sorted(files)
                if files and not self.video_var.get():
                    self.video_var.set(files[0])
        except Exception:
            self.cbo_video["values"] = []

    def _on_combo_select(self, event=None):
        name = self.video_var.get().strip()
        if not name:
            return
        ruta = os.path.join(self.videos_dir, name)
        if os.path.exists(ruta):
            self.cargar_video(ruta)
        else:
            messagebox.showwarning("No encontrado", f"El archivo ya no existe:\n{ruta}")

    def cargar_video(self, ruta: str | None = None):
        try:
            if not ruta:
                ruta = filedialog.askopenfilename(
                    title="Seleccionar Video",
                    initialdir=self.last_dir,
                    filetypes=[("Archivos de video", "*.mp4 *.avi *.mkv *.mov *.webm *.mpg")]
                )
                if not ruta:
                    return

            if self.video_cargado:
                try:
                    if self.use_tkv:
                        self.videoplayer.stop()
                    else:
                        self._cv2_stop()
                except Exception:
                    pass

            if self.use_tkv:
                self.videoplayer.load(ruta)
            else:
                self._cv2_load(ruta)

            self.video_cargado = True
            self.is_paused = False
            self.video_path = ruta  # Guardar ruta para pasarla a siguiente script
            self.title(f"Reproductor de Video – {os.path.basename(ruta)}")

        except Exception as e:
            messagebox.showerror("Error al cargar", f"No se pudo cargar el video.\n\n{e}")

    def play_video(self):
        if not self.video_cargado:
            messagebox.showinfo("Sin video", "Cargue o seleccione un video antes de reproducir.")
            return
        if self.use_tkv:
            self.videoplayer.play()
        else:
            self._cv2_play()
        self.is_paused = False

    def pause_video(self):
        if not self.video_cargado:
            return
        if self.use_tkv:
            self.videoplayer.pause()
        else:
            self._cv2_pause()
        self.is_paused = True

    def stop_video(self):
        if not self.video_cargado:
            return
        if self.use_tkv:
            self.videoplayer.stop()
        else:
            self._cv2_stop()
        self.is_paused = False

    def _toggle_play_pause(self, event=None):
        if not self.video_cargado:
            return
        self.play_video() if self.is_paused else self.pause_video()

    def _esc_stop(self, event=None):
        if not self.video_cargado:
            self.on_close()
        else:
            self.stop_video()

    def on_close(self):
        if self.video_cargado:
            if self.use_tkv:
                self.videoplayer.stop()
            else:
                self._cv2_stop()
        self.destroy()

    # Fallback OpenCV
    def _cv2_load(self, ruta: str):
        import cv2
        if getattr(self, 'cap', None) is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(ruta)
        if not self.cap.isOpened():
            raise RuntimeError("No se pudo abrir el video con OpenCV")
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._fps = float(fps)
        self._delay_ms = max(10, int(1000.0 / self._fps))

    def _cv2_play(self):
        if self._job_id is None:
            self._update_frame()

    def _cv2_pause(self):
        if self._job_id is not None:
            self.after_cancel(self._job_id)
            self._job_id = None

    def _cv2_stop(self):
        self._cv2_pause()
        if getattr(self, 'cap', None) is not None:
            self.cap.release()
        if hasattr(self, 'video_label'):
            self.video_label.configure(image='')
        self._frame_image = None

    def _update_frame(self):
        import cv2
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if not ok:
                return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        label_w = self.video_label.winfo_width()
        label_h = self.video_label.winfo_height()
        fh, fw = frame_rgb.shape[:2]
        scale = min(label_w / fw, label_h / fh)
        resized = cv2.resize(frame_rgb, (int(fw * scale), int(fh * scale)))
        canvas = Image.new('RGB', (label_w, label_h), (0, 0, 0))
        img = Image.fromarray(resized)
        offset = ((label_w - resized.shape[1]) // 2, (label_h - resized.shape[0]) // 2)
        canvas.paste(img, offset)
        self._frame_image = ImageTk.PhotoImage(canvas)
        self.video_label.configure(image=self._frame_image)
        self._job_id = self.after(self._delay_ms, self._update_frame)


if __name__ == "__main__":
    VideoPlayer().mainloop()
