#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Autenticación - Contador de Carros
Copyright (c) 2025 Angel Lluilema
Todos los derechos reservados.
"""

import tkinter as tk
from tkinter import messagebox
import hashlib, json, os

# === CONFIGURACIÓN ===
RUTA_APP = os.path.dirname(os.path.abspath(__file__))
ARCHIVO_USUARIOS = os.path.join(RUTA_APP, "usuarios.json")
SALT = "bit_future_salt_v2"

def hash_pw(pw: str):
    return hashlib.sha256((SALT + pw).encode("utf-8")).hexdigest()

def cargar_usuarios():
    if not os.path.exists(ARCHIVO_USUARIOS):
        with open(ARCHIVO_USUARIOS, "w", encoding="utf-8") as f:
            json.dump({"admin": {"pw": hash_pw("admin123")}}, f, indent=2)
    with open(ARCHIVO_USUARIOS, "r", encoding="utf-8") as f:
        return json.load(f)

def autenticar(user, pw):
    usuarios = cargar_usuarios()
    if user in usuarios and usuarios[user]["pw"] == hash_pw(pw):
        return True
    return False

def registrar(user, pw):
    usuarios = cargar_usuarios()
    if user in usuarios:
        return False
    usuarios[user] = {"pw": hash_pw(pw)}
    with open(ARCHIVO_USUARIOS, "w", encoding="utf-8") as f:
        json.dump(usuarios, f, indent=2)
    return True

# === INTERFAZ PRINCIPAL ===
class FuturisticLogin(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Prometheo Traffic – Acceso")
        self.geometry("420x480")
        self.resizable(False, False)

        # === Fondo gradiente animado ===
        self.canvas = tk.Canvas(self, width=420, height=480, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)

        # Paleta más formal y temática de tránsito: azules profundos con acento cian
        self.colors = [
            (11, 30, 57),   # navy profundo
            (30, 58, 138),  # steel blue
            (0, 194, 255),  # cian vibrante
        ]
        self.phase = 0
        self.update_gradient()

        # === Contenedor ===
        self.frame = tk.Frame(self.canvas, bg="#0a0a0f", highlightbackground="#00e5ff",
                              highlightthickness=2)
        self.frame.place(relx=0.5, rely=0.5, anchor="center", width=320, height=380)

        tk.Label(self.frame, text="🚦 INICIAR SESIÓN", fg="#00e5ff", bg="#0a0a0f",
                 font=("Consolas", 18, "bold")).pack(pady=(20, 15))

        # === Usuario ===
        self.make_label("Usuario")
        self.user = self.make_entry()

        # === Contraseña ===
        self.make_label("Contraseña")
        self.pw = self.make_entry(show="*")

        # === Botón login ===
        self.btn_login = tk.Button(self.frame, text="> ACCEDER <", font=("Consolas", 11, "bold"),
                                   bg="#00e5ff", fg="#0a0a0f", bd=0, relief="flat",
                                   activebackground="#ffc107", activeforeground="#0a0a0f",
                                   width=20, height=2, cursor="hand2", command=self.login)
        self.btn_login.pack(pady=(8, 15))
        self.btn_login.bind("<Enter>", lambda e: self.btn_login.config(bg="#ffc107"))
        self.btn_login.bind("<Leave>", lambda e: self.btn_login.config(bg="#00e5ff"))

        # === Crear cuenta ===
        tk.Label(self.frame, text="¿No tiene una cuenta?", fg="#ffffff",
                 bg="#0a0a0f", font=("Consolas", 9)).pack(pady=(10, 2))
        tk.Button(self.frame, text="Crear cuenta", fg="#00e5ff", bg="#0a0a0f",
                  font=("Consolas", 9, "underline"), bd=0, cursor="hand2",
                  command=self.registrar_usuario).pack()

    # === UI Helpers ===
    def make_label(self, text):
        tk.Label(self.frame, text=text, fg="#eaf9ff", bg="#0a0a0f",
                 font=("Consolas", 10, "bold")).pack(anchor="w", padx=30, pady=(6, 0))

    def make_entry(self, **kwargs):
        e = tk.Entry(self.frame, font=("Consolas", 11), fg="#00e5ff", bg="#1a1a1f",
                     insertbackground="#00e5ff", relief="flat", justify="center", **kwargs)
        e.pack(ipady=6, pady=(3, 10), ipadx=10)
        return e

    # === Animación del gradiente ===
    def update_gradient(self):
        r1, g1, b1 = self.colors[0]
        r2, g2, b2 = self.colors[1]
        steps = 120
        ratio = abs(((self.phase % (2 * steps)) - steps) / steps)
        r = int(r1 * (1 - ratio) + r2 * ratio)
        g = int(g1 * (1 - ratio) + g2 * ratio)
        b = int(b1 * (1 - ratio) + b2 * ratio)
        color = f"#{r:02x}{g:02x}{b:02x}"
        self.canvas.configure(bg=color)
        self.phase += 1
        self.after(60, self.update_gradient)

    # === Login ===
    def login(self):
        user, pw = self.user.get().strip(), self.pw.get()
        if autenticar(user, pw):
            messagebox.showinfo("ACCESO CONCEDIDO", f"Bienvenido {user.upper()}")
            self.destroy()
        else:
            messagebox.showerror("DENEGADO", "Usuario o contraseña incorrectos")

    # === Registro ===
    def registrar_usuario(self):
        user, pw = self.user.get().strip(), self.pw.get()
        if not user or not pw:
            messagebox.showwarning("Atención", "Debe ingresar usuario y contraseña.")
            return
        if registrar(user, pw):
            messagebox.showinfo("Registro exitoso", f"Usuario '{user}' creado correctamente.")
        else:
            messagebox.showerror("Error", "El usuario ya existe.")

# === EJECUTAR ===
if __name__ == "__main__":
    FuturisticLogin().mainloop()
