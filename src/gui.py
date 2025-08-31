import os
import sys
import threading
import json
import cv2
import numpy as np
import matplotlib
import tkinter as tk

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from pathlib import Path
from tkinter import filedialog as fd, ttk, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD

from main import (
    get_points_and_edges_from_contours,
    normalize_and_scale_views,
    export_obj,
    get_view_scale
)

from vistas import (
    Projection,
    reconstruct_points_3d,
    reconstruct_edges_3d,
    check_projection_connectivity,
    check_model_consistency,
    find_cycles,
    cycles_to_faces,
    calculate_collinear_edges,
    align_view_auto,
    filter_invalid_vertices
)

APP_TITLE = "visione"
PREFERENCES_FILE = Path.home() / ".visione_prefs.json"
PADDING = 10


class Application(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.title(APP_TITLE)
        self.geometry("1280x720")

        # Estilo de la aplicación
        style = ttk.Style()
        style.theme_use("xpnative")
        matplotlib.rcParams.update({
            "axes.edgecolor": "#B5B5B5",
            "axes.linewidth": 0.8,
            "xtick.color": "#9C9C9C",
            "ytick.color": "#9C9C9C",
            "grid.color": "#CCCCCC",
            "grid.alpha": 0.3
        })

        # Cargar preferencias
        self.last_dir = str(Path.home())
        self.load_preferences()

        self.front_path = tk.StringVar()
        self.right_path = tk.StringVar()
        self.top_path = tk.StringVar()

        for var in (self.front_path, self.right_path, self.top_path):
            var.trace_add("write", lambda *args: self.on_path_change())

        self.noise_threshold = tk.IntVar(value=100)
        self.vertex_distance = tk.IntVar(value=15)
        self.matching_tolerance = tk.DoubleVar(value=2.0)
        self.geometry_tolerance = tk.DoubleVar(value=1.5)
        self.kernel_shape = tk.IntVar(value=0)
        self.approx_ratio = tk.DoubleVar(value=1.0)

        self.front_points_2d, self.front_edges_2d = np.array([]), np.array([])
        self.left_points_2d, self.left_edges_2d = np.array([]), np.array([])
        self.top_points_2d, self.top_edges_2d = np.array([]), np.array([])

        self.points_3d = np.array([])
        self.edges_3d = np.array([])
        self.faces = []

        # Estado de la aplicación
        self.is_model_ready = False  # Indica si hay algún modelo listo para exportar
        self.is_working = False  # Indica si se está reconstruyendo

        # Hilo de reconstrucción
        self.cancel_event = threading.Event()
        self.reconstruct_thread = None

        self.build_application()

    def build_application(self):
        # Distribución de la aplicación
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        header = ttk.Frame(self, padding=PADDING)
        header.grid(row=0, column=0, sticky="ew")
        for i in range(5):
            header.columnconfigure(i, weight=1)

        # Cargar imágenes
        ttk.Button(header, text="Cargar alzado",
                   command=lambda: self.add_file(self.front_path)).grid(row=0, column=0, padx=PADDING, sticky="ew")
        self.front_entry = ttk.Entry(header, textvariable=self.front_path)
        self.front_entry.grid(row=0, column=1, padx=10, sticky="ew")

        ttk.Button(header, text="Cargar planta",
                   command=lambda: self.add_file(self.top_path)).grid(row=1, column=0, padx=PADDING, sticky="ew")
        self.top_entry = ttk.Entry(header, textvariable=self.top_path)
        self.top_entry.grid(row=1, column=1, padx=10, sticky="ew")

        ttk.Button(header, text="Cargar perfil",
                   command=lambda: self.add_file(self.right_path)).grid(row=2, column=0, padx=PADDING, sticky="ew")
        self.left_entry = ttk.Entry(header, textvariable=self.right_path)
        self.left_entry.grid(row=2, column=1, padx=10, sticky="ew")

        # Permitir drag-and-drop de una vista
        for widget, target in [
            (self.front_entry, self.front_path),
            (self.top_entry, self.top_path),
            (self.left_entry, self.right_path)
        ]:
            widget.drop_target_register(DND_FILES)
            widget.dnd_bind('<<Drop>>', lambda e, t=target: self.on_drop_file(e, target=t))

        # Permitir drag-and-drop de todos los archivos
        self.drop_target_register(DND_FILES)
        self.dnd_bind('<<Drop>>', self.on_drop_anywhere)

        # Introducir parámetros
        params = ttk.LabelFrame(header, text="Parámetros de reconstrucción", padding=PADDING / 2)
        params.grid(row=0, column=2, rowspan=3, columnspan=2, padx=PADDING * 4, pady=PADDING, sticky="ew")
        for i in range(2):
            params.columnconfigure(i, weight=1)

        ttk.Label(params, text="Área mínima de contorno (px²)").grid(row=0, column=0, sticky="w")
        self.noise_scale = ttk.Scale(params, from_=0, to=500, orient="horizontal", variable=self.noise_threshold,
                                     command=lambda e: self.update_labels())
        self.noise_scale.grid(row=0, column=1, sticky="ew")
        self.noise_scale.config(state="disabled")
        self.noise_threshold_label = ttk.Label(params, text=str(self.noise_threshold.get()))
        self.noise_threshold_label.grid(row=0, column=2, sticky="w")

        ttk.Label(params, text="Distancia para fusionar vértices (px)").grid(row=1, column=0, sticky="w")
        self.vertex_scale = ttk.Scale(params, from_=1, to=150, orient="horizontal", variable=self.vertex_distance,
                                      command=lambda e: self.update_labels())
        self.vertex_scale.grid(row=1, column=1, sticky="ew")
        self.vertex_scale.config(state="disabled")
        self.vertex_distance_label = ttk.Label(params, text=str(self.vertex_distance.get()))
        self.vertex_distance_label.grid(row=1, column=2, sticky="w")

        ttk.Label(params, text="Tolerancia de emparejamiento (%)").grid(row=2, column=0, sticky="w")
        self.matching_scale = ttk.Scale(params, from_=0.5, to=20.0, orient="horizontal",
                                        variable=self.matching_tolerance,
                                        command=lambda e: self.update_labels())
        self.matching_scale.grid(row=2, column=1, sticky="ew")
        self.matching_scale.config(state="disabled")
        self.matching_tolerance_label = ttk.Label(params, text=f"{self.matching_tolerance.get():.1f}%")
        self.matching_tolerance_label.grid(row=2, column=2, sticky="w")

        ttk.Label(params, text="Tolerancia geométrica (%)").grid(row=3, column=0, sticky="w")
        self.geometry_scale = ttk.Scale(params, from_=0.5, to=15.0, orient="horizontal",
                                        variable=self.geometry_tolerance,
                                        command=lambda e: self.update_labels())
        self.geometry_scale.grid(row=3, column=1, sticky="ew")
        self.geometry_scale.config(state="disabled")
        self.geometry_tolerance_label = ttk.Label(params, text=f"{self.geometry_tolerance.get():.1f}%")
        self.geometry_tolerance_label.grid(row=3, column=2, sticky="w")

        ttk.Label(params, text="Aproximación de contorno (%)").grid(row=4, column=0, sticky="w")
        self.approx_scale = ttk.Scale(params, from_=0.0, to=5.0, orient="horizontal",
                                      variable=self.approx_ratio,
                                      command=lambda e: self.update_labels())
        self.approx_scale.grid(row=4, column=1, sticky="ew")
        self.approx_scale.config(state="disabled")
        self.approx_ratio_label = ttk.Label(params, text=f"{self.approx_ratio.get():.2f}%")
        self.approx_ratio_label.grid(row=4, column=2, sticky="w")

        ttk.Label(params, text="Cierre de trazos discontinuos (px)").grid(row=5, column=0, sticky="w")
        self.kernel_scale = ttk.Scale(params, from_=0, to=35, orient="horizontal",
                                      variable=self.kernel_shape,
                                      command=lambda e: self.update_labels())
        self.kernel_scale.grid(row=5, column=1, sticky="ew")
        self.kernel_scale.config(state="disabled")
        self.kernel_shape_label = ttk.Label(params, text="Off")
        self.kernel_shape_label.grid(row=5, column=2, sticky="w")

        # Botones de acción
        actions = ttk.Frame(header)
        actions.grid(row=0, column=4, rowspan=3, sticky="nesw")
        self.reconstruct_button = ttk.Button(
            actions, text="Reconstruir modelo", command=self.reconstruct_async, state="disabled")
        self.reconstruct_button.pack(fill="x", padx=PADDING * 2, pady=PADDING + PADDING / 2)

        self.export_button = ttk.Button(actions, text="Exportar", command=self.export_obj, state="disabled")
        self.export_button.pack(fill="x", padx=PADDING * 2, pady=PADDING + PADDING / 2)

        self.cancel_button = ttk.Button(actions, text="Cancelar", command=self.cancel_reconstruction, state="disabled")
        self.cancel_button.pack(fill="x", padx=PADDING * 2, pady=PADDING + PADDING / 2)

        # Barra de progreso
        self.progress_bar = ttk.Progressbar(actions, mode="indeterminate", orient=tk.HORIZONTAL)

        # Pestañas de la aplicación
        body = ttk.Notebook(self)
        body.grid(row=1, column=0, padx=PADDING, pady=PADDING, sticky="nesw")

        # Pestaña 2D
        self.tab_2d = ttk.Frame(body)
        body.add(self.tab_2d, text="Vistas 2D")

        # Vistas 2D
        self.fig_2d, self.axs_2d = plt.subplots(1, 3, figsize=(9, 3))
        self.axs_2d[0].set_title("Alzado")
        self.axs_2d[1].set_title("Perfil")
        self.axs_2d[2].set_title("Planta")
        for ax in self.axs_2d:
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True)
        self.canvas_2d = FigureCanvasTkAgg(self.fig_2d, self.tab_2d)
        self.canvas_2d.get_tk_widget().pack(fill="both", expand=True, padx=PADDING, pady=PADDING)

        # Pestaña 3D
        self.tab_3d = ttk.Frame(body)
        body.add(self.tab_3d, text="Malla 3D")

        # Vista 3D
        self.fig_3d = plt.figure(figsize=(10, 8))
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.ax_3d.set_box_aspect([1, 1, 1])
        self.ax_3d.set_title("Malla 3D Reconstruida")
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, self.tab_3d)
        self.canvas_3d.get_tk_widget().pack(fill="both", expand=True, padx=PADDING, pady=PADDING)

        # Pestaña de depuración
        self.tab_log = ttk.Frame(body, padding=PADDING)
        body.add(self.tab_log, text="Consola")
        self.text_log = tk.Text(self.tab_log, height=10)
        self.text_log.pack(fill="both", expand=True)
        self.text_log.tag_configure("ok", foreground="#2E2828")
        self.text_log.tag_configure("error", foreground="#ef4444")

    def update_labels(self):
        self.noise_threshold_label.config(text=str(self.noise_threshold.get()))
        self.vertex_distance_label.config(text=str(self.vertex_distance.get()))
        self.matching_tolerance_label.config(text=f"{self.matching_tolerance.get():.1f}%")
        self.geometry_tolerance_label.config(text=f"{self.geometry_tolerance.get():.1f}%")
        self.approx_ratio_label.config(text=f"{self.approx_ratio.get():.2f}%")

        k = int(self.kernel_shape.get())
        self.kernel_shape_label.config(text=("Off" if k == 0 else f"{k} px"))

    def update_params(self):
        params = [
            self.noise_scale, self.vertex_scale, self.matching_scale,
            self.geometry_scale, self.approx_scale, self.kernel_scale
        ]

        if self.paths_ready():
            for p in params:
                p.config(state="normal")
        else:
            for p in params:
                p.config(state="disabled")

    def update_dynamic_sliders(self, area, diagonal):
        self.noise_scale.config(to=int(area * 0.002))  # Hasta el 0.2% del área
        self.vertex_scale.config(to=int(diagonal * 0.1))  # Hasta el 10% de la diagonal
        self.update_labels()

    @staticmethod
    def get_image_params(path):
        image = cv2.imread(path)
        if image is None:
            return
        diagonal = np.linalg.norm(image.shape[:2])
        area = image.shape[0] * image.shape[1]
        return area, diagonal

    def apply_sliders(self):
        paths = [self.front_path.get(), self.right_path.get(), self.top_path.get()]
        params = []

        for path in paths:
            if os.path.isfile(path):
                area, diagonal = self.get_image_params(path)
                if area is not None and diagonal is not None:
                    params.append((area, diagonal))

        if not params:
            return

        max_area = max(area for area, _ in params)
        max_diagonal = max(diagonal for _, diagonal in params)

        self.update_dynamic_sliders(max_area, max_diagonal)
        self.update_labels()

    def add_file(self, file):
        # Formatos de imagen aceptados
        filetypes = (
            ("Imágenes", ("*.png", "*.jpg", "*.jpeg")),
            ("Todos", "*.*")
        )

        # Abrir explorador de archivos
        path = fd.askopenfilename(
            title="Selecciona una imagen",
            initialdir=self.last_dir,
            filetypes=filetypes
        )

        if path:
            file.set(path)

            # Actualizar preferencias
            self.last_dir = os.path.dirname(path) or self.last_dir
            self.save_preferences()

            # Habilitar parámetros
            self.update_params()
            if self.paths_ready():
                self.apply_sliders()

    def split_paths(self, data):
        paths = self.tk.splitlist(data)
        return [os.path.normpath(p) for p in paths]

    def auto_assign(self, paths, overwrite=False):
        mapping = {
            "front": self.front_path, "elevation": self.front_path, "alzado": self.front_path,
            "top": self.top_path, "plan": self.top_path, "planta": self.top_path,
            "right": self.right_path, "section": self.right_path, "perfil": self.right_path
        }

        assigned = set()

        for p in paths:
            name = os.path.splitext(os.path.basename(p))[0].lower()
            for key, var in mapping.items():
                if key in name and (overwrite or var.get() == ""):
                    var.set(p)
                    assigned.add(p)
                    break

    def on_drop_file(self, event, target=None):
        paths = self.split_paths(event.data)
        if not paths:
            return

        if target is not None and len(paths) == 1:
            target.set(paths[0])
        else:
            self.auto_assign(paths, overwrite=True)

        self.on_path_change()

    def on_drop_anywhere(self, event):
        paths = self.split_paths(event.data)
        if not paths:
            return

        self.auto_assign(paths, overwrite=True)
        self.on_path_change()

    def reconstruct_async(self):
        # Establecer estado de la aplicación
        self.cancel_event.clear()
        self.is_working = True
        self.update_buttons()

        # Actualizar barra de progreso
        self.progress_bar.pack(fill="x", padx=PADDING * 2, pady=PADDING)
        self.progress_bar.start(10)

        # Evita congelar la UI durante la reconstrucción
        self.reconstruct_thread = threading.Thread(target=self.reconstruct, daemon=True)
        self.reconstruct_thread.start()

    def finish_reconstruction(self, success):
        self.is_working = False
        self.is_model_ready = bool(success)
        self.update_buttons()

        # Actualizar barra de progreso
        self.progress_bar.stop()
        self.progress_bar.pack_forget()

    def check_finish_reconstruction(self):
        # Comprobar cancelación de reconstrucción por el usuario
        if self.cancel_event.is_set():
            self.log("Reconstrucción cancelada.")
            self.finish_reconstruction(False)
            return

    def reconstruct(self):
        # Validar rutas
        if not (os.path.isfile(self.front_path.get())
                and os.path.isfile(self.right_path.get())
                and os.path.isfile(self.top_path.get())):
            self.log("Debes cargar las imágenes de todas las vistas.", error=True)
            self.after(0, lambda: messagebox.showwarning(APP_TITLE, "No se han cargado las imágenes de las vistas."))
            return

        self.log("Iniciando reconstrucción...")

        # Cargar imágenes de vistas
        front_view = cv2.imread(self.front_path.get())
        left_view = cv2.imread(self.right_path.get())
        top_view = cv2.imread(self.top_path.get())
        if front_view is None or left_view is None or top_view is None:
            self.log("Error al leer una o más imágenes. Comprueba el formato.", error=True)
            self.after(0, lambda: messagebox.showerror(APP_TITLE, "No se pudo leer alguna imagen."))
            self.finish_reconstruction(False)
            return

        success = False

        try:
            k = int(self.kernel_shape.get())
            use_edges = k > 0
            kernel_size = k if k > 0 else 0

            approx_ratio = self.approx_ratio.get() / 100.0

            # Detectar los vértices y aristas en cada vista
            front_points_2d, front_edges_2d = get_points_and_edges_from_contours(
                front_view,
                noise_threshold=int(self.noise_threshold.get()),
                vertex_distance=int(self.vertex_distance.get()),
                hidden_edges=use_edges,
                kernel_shape=kernel_size,
                approx_ratio=approx_ratio
            )
            left_points_2d, left_edges_2d = get_points_and_edges_from_contours(
                left_view,
                noise_threshold=int(self.noise_threshold.get()),
                vertex_distance=int(self.vertex_distance.get()),
                hidden_edges=use_edges,
                kernel_shape=kernel_size,
                approx_ratio=approx_ratio
            )
            top_points_2d, top_edges_2d = get_points_and_edges_from_contours(
                top_view,
                noise_threshold=int(self.noise_threshold.get()),
                vertex_distance=int(self.vertex_distance.get()),
                hidden_edges=use_edges,
                kernel_shape=kernel_size,
                approx_ratio=approx_ratio
            )

            # Comprobar cancelación de reconstrucción por el usuario
            self.check_finish_reconstruction()
            if self.cancel_event.is_set():
                return

            # Corregir ortogonalidad de las vistas
            front_points_2d = align_view_auto(front_points_2d, front_edges_2d)
            left_points_2d = align_view_auto(left_points_2d, left_edges_2d)
            top_points_2d = align_view_auto(top_points_2d, top_edges_2d)

            # Invertir el eje vertical del modelo
            if len(front_points_2d) > 0:
                front_points_2d[:, 1] = front_view.shape[0] - front_points_2d[:, 1]

            if len(left_points_2d) > 0:
                left_points_2d[:, 1] = left_view.shape[0] - left_points_2d[:, 1]

            if len(top_points_2d) > 0:
                top_points_2d[:, 1] = top_view.shape[0] - top_points_2d[:, 1]

            # Normalizar y escalar las vistas
            front_points_2d, left_points_2d, top_points_2d = normalize_and_scale_views(
                front_points_2d, left_points_2d, top_points_2d
            )

            self.front_points_2d, self.front_edges_2d = front_points_2d, front_edges_2d
            self.left_points_2d, self.left_edges_2d = left_points_2d, left_edges_2d
            self.top_points_2d, self.top_edges_2d = top_points_2d, top_edges_2d

            # Mostrar vistas 2D
            self.draw_views()

            # Intercambiar coordenadas del perfil
            if len(left_points_2d) > 0:
                left_points_2d = left_points_2d[:, [1, 0]]

            # Comprobar cancelación de reconstrucción por el usuario
            self.check_finish_reconstruction()
            if self.cancel_event.is_set():
                return

            # Crear objetos de proyección 2D
            elevation = Projection(front_points_2d, front_edges_2d, np.array([0, 0, 1]), 'elevation')
            section = Projection(left_points_2d, left_edges_2d, np.array([1, 0, 0]), 'section')
            plan = Projection(top_points_2d, top_edges_2d, np.array([0, 1, 0]), 'plan')

            # Construir soporte colineal en cada vista
            calculate_collinear_edges(plan)
            calculate_collinear_edges(elevation)
            calculate_collinear_edges(section)

            # Verificar la conectividad de cada proyección
            if not check_projection_connectivity((plan, elevation, section)):
                self.log("Conectividad inconsistente. Los puntos deben tener al menos dos conexiones.", error=True)
                self.after(0, lambda: messagebox.showwarning(APP_TITLE,
                                                             "Inconsistencia en la conectividad de las proyecciones."))
                return

            # Calcular tolerancias en función de la diagonal del alzado
            scale = max(get_view_scale(front_points_2d), get_view_scale(left_points_2d), get_view_scale(top_points_2d))

            matching_tolerance = scale * (self.matching_tolerance.get() / 100.0)
            geometry_tolerance = scale * (self.geometry_tolerance.get() / 100.0)
            self.log(f"Tolerancia de emparejamiento: {matching_tolerance:.2f} "
                     f"| Tolerancia geométrica: {geometry_tolerance:.2f}")

            # Reconstruir el modelo 3D a partir de las proyecciones ortogonales
            points_3d = reconstruct_points_3d(plan, elevation, section, tolerance=matching_tolerance)
            if len(points_3d) < 4:
                self.log("No se han podido reconstruir suficientes puntos 3D para formar una figura.", error=True)
                self.after(0, lambda: messagebox.showwarning(APP_TITLE, "No hay suficientes puntos 3D."))
                return

            # Comprobar cancelación de reconstrucción por el usuario
            self.check_finish_reconstruction()
            if self.cancel_event.is_set():
                return

            edges_3d = reconstruct_edges_3d(points_3d, plan, elevation, section, tolerance=matching_tolerance)
            if len(edges_3d) == 0:
                self.log("No se han podido reconstruir suficientes aristas 3D para formar una figura.", error=True)
                self.after(0, lambda: messagebox.showwarning(APP_TITLE, "No se han reconstruido aristas 3D."))
                return

            # Filtrar vértices
            points_3d, edges_3d = filter_invalid_vertices(points_3d, edges_3d)

            # Comprobar cancelación de reconstrucción por el usuario
            self.check_finish_reconstruction()
            if self.cancel_event.is_set():
                return

            # Encontrar ciclos
            plan_cycles = find_cycles(plan)
            elevation_cycles = find_cycles(elevation)
            section_cycles = find_cycles(section)

            # Comprobar cancelación de reconstrucción por el usuario
            self.check_finish_reconstruction()
            if self.cancel_event.is_set():
                return

            # Reconstruir caras
            all_faces = []
            all_faces.extend(
                cycles_to_faces(plan, plan_cycles, points_3d, edges_3d, matching_tolerance, geometry_tolerance))
            all_faces.extend(
                cycles_to_faces(elevation, elevation_cycles, points_3d, edges_3d, matching_tolerance,
                                geometry_tolerance))
            all_faces.extend(
                cycles_to_faces(section, section_cycles, points_3d, edges_3d, matching_tolerance, geometry_tolerance))

            unique_faces = set()
            faces = []
            for face in all_faces:
                key = tuple(sorted(face))
                if key not in unique_faces:
                    faces.append(face)
                    unique_faces.add(key)

            # Comprobar la consistencia global del modelo 3D reconstruido
            if not check_model_consistency(points_3d, edges_3d):
                self.log("Inconsistencia global en el modelo 3D reconstruido. Comprueba las proyecciones.", error=True)
                self.after(0, lambda: messagebox.showwarning(APP_TITLE, "El modelo es inconsistente."))
                return

            self.points_3d, self.edges_3d, self.faces = points_3d, edges_3d, faces

            # Visualizar el modelo reconstruido
            self.draw_3d_mesh()

            self.log(f"Listo. Vértices: {len(points_3d)} | Aristas: {len(edges_3d)} | Caras: {len(faces)}")
            success = True
        except Exception as e:
            self.log(f"Error inesperado: {e}", error=True)
            self.after(0, lambda: messagebox.showerror(APP_TITLE, f"Error inesperado: \n {e}"))
        finally:
            self.after(0, lambda: self.finish_reconstruction(success))

    def cancel_reconstruction(self):
        if self.is_working:
            self.log("Cancelando reconstrucción...")
            self.cancel_event.set()

    def draw_views(self):
        axes = self.axs_2d
        data = [
            ("Alzado", self.front_points_2d, self.front_edges_2d),
            ("Planta", self.top_points_2d, self.top_edges_2d),
            ("Perfil", self.left_points_2d, self.left_edges_2d)
        ]

        for ax, (title, points, edges) in zip(axes, data):
            ax.clear()
            ax.set_title(title, fontweight="semibold")
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True)

            if len(points) > 0:
                ax.scatter(points[:, 0], points[:, 1], c='#0b1220')
                if len(edges) > 0:
                    lines = [(points[u], points[v]) for u, v in edges]
                    ax.add_collection(LineCollection(lines, colors='#60a5fa'))

        self.canvas_2d.draw_idle()

    def draw_3d_mesh(self):
        self.ax_3d.clear()
        self.ax_3d.set_title("Malla 3D Reconstruida", fontweight="semibold")

        if len(self.points_3d) > 0 and len(self.faces) > 0:
            # Intercambiar ejes para visualización
            points = self.points_3d[:, [0, 2, 1]]

            # Construir la malla a partir de las caras
            mesh = [points[list(face)] for face in self.faces]
            poly3d = Poly3DCollection(mesh, facecolors='#60a5fa', alpha=0.5, edgecolors='#0b1220')
            self.ax_3d.add_collection3d(poly3d)
            self.ax_3d.auto_scale_xyz(points[:, 0], points[:, 1], points[:, 2])

            # Ajustar relación de aspecto
            min_bounds, max_bounds = np.min(points, axis=0), np.max(points, axis=0)
            center = (max_bounds + min_bounds) / 2.0
            max_range = float(np.max(max_bounds - min_bounds))

            self.ax_3d.set_xlim(center[0] - max_range / 2, center[0] + max_range / 2)
            self.ax_3d.set_ylim(center[1] - max_range / 2, center[1] + max_range / 2)
            self.ax_3d.set_zlim(center[2] - max_range / 2, center[2] + max_range / 2)
            self.ax_3d.set_box_aspect([1, 1, 1])

            # Ajustar el ángulo de visualización
            self.ax_3d.view_init(elev=30, azim=-45)

        self.canvas_3d.draw_idle()

    def export_obj(self):
        if len(self.points_3d) == 0 or len(self.faces) == 0:
            self.log("No hay ninguna malla para exportar. Debes generar una primero.", error=True)
            self.after(0, lambda: messagebox.showinfo(APP_TITLE, "No hay malla para exportar."))
            return

        self.log("Exportando modelo...")

        path = fd.asksaveasfilename(
            defaultextension=".obj",
            filetypes=[("OBJ", "*.obj")],
            initialfile="modelo.obj",
            title="Guardar como"
        )

        if path:
            try:
                export_obj(path, self.points_3d, self.faces)
                self.log(f"Modelo exportado: {path}")
                self.after(0, lambda: messagebox.showinfo(APP_TITLE, "Exportación completada."))
            except Exception as e:
                self.log(f"Error al exportar: {e}", error=True)
                self.after(0, lambda: messagebox.showerror(APP_TITLE, f"Error al exportar:\n{e}"))

    def on_path_change(self):
        self.is_model_ready = False
        self.update_buttons()
        self.update_params()

        # Calibrar sliders cuando estén todas las vistas
        if self.paths_ready():
            self.apply_sliders()

        # Limpiar el canvas
        self.ax_3d.clear()
        self.ax_3d.set_title("Malla 3D Reconstruida")
        self.canvas_3d.draw_idle()

    def paths_ready(self):
        return (os.path.isfile(self.front_path.get())
                and os.path.isfile(self.right_path.get())
                and os.path.isfile(self.top_path.get()))

    def update_buttons(self):
        if self.is_working:
            self.reconstruct_button.config(state="disabled")
            self.export_button.config(state="disabled")
            self.cancel_button.config(state="normal")
        else:
            self.reconstruct_button.config(state="normal" if self.paths_ready() else "disabled")
            self.export_button.config(state="normal" if self.is_model_ready else "disabled")
            self.cancel_button.config(state="disabled")

    def log(self, message, error=False):
        tag = "error" if error else "ok"
        if not self.text_log.tag_cget(tag, "foreground"):
            self.text_log.tag_configure("ok", foreground="#2E2828")
            self.text_log.tag_configure("error", foreground="#ef4444")
        self.text_log.insert(tk.END, ("[ERROR] " if error else "") + message + "\n", tag)
        self.text_log.see(tk.END)

    def load_preferences(self):
        try:
            data = json.loads(PREFERENCES_FILE.read_text(encoding="utf-8"))
            self.last_dir = data.get("last_dir", self.last_dir)
        except Exception:
            pass

    def save_preferences(self):
        try:
            PREFERENCES_FILE.write_text(json.dumps({"last_dir": self.last_dir}), encoding="utf-8")
        except Exception:
            pass

    def on_close(self):
        if self.is_working and self.reconstruct_thread:
            self.cancel_event.set()
            self.reconstruct_thread.join()

        # Cerrar la aplicación
        sys.exit(0)


if __name__ == "__main__":
    Application().mainloop()
