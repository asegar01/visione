from PIL import Image
import numpy as np
import open3d
import math
import cv2

from skimage import morphology
from skimage.util import img_as_ubyte

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from vistas import (
    Projection,
    plot_3d_mesh,
    reconstruct_points_3d,
    reconstruct_edges_3d,
    check_projection_connectivity,
    check_model_consistency,
    find_cycles,
    cycles_to_faces,
    calculate_collinear_edges,
    align_view_auto
)


def grid_detection_calibration(image, min_line_length=100, max_line_gap=10):
    """
    Detecta las líneas de la cuadrícula de la imagen y calcula un factor de escala.

    :param image: Imagen de entrada.
    :param min_line_length: Longitud mínima para considerar una línea.
    :param max_line_gap: Máxima separación entre segmentos para unir a una misma línea.
    :return:
        - image_out: Imagen con las líneas detectadas.
        - transform: Diccionario para convertir píxeles a coordenadas reales.
    """
    # Convertir la imagen a escala de grises
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detectar bordes
    edges = cv2.Canny(image_gray, 100, 200)

    # Detectar líneas
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=50,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    # Filtrar líneas horizontales y verticales
    h_lines = []
    v_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1
            angle = math.degrees(math.atan2(dy, dx))
            angle_threshold = 10
            if abs(angle) < angle_threshold:
                h_lines.append((x1, y1, x2, y2))
            elif abs(angle - 90) < angle_threshold or abs(angle + 90) < angle_threshold:
                v_lines.append((x1, y1, x2, y2))

    # Realizar copia de la imagen para visualizar las líneas
    image_out = image.copy()
    for (x1, y1, x2, y2) in h_lines:
        cv2.line(image_out, (x1, y1), (x2, y2), (255, 0, 0), 3)
    for (x1, y1, x2, y2) in v_lines:
        cv2.line(image_out, (x1, y1), (x2, y2), (255, 0, 0), 3)

    # Calcular el centro de una línea
    def line_center(line):
        x1, y1, x2, y2 = line
        return (x1 + x2) / 2, (y1 + y2) / 2

    # Ordenar líneas según su posición
    h_centers = sorted(h_lines, key=lambda ln: line_center(ln)[1])
    v_centers = sorted(v_lines, key=lambda ln: line_center(ln)[0])

    # Calcular separación entre líneas adyacentes
    h_spacings = []
    for i in range(len(h_centers) - 1):
        _, y_a = line_center(h_centers[i])
        _, y_b = line_center(h_centers[i + 1])
        h_spacings.append(abs(y_b - y_a))

    v_spacings = []
    for i in range(len(v_centers) - 1):
        x_a, _ = line_center(v_centers[i])
        x_b, _ = line_center(v_centers[i + 1])
        v_spacings.append(abs(x_b - x_a))

    # Definir la escala según la mediana de las separaciones
    h_spacings_mean = np.median(h_spacings) if h_spacings else 1
    v_spacings_mean = np.median(v_spacings) if v_spacings else 1

    # Definir factor de escala
    transform = {
        'scale_x': 1.0 / v_spacings_mean,
        'scale_y': 1.0 / h_spacings_mean
    }

    return image_out, transform


def crop_image(image):
    """
    Recorta la imagen para ajustarla a la figura principal.

    :param image: Imagen a recortar que contiene la figura.
    :return:
        - image_cropped: Imagen recortada.
        - x: Coordenada x del origen del recorte en la imagen original.
        - y: Coordenada y del origen del recorte en la imagen original.
        -
    """
    # Convertir la imagen a escala de grises
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarizar la imagen
    _, image_bin = cv2.threshold(
        image_gray,
        127,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Encontrar los contornos exteriores para identificar la figura principal
    contours, _ = cv2.findContours(image_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Obtener la caja delimitadora (bounding box) de la figura
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)

    # Recortar la imagen original
    image_cropped = image[y: y + h, x: x + w]

    return image_cropped, x, y


def get_points_and_edges_from_contours(image, noise_threshold=100, vertex_distance=15,
                                       hidden_edges=False, kernel_shape=0, approx_ratio=0.01):
    """
    Detecta los vértices y aristas de la figura en la imagen a partir de sus contornos.

    :param image: Imagen de entrada.
    :param noise_threshold: Umbral de área para filtrar pequeños contornos considerados como ruido.
    :param vertex_distance: Distancia máxima para fusionar dos puntos en un único vértice.
    :return:
        - Lista de tuplas (x, y) con las coordenadas de los vértices únicos detectados.
        - Lista de aristas, definidas como pares de índices de los vértices.
    """
    # Recortar la imagen
    image_cropped, offset_x, offset_y = crop_image(image)

    # Convertir la imagen a escala de grises
    image_cropped_gray = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY)

    # Binarizar la imagen
    _, image_cropped_bin = cv2.threshold(
        image_cropped_gray,
        127,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    if hidden_edges:
        if kernel_shape % 2 == 0:
            kernel_shape += 1

        kernel_x = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_shape, 1))
        kernel_y = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_shape))
        image_cropped_bin = cv2.morphologyEx(image_cropped_bin, cv2.MORPH_CLOSE, kernel_x)
        image_cropped_bin = cv2.morphologyEx(image_cropped_bin, cv2.MORPH_CLOSE, kernel_y)

    # Adelgazar la imagen
    out_thin = morphology.thin(image_cropped_bin)
    image_thin = img_as_ubyte(out_thin)

    # Encontrar contornos
    contours, _ = cv2.findContours(image_thin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    points = []
    edges = set()
    vertex_map = {}

    def get_unique_point(point):
        """
        Comprueba si el punto ya existe para tener vértices únicos.

        :param point: Tupla (x, y) con las coordenadas del punto a comprobar.
        :return: Índice del vértice único.
        """
        for p, index in vertex_map.items():
            if np.linalg.norm(np.array(p) - np.array(point)) < vertex_distance:
                return index

        index = len(points)
        points.append(point)
        vertex_map[point] = index
        return index

    for c in contours:
        # Filtrar ruido
        if cv2.contourArea(c) < noise_threshold:
            continue

        # Calcula la longitud del contorno y define la tolerancia
        epsilon = float(approx_ratio) * cv2.arcLength(c, False)

        # Aproximar el contorno a un polígono
        approx = cv2.approxPolyDP(c, epsilon, False)

        # Obtener los índices de los vértices del polígono
        index_path = [get_unique_point(tuple(pt.ravel())) for pt in approx]

        # Crear las aristas a partir del polígono
        for i in range(len(index_path)):
            u = index_path[i]
            v = index_path[(i + 1) % len(index_path)]
            if u != v:
                edges.add(tuple(sorted((u, v))))

    # Aplicar desplazamiento del recorte a los puntos
    points = np.array(points, dtype=np.float32)
    if points.size > 0:
        points[:, 0] += offset_x
        points[:, 1] += offset_y

    return points, np.array(list(edges), dtype=int)


def pixel_to_real(px, py, transform):
    """
    Convierte las coordenadas de un pixel (px, py) a coordenadas reales, usando el factor de escala.

    :param px: Coordenada x en píxeles.
    :param py: Coordenada y en píxeles.
    :param transform: Diccionario que contiene las escalas.
    :return:
        - real_x: Coordenada x en unidades reales.
        - real_y: Coordenada y en unidades reales.
    """
    sx = transform['scale_x']
    sy = transform['scale_y']
    real_x = px * sx
    real_y = py * sy

    return real_x, real_y


def normalize_and_scale_views(front_points, left_points, top_points):
    """
    Ajusta las vistas ortogonales para que compartan un origen común.

    :param front_points: Array de puntos 2D de la vista frontal (alzado).
    :param left_points: Array de puntos 2D de la vista lateral (perfil).
    :param top_points: Array de puntos 2D de la vista superior (planta).
    :return: Tupla con los puntos normalizados y escalados de las vistas.
    """
    if len(front_points) == 0 or len(left_points) == 0 or len(top_points) == 0:
        return front_points, left_points, top_points

    # Calcular límites de cada vista
    front_min, front_max = np.min(front_points, axis=0), np.max(front_points, axis=0)
    left_min, left_max = np.min(left_points, axis=0), np.max(left_points, axis=0)
    top_min, top_max = np.min(top_points, axis=0), np.max(top_points, axis=0)

    # Trasladar las vistas al origen
    front_norm = front_points - front_min
    left_norm = left_points - left_min
    top_norm = top_points - top_min

    # Calcular dimensiones del objeto
    width = front_max[0] - front_min[0]  # Anchura (X) -> Vistas frontal y superior
    height = front_max[1] - front_min[1]  # Altura (Y) -> Vistas frontal y lateral
    depth = top_max[1] - top_min[1]  # Profundidad (Z) -> Vistas superior y lateral

    # Calcular rangos de cada vista
    front_range = front_max - front_min
    left_range = left_max - left_min
    top_range = top_max - top_min

    # Escalar alzado (XY)
    front_norm[:, 0] *= (width / front_range[0])
    front_norm[:, 1] *= (height / front_range[1])

    # Escalar perfil (ZY)
    left_norm[:, 0] *= (depth / left_range[0])
    left_norm[:, 1] *= (height / left_range[1])

    # Escalar planta (XZ)
    top_norm[:, 0] *= (width / top_range[0])
    top_norm[:, 1] *= (depth / top_range[1])

    return front_norm, left_norm, top_norm


def get_view_scale(points):
    """
    Calcula una escala de referencia para la vista basada en la diagonal de su bounding box.

    :param points: Array de puntos 2D.
    :return: Longitud de la diagonal.
    """
    if len(points) == 0:
        return 1.0

    point_min, point_max = np.min(points, axis=0), np.max(points, axis=0)
    diagonal = np.linalg.norm(point_max - point_min)

    return max(diagonal, 1.0)


def show_views(name, front_points, front_edges, left_points, left_edges, top_points, top_edges,
               front_hidden=None, left_hidden=None, top_hidden=None):
    """
    Muestra una nueva ventana con los puntos y aristas detectados en cada vista.

    :param name: Nombre de la ventana.
    :param front_points: Array de puntos 2D de la vista frontal (alzado).
    :param front_edges: Array de aristas 2D de la vista frontal (alzado).
    :param left_points: Array de puntos 2D de la vista lateral (perfil).
    :param left_edges: Array de aristas 2D de la vista lateral (perfil).
    :param top_points: Array de puntos 2D de la vista superior (planta).
    :param top_edges: Array de aristas 2D de la vista superior (planta).
    :param front_hidden:
    :param left_hidden:
    :param top_hidden:
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
    fig.suptitle(name, fontsize=15)

    # Vista frontal (alzado)
    ax1.set_title("Alzado")
    if len(front_points) > 0:
        ax1.scatter(front_points[:, 0], front_points[:, 1], c='r')
        front_lines = [(front_points[u], front_points[v]) for u, v in front_edges]
        ax1.add_collection(LineCollection(front_lines, colors='b'))
        if front_hidden is not None and len(front_hidden) > 0:
            front_hidden_lines = [(front_points[u], front_points[v]) for u, v in front_hidden]
            ax1.add_collection(LineCollection(front_hidden_lines, colors='gray', linestyles='dashed'))
    ax1.set_aspect('equal', adjustable='box')
    ax1.grid(True)

    # Vista lateral (perfil)
    ax2.set_title("Perfil")
    if len(left_points) > 0:
        ax2.scatter(left_points[:, 0], left_points[:, 1], c='r')
        left_lines = [(left_points[u], left_points[v]) for u, v in left_edges]
        ax2.add_collection(LineCollection(left_lines, colors='b'))
        if left_hidden is not None and len(left_hidden) > 0:
            left_hidden_lines = [(left_points[u], left_points[v]) for u, v in left_hidden]
            ax2.add_collection(LineCollection(left_hidden_lines, colors='gray', linestyles='dashed'))
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True)

    # Vista superior (planta)
    ax3.set_title("Planta")
    if len(top_points) > 0:
        ax3.scatter(top_points[:, 0], top_points[:, 1], c='r')
        top_lines = [(top_points[u], top_points[v]) for u, v in top_edges]
        ax3.add_collection(LineCollection(top_lines, colors='b'))
        if top_hidden is not None and len(top_hidden) > 0:
            top_hidden_lines = [(top_points[u], top_points[v]) for u, v in top_hidden]
            ax3.add_collection(LineCollection(top_hidden_lines, colors='gray', linestyles='dashed'))
    ax3.set_aspect('equal', adjustable='box')
    ax3.grid(True)

    plt.show()


def show_window(name, image):
    """
    Muestra una nueva ventana dimensionada.

    :param name: Nombre de la ventana.
    :param image: Imagen a mostrar.
    """
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 1920, 1080)
    cv2.imshow(name, image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def export_obj(path, points, faces):
    """
    Exporta una malla 3D al formato OBJ.

    :param path: Ruta de destino del archivo.
    :param points: Array de vértices 3D.
    :param faces: Array de caras, definidas como una tupla con los índices de los vértices que la componen.
    """
    # Calcular factor de escala
    max_size = np.max(np.abs(points))
    scale = 1.0 / max_size if max_size != 0 else 1.0

    with open(path, 'w') as file:
        for vertex in points:
            scaled_vertex = vertex * scale
            file.write(f'v {scaled_vertex[0]:.6f} {scaled_vertex[1]:.6f} {scaled_vertex[2]:.6f}\n')
        for face in faces:
            idx = ' '.join(str(i + 1) for i in face)
            file.write(f'f {idx}\n')


def main():
    # Cargar imágenes de vistas
    try:
        front_view = cv2.imread('../examples/complex_4_front.png')
        right_view = cv2.imread('../examples/complex_4_right.png')
        top_view = cv2.imread('../examples/complex_4_top.png')

        if front_view is None or right_view is None or top_view is None:
            print("Error: No se pudo cargar una o más imágenes. Asegúrate de que las rutas son correctas")
            return
    except Exception as e:
        print(f"Error al leer las imágenes: {e}")
        return

    # Calibrar cuadrícula en cada vista
    # front_calibrate, front_transform = grid_detection_calibration(front_view)
    # left_calibrate, left_transform = grid_detection_calibration(right_view)
    # top_calibrate, top_transform = grid_detection_calibration(top_view)

    # Detectar los vértices y aristas en cada vista
    front_points_2d, front_edges_2d = get_points_and_edges_from_contours(front_view)
    right_points_2d, right_edges_2d = get_points_and_edges_from_contours(right_view)
    top_points_2d, top_edges_2d = get_points_and_edges_from_contours(top_view)

    # Corregir ortogonalidad de las vistas
    front_points_2d = align_view_auto(front_points_2d, front_edges_2d)
    right_points_2d = align_view_auto(right_points_2d, right_edges_2d)
    top_points_2d = align_view_auto(top_points_2d, top_edges_2d)

    # Invertir el eje vertical del modelo
    if len(front_points_2d) > 0:
        front_points_2d[:, 1] = front_view.shape[0] - front_points_2d[:, 1]

    if len(right_points_2d) > 0:
        right_points_2d[:, 1] = right_view.shape[0] - right_points_2d[:, 1]

    if len(top_points_2d) > 0:
        top_points_2d[:, 1] = top_view.shape[0] - top_points_2d[:, 1]

    # Normalizar y escalar las vistas
    front_points_2d, right_points_2d, top_points_2d = normalize_and_scale_views(
        front_points_2d, right_points_2d, top_points_2d
    )

    show_views(
        "Vistas normalizadas",
        front_points_2d, front_edges_2d,
        right_points_2d, right_edges_2d,
        top_points_2d, top_edges_2d
    )

    # Intercambiar coordenadas del perfil
    if len(right_points_2d) > 0:
        right_points_2d = right_points_2d[:, [1, 0]]

    # Crear objetos de proyección 2D
    plan = Projection(top_points_2d, top_edges_2d, np.array([0, 1, 0]), 'plan')
    elevation = Projection(front_points_2d, front_edges_2d, np.array([0, 0, 1]), 'elevation')
    section = Projection(right_points_2d, right_edges_2d, np.array([1, 0, 0]), 'section')

    # Construir soporte colineal en cada vista
    calculate_collinear_edges(plan)
    calculate_collinear_edges(elevation)
    calculate_collinear_edges(section)

    # Verificar la conectividad de cada proyección
    if not check_projection_connectivity((plan, elevation, section)):
        print("Inconsistencia en la conectividad de las proyecciones.")
        return

    scale = max(get_view_scale(front_points_2d), get_view_scale(right_points_2d), get_view_scale(top_points_2d))

    matching_tolerance = scale * 0.01
    geometry_tolerance = scale * 0.01

    print(f"Tolerancia de emparejamiento: {matching_tolerance:.2f}")
    print(f"Tolerancia geométrica: {geometry_tolerance:.2f}")

    # Reconstruir el modelo 3D a partir de las proyecciones ortogonales
    points_3d = reconstruct_points_3d(plan, elevation, section, tolerance=matching_tolerance)
    if len(points_3d) < 4:
        print("No se han podido reconstruir suficientes puntos 3D para formar una figura.")
        return

    edges_3d = reconstruct_edges_3d(points_3d, plan, elevation, section, tolerance=matching_tolerance)
    if len(edges_3d) == 0:
        print("No se han podido reconstruir suficientes aristas 3D para formar una figura.")
        return

    # Encontrar ciclos
    plan_cycles = find_cycles(plan)
    elevation_cycles = find_cycles(elevation)
    section_cycles = find_cycles(section)

    # Reconstruir caras
    all_faces = []
    all_faces.extend(
        cycles_to_faces(plan, plan_cycles, points_3d, edges_3d, matching_tolerance, geometry_tolerance))
    all_faces.extend(
        cycles_to_faces(elevation, elevation_cycles, points_3d, edges_3d, matching_tolerance, geometry_tolerance))
    all_faces.extend(
        cycles_to_faces(section, section_cycles, points_3d, edges_3d, matching_tolerance, geometry_tolerance))

    # Comprobar la consistencia global del modelo 3D reconstruido
    if not check_model_consistency(points_3d, edges_3d):
        print("Inconsistencia global en el modelo 3D reconstruido.")
        return

    unique_faces = set()
    faces = []
    for face in all_faces:
        key = tuple(sorted(face))
        if key not in unique_faces:
            faces.append(face)
            unique_faces.add(key)

    if not faces:
        print("No se pudo reconstruir ninguna cara.")
        return

    # Intercambiar ejes para visualización
    points_3d_mat = points_3d[:, [0, 2, 1]]

    # Visualizar el modelo reconstruido
    plot_3d_mesh(points_3d_mat, faces)

    # Exportar a formato .obj
    export_obj("../output/model.obj", points_3d, faces)


if __name__ == '__main__':
    main()
