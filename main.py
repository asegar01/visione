from PIL import Image
import numpy as np
import open3d
import math
import cv2

from vistas import (
    Projection,
    plot_3d_mesh,
    reconstruct_points_3d,
    reconstruct_edges_3d,
    check_projection_connectivity,
    check_model_consistency,
    find_cycles,
    cycles_to_faces
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
        _, yA = line_center(h_centers[i])
        _, yB = line_center(h_centers[i + 1])
        h_spacings.append(abs(yB - yA))

    v_spacings = []
    for i in range(len(v_centers) - 1):
        xA, _ = line_center(v_centers[i])
        xB, _ = line_center(v_centers[i + 1])
        v_spacings.append(abs(xB - xA))

    # Definir la escala según la mediana de las separaciones
    h_spacings_mean = np.median(h_spacings) if h_spacings else 1
    v_spacings_mean = np.median(v_spacings) if v_spacings else 1

    # Definir factor de escala
    transform = {
        'scale_x': 1.0 / v_spacings_mean,
        'scale_y': 1.0 / h_spacings_mean
    }

    return image_out, transform


def get_points_and_edges_from_contours(image):
    """
    Detecta los vértices y aristas de la figura en la imagen a partir de sus contornos.

    :param image: Imagen de entrada.
    :return:
        - Lista de tuplas (x, y) con las coordenadas de los vértices únicos detectados.
        - Lista de aristas, definidas como pares de índices de los vértices.
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

    # Encontrar contornos
    contours, _ = cv2.findContours(image_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
            if np.linalg.norm(np.array(p) - np.array(point)) < 15:
                return index

        index = len(points)
        points.append(point)
        vertex_map[point] = index
        return index

    for c in contours:
        # Filtrar ruido
        if cv2.contourArea(c) < 100:
            continue

        # Calcula la longitud del contorno y define la tolerancia
        epsilon = 0.02 * cv2.arcLength(c, True)

        # Aproximar el contorno a un polígono
        approx = cv2.approxPolyDP(c, epsilon, True)

        # Obtener los índices de los vértices del polígono
        index_path = [get_unique_point(tuple(pt.ravel())) for pt in approx]

        # Crear las aristas a partir del polígono
        for i in range(len(index_path)):
            u = index_path[i]
            v = index_path[(i + 1) % len(index_path)]
            edge = tuple(sorted((u, v)))
            edges.add(edge)

    return np.array(points, dtype=np.float32), np.array(list(edges), dtype=int)


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
    Ajusta las vistas ortogonales para que compartan un origen y escala común.

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

    # Calcular rangos de cada vista
    front_range = front_max - front_min
    left_range = left_max - left_min
    top_range = top_max - top_min

    # Calcular dimensiones máximas del objeto
    max_width = max(front_range[0], top_range[0])  # Anchura (X) -> Vistas frontal y superior
    max_height = max(front_range[1], left_range[0])  # Altura (Y) -> Vistas frontal y lateral
    max_depth = max(top_range[1], left_range[1])  # Profundidad (Z) -> Vistas superior y lateral

    # Normalizar y escalar cada vista
    def align_view(points, view_origin, view_range, target_dimensions):
        # Trasladar puntos al origen
        norm_points = points - view_origin

        # Invertir el eje Y
        norm_points[:, 1] = view_range[1] - norm_points[:, 1]

        # Escalar los puntos a las dimensiones
        scale = np.array(target_dimensions) / view_range
        scaled_points = norm_points * scale

        return scaled_points

    front_norm = align_view(front_points, front_min, front_range, (max_width, max_height))
    left_norm = align_view(left_points, left_min, left_range, (max_height, max_depth))
    top_norm = align_view(top_points, top_min, top_range, (max_width, max_depth))

    return front_norm, left_norm, top_norm


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
    with open(path, 'w') as file:
        for vertex in points:
            file.write(f'v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n')
        for face in faces:
            idx = ' '.join(str(i + 1) for i in face)
            file.write(f'f {idx}\n')


def main():
    # Cargar imágenes de vistas
    try:
        front_view = cv2.imread('examples/square_front.png')
        left_view = cv2.imread('examples/triangle.png')
        top_view = cv2.imread('examples/square_top.png')

        if front_view is None or left_view is None or top_view is None:
            print("Error: No se pudo cargar una o más imágenes. Asegúrate de que las rutas son correctas")
            return
    except Exception as e:
        print(f"Error al leer las imágenes: {e}")
        return

    # Calibrar cuadrícula en cada vista
    # front_calibrate, front_transform = grid_detection_calibration(front_view)
    # left_calibrate, left_transform = grid_detection_calibration(left_view)
    # top_calibrate, top_transform = grid_detection_calibration(top_view)

    # Detectar los vértices y aristas en cada vista
    front_points_2d, front_edges_2d = get_points_and_edges_from_contours(front_view)
    left_points_2d, left_edges_2d = get_points_and_edges_from_contours(left_view)
    top_points_2d, top_edges_2d = get_points_and_edges_from_contours(top_view)

    # Normalizar y escalar las vistas
    front_points_2d, left_points_2d, top_points_2d = normalize_and_scale_views(
        front_points_2d, left_points_2d, top_points_2d
    )

    # Crear objetos de proyección 2D
    elevation = Projection(front_points_2d, front_edges_2d, np.array([0, 0, 1]), 'elevation')
    section = Projection(left_points_2d, left_edges_2d, np.array([1, 0, 0]), 'section')
    plan = Projection(top_points_2d, top_edges_2d, np.array([0, 1, 0]), 'plan')

    # Verificar la conectividad de cada proyección
    if not check_projection_connectivity((plan, elevation, section)):
        print("Inconsistencia en la conectividad de las proyecciones.")
        return

    tolerance = 5.0

    # Reconstruir el modelo 3D a partir de las proyecciones ortogonales
    points_3d = reconstruct_points_3d(plan, elevation, section, tolerance)
    if len(points_3d) < 4:
        print("No se han podido reconstruir suficientes puntos 3D para formar una figura.")
        return

    edges_3d = reconstruct_edges_3d(points_3d, plan, elevation, section, tolerance)
    if len(edges_3d) == 0:
        print("No se han podido reconstruir suficientes aristas 3D para formar una figura.")
        return

    # Encontrar ciclos y reconstruir caras
    plan_cycles = find_cycles(plan)
    elevation_cycles = find_cycles(elevation)
    section_cycles = find_cycles(section)

    all_faces = []
    all_faces += cycles_to_faces(plan, plan_cycles, points_3d, edges_3d, tolerance)
    all_faces += cycles_to_faces(elevation, elevation_cycles, points_3d, edges_3d, tolerance)
    all_faces += cycles_to_faces(section, section_cycles, points_3d, edges_3d, tolerance)

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

    # Visualizar el modelo reconstruido
    plot_3d_mesh(points_3d, faces)

    # Exportar a formato .obj
    # export_obj("output/model.obj", points_3d, faces)


if __name__ == '__main__':
    main()
