from PIL import Image
import numpy as np
import open3d
import math
import cv2


# Profundidad de la imagen
# model = cv2.dnn.readNetFromTorch('midas_model.pt')


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
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=min_line_length, maxLineGap=max_line_gap)

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


def corner_detection(image):
    """
    Detecta las esquinas (vértices) de la figura en la imagen.

    :param image: Imagen de entrada.
    :return:
        - corners: Lista de tuplas (x, y) con las coordenadas de cada esquina detectada.
    """
    # Convertir la imagen a escala de grises
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binarizar la imagem
    # _, image_bin = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Encontrar contornos
    contours, _ = cv2.findContours(image_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    corners = []
    if contours:
        # Extraer el contorno más grande
        c = max(contours, key=cv2.contourArea)

        # Calcula la longitud del contorno y define la tolerancia
        epsilon = 0.01 * cv2.arcLength(c, True)

        # Aproximar el contorno a un polígono
        approx = cv2.approxPolyDP(c, epsilon, True)

        # Añadir vértices a la lista
        for pt in approx:
            x, y = pt[0]
            corners.append((x, y))

    return corners


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


def poisson_reconstruction(points):
    """
    Reconstruye una malla poligonal a partir de una nube de puntos.

    :param points: Array de puntos en 3D.
    :return:
        - mesh: Malla reconstruida.
    """
    # Generar nube de puntos
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)

    # Estimación de normales
    #pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamKNN(knn=30))
    pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Reconstruir la malla
    mesh, densities = open3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)

    # Filtrar la malla
    # densities = np.asarray(densities)
    # density_threshold = np.quantile(densities, 0.1)
    # vertices_to_remove = densities < density_threshold
    # mesh.remove_vertices_by_mask(vertices_to_remove)

    return mesh


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


def main():
    # Cargar imágenes de vistas
    front_view_path = 'examples/grid_square.png'
    left_view_path = 'examples/grid_square.png'
    top_view_path = 'examples/grid_square.png'

    front_view = cv2.imread(front_view_path)
    left_view = cv2.imread(left_view_path)
    top_view = cv2.imread(top_view_path)

    if front_view is None or left_view is None or top_view is None:
        print("Error al cargar las imágenes.")
        return

    # Calibrar cuadrícula en cada vista
    front_calibrate, front_transform = grid_detection_calibration(front_view)
    left_calibrate, left_transform = grid_detection_calibration(left_view)
    top_calibrate, top_transform = grid_detection_calibration(top_view)

    show_window('Hough', front_calibrate)

    # Detectar esquinas del objeto en cada vista
    front_corners = corner_detection(front_view)
    left_corners = corner_detection(left_view)
    top_corners = corner_detection(top_view)

    # Emparejar puntos entre vistas
    # se podrían ordenar y asociar las i-ésimas esquinas de cada vista --> mismo número de esquinas por vista
    n_corners = min(len(front_corners), len(left_corners), len(top_corners))

    # Convertir coordenadas de cuadrícula a coordenadas reales
    front_corners_real = []
    for (px, py) in front_corners:
        front_corners_real.append(pixel_to_real(px, py, front_transform))

    left_corners_real = []
    for (px, py) in left_corners:
        left_corners_real.append(pixel_to_real(px, py, left_transform))

    top_corners_real = []
    for (px, py) in top_corners:
        top_corners_real.append(pixel_to_real(px, py, top_transform))

    # Reconstruir puntos 3D
    points = []
    for i in range(n_corners):
        x_front, y_front = front_corners_real[i]
        z_left, y_left = left_corners_real[i]
        x_top, z_top = top_corners_real[i]

        x = (x_front + x_top) / 2.0
        y = (y_front + y_left) / 2.0
        z = (z_left + z_top) / 2.0

        points.append([x, y, z])

    points = np.array(points, dtype=np.float32)

    # Reconstrucción de la superficie
    mesh = poisson_reconstruction(points)

    # Visualizar el modelo
    open3d.visualization.draw_geometries([mesh])


if __name__ == '__main__':
    main()


# orb = cv2.ORB_create(nfeatures=1000)

# Detectar keypoints
# kp_front, des_front = orb.detectAndCompute(front_view, None)
# kp_left, des_left = orb.detectAndCompute(left_view, None)
# kp_top, des_top = orb.detectAndCompute(top_view, None)

# draw only keypoints location,not size and orientation
# img_kp_front = cv2.drawKeypoints(front_view, kp_front, None, color=(0, 255, 0), flags=0)
# img_kp_left = cv2.drawKeypoints(left_view, kp_left, None, color=(0, 255, 0), flags=0)
# img_kp_top = cv2.drawKeypoints(top_view, kp_top, None, color=(0, 255, 0), flags=0)

# create BFMatcher object
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# bf = cv2.BFMatcher(cv2.NORM_HAMMING) --> match KNN

# Emparejamiento frontal, lateral
# matches_fl = bf.match(des_front, des_left)

# Sort them in the order of their distance.
# matches_fl = sorted(matches_fl, key=lambda x: x.distance)

# Emparejamiento frontal, superior
# matches_fs = bf.match(des_front, des_top)

# Sort them in the order of their distance.
# matches_fs = sorted(matches_fs, key=lambda x: x.distance)

# Apply ratio test
# points = []
# for m, n in zip(matches_fl, matches_fs):
# Aplicar filtro
# if m.distance < 0.75 * n.distance:

# Vista frontal
# x_front, y_front = kp_front[m.queryIdx].pt

# Vista lateral
# z_left, y_left = kp_left[m.trainIdx].pt

# Vista superior
# x_top, z_top = kp_top[m.queryIdx].pt

# x = (x_front + x_top) / 2.0
# y = (y_front + y_left)
# z = (z_left + z_top) / 2.0

# points.append([x, y, z])

# points_arr = np.array(points)

# cv.drawMatchesKnn expects list of lists as matches.
# img3 = cv2.drawMatches(front_view, kp_front, left_view, kp_left, points_arr, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# point_cloud = open3d.geometry.PointCloud()
# point_cloud.points = open3d.utility.Vector3dVector(points_arr)
# open3d.visualization.draw_geometries([point_cloud])
