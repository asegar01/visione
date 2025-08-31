import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.collections as mc
from scipy.spatial import Delaunay, ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from itertools import product
import networkx as nx

from graph import Graph, Vertex


class Projection:
    """Orthogonal projection of a point cloud"""

    def __init__(self, points: np.ndarray, edges: np.ndarray, normal: np.ndarray, name: str):
        self.points = points
        self.edges = edges
        self.normal = normal
        self.name = name
        self.support_edges = edges.copy()

    def __repr__(self):
        return f'Projection(\n\t{repr(self.points)},\n\t{repr(self.edges)},\n\t{repr(self.normal)},\n\t{repr(self.name)}\n)'


def project_cloud_axis(points: np.ndarray, edges: np.ndarray, axis: int, name: str):
    """Project a 3D point cloud into a coordinate plane"""

    mask = tuple(k for k in range(points.shape[1]) if k != axis)

    # Project points
    projection = points[:, mask]

    # Normal to the projection plane
    normal = np.zeros((1, 3))
    normal[0, axis] = 1

    return Projection(projection, edges, normal, name)


def project_cloud(points: np.ndarray, edges: np.ndarray):
    """Convert a 3D point cloud to its orthogonal projections"""

    # Stacked 3D points
    assert len(points.shape) == 2 and points.shape[1] == 3
    # Stacked edge extrema
    assert len(edges.shape) == 2 and edges.shape[1] == 2

    # Orthogonal projections
    plan = project_cloud_axis(points, edges, 1, 'plan')
    elevation = project_cloud_axis(points, edges, 2, 'elevation')
    section = project_cloud_axis(points, edges, 0, 'section')

    return plan, elevation, section


def add_noise(proj: Projection):
    """Add noise to a projection"""

    new_points = np.random.normal(0.0, 0.1, size=proj.points.shape)
    return Projection(new_points, proj.edges, proj.normal, proj.name)


def plot_projections(projs):
    """Plot the projections"""

    fig, (*axs,) = plt.subplots(1, len(projs))

    for proj, ax in zip(projs, axs):
        ax.set_title(proj.name)
        lines = mc.LineCollection([
            (proj.points[edge[0]], proj.points[edge[1]])
            for edge in proj.edges
        ])
        ax.add_collection(lines)
        ax.scatter(proj.points[:, 0], proj.points[:, 1])

    return fig


def plot_3d_model(points, edges):
    """
    Dibujar el modelo 3D utilizando los puntos y aristas proporcionados.

    :param points: Matriz de puntos 3D.
    :param edges: Matriz de aristas (pares de índices).
    :return:
        - fig: Figura con el modelo 3D dibujado.
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Trazar los puntos
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o', s=50)

    # Dibujar cada arista conectando los puntos correspondientes
    for edge in edges:
        p1, p2 = points[edge[0]], points[edge[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'k-')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Modelo 3D Reconstruido')
    plt.show()

    return fig


def plot_3d_mesh(points, faces):
    """
    Dibuja la malla 3D a partir de un conjunto de puntos y las caras resultantes.

    :param points: Matriz de puntos 3D.
    :param faces: Conjunto de caras (índices de puntos que forman cada cara).
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Construir la malla a partir de las caras
    mesh = [points[list(face)] for face in faces]
    poly3d = Poly3DCollection(mesh, facecolors='cyan', alpha=0.8, edgecolors='k')
    ax.add_collection3d(poly3d)

    # Mostrar los vértices
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', s=10, marker='o')

    # Ajustar relación de aspecto
    max_bounds, min_bounds = np.max(points, axis=0), np.min(points, axis=0)
    center = (max_bounds + min_bounds) / 2.0
    max_range = np.max(max_bounds - min_bounds)

    ax.set_xlim(center[0] - max_range / 2, center[0] + max_range / 2)
    ax.set_ylim(center[1] - max_range / 2, center[1] + max_range / 2)
    ax.set_zlim(center[2] - max_range / 2, center[2] + max_range / 2)

    ax.set_box_aspect([1, 1, 1])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Malla 3D Reconstruida')

    # Ajustar el ángulo de visualización
    ax.view_init(elev=30, azim=-45)

    plt.show()


def extract_edges(faces):
    """
    Extrae los bordes (aristas) de las caras de la malla.

    :param faces: Conjunto de caras (índices de puntos que forman cada cara).
    :return:
        - boundary_edges: Lista de aristas únicas que forman parte del borde.
    """
    edge_count = {}

    # Se obtiene la frecuencia de cada arista en cada cara
    for face in faces:
        n = len(face)
        for i in range(n):
            # Se ordenan los índices para evitar duplicados
            edge = tuple(sorted((face[i], face[(i + 1) % n])))
            edge_count[edge] = edge_count.get(edge, 0) + 1

    # Se consideran los bordes que aparecen una única vez
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
    return boundary_edges


def reconstruct_points_3d(plan_proj, elevation_proj, section_proj, tolerance=1e-2):
    """
    Reconstruye los puntos del modelo 3D a partir de las proyecciones ortogonales.

    Para cada punto de la proyección de alzado se buscan candidatos en las otras proyecciones
    que tengan coordenadas compatibles y se reconstruye la coordenada Z promediando los valores
    de planta y perfil.

    :param plan_proj: Proyección en el plano XZ. (planta)
    :param elevation_proj: Proyección en el plano XY. (alzado)
    :param section_proj: Proyección en el plano YZ. (perfil)
    :param tolerance: Tolerancia para la comparación de coordenadas.
    :return: Array de puntos 3D reconstruidos del modelo.
    """

    matches = []  # Lista para almacenar las coincidencias

    # Para cada punto en la proyección de alzado (XY)
    for i, (x_elevation, y_elevation) in enumerate(elevation_proj.points):
        # Buscar candidatos en la proyección de planta que tengan una X similar
        candidates_plan = [
            j for j, (x_plan, z_plan) in enumerate(plan_proj.points)
            if abs(x_plan - x_elevation) < tolerance
        ]

        # Buscar candidatos en la proyección de perfil que tengan una Y similar
        candidates_section = [
            k for k, (y_section, z_section) in enumerate(section_proj.points)
            if abs(y_section - y_elevation) < tolerance
        ]

        if not candidates_plan or not candidates_section:
            print(f"No se encontraron candidatos para el punto alzado {i}: ({x_elevation}, {y_elevation})")
            continue

        # Obtener la coordenada Z de ambas proyecciones
        for j in candidates_plan:
            x_plan, z_plan = plan_proj.points[j]
            for k in candidates_section:
                y_section, z_section = section_proj.points[k]
                if abs(z_plan - z_section) < tolerance:
                    # Reconstruir el punto 3D (x, y, z), donde z es el promedio de los dos valores
                    z = (z_plan + z_section) / 2.0
                    matches.append((x_elevation, y_elevation, z))

    unique_points = {}

    for (x, y, z) in matches:
        key = (round(x / tolerance), round(y / tolerance), round(z / tolerance))

        if key not in unique_points:
            unique_points[key] = (x, y, z)

    points_3d = np.array(list(unique_points.values()), dtype=np.float32)

    return points_3d


def get_angle(a, b):
    """
    Calcula el ángulo en grados entre dos vectores mediante el producto escalar.

    :param a: Primer vector.
    :param b: Segundo vector.
    :return: Ángulo en grados.
    """
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0

    # Coseno del ángulo
    cos = np.dot(a, b) / (na * nb)
    cos = np.clip(cos, -1.0, 1.0)

    # Convertir a ángulo en grados
    angle = np.degrees(np.arccos(cos))
    return angle


def is_collinear(a, b, collinear_tolerance):
    """
    Comprueba si dos vectores son colineales.

    :param a: Primer vector.
    :param b: Segundo vector.
    :param collinear_tolerance: Tolerancia para la comparación en grados.
    :return:
        - True si los vectores son colineales.
        - False en caso contrario.
    """
    angle = get_angle(a, b)
    return (angle <= collinear_tolerance) or (abs(angle - 180.0) <= collinear_tolerance)


def calculate_collinear_edges(proj, collinear_tolerance=3.0):
    """
    Calcula y añade las aristas indirectas que existen entre vértices colineales.

    Esta función encuentra todos los pares de vértices (u, v) conectados por un camino de
    aristas colineales y los añade al conjunto de aristas de soporte de la proyección.

    :param proj: Proyección a procesar.
    :param collinear_tolerance: Tolerancia para la colinealidad en grados.
    """
    graph = build_projection_graph(proj)

    # Pares (u, v) alcanzables mediante mediante un camino colineal
    closure = set()

    for u in range(len(proj.points)):
        for neighbor in graph.adj.get(u, Vertex(u)).neighbors():
            v = neighbor.name
            if v == u:
                continue

            direction = proj.points[v] - proj.points[u]
            n = np.linalg.norm(direction)
            if n < 1e-12:
                continue
            direction = direction / n if n > 1e-12 else direction

            stack = [v]
            visited = {u}

            while stack:
                current = stack.pop()
                if current in visited:
                    continue

                closure.add(tuple(sorted((u, current))))
                visited.add(current)

                for neighbor_next in graph.adj.get(current, Vertex(current)).neighbors():
                    w = neighbor_next.name
                    if w in visited:
                        continue

                    direction_next = proj.points[w] - proj.points[current]
                    if np.linalg.norm(direction_next) < 1e-12:
                        continue

                    if is_collinear(direction, direction_next, collinear_tolerance) and \
                            is_collinear(direction, proj.points[w] - proj.points[u], collinear_tolerance):
                        stack.append(w)

    support = {tuple(sorted(tuple(edge))) for edge in proj.edges}
    support |= closure
    proj.support_edges = np.array(sorted(support), dtype=int)


def reconstruct_edges_3d(points_3d, plan_proj, elevation_proj, section_proj, tolerance=1e-2):
    """
    Reconstruye las aristas del modelo 3D a partir de las proyecciones ortogonales.

    Para cada arista se debe cumplir que su proyección en cada vista corresponda a una arista 2D
    existente o converja en un único punto.

    :param points_3d: Array de puntos 3D.
    :param plan_proj: Proyección en el plano XZ. (planta)
    :param elevation_proj: Proyección en el plano XY. (alzado)
    :param section_proj: Proyección en el plano YZ. (perfil)
    :param tolerance: Tolerancia para la comparación de coordenadas.
    :param angle_tolerance: Tolerancia para la detección de aristas colineales.
    :return: Array de aristas 3D reconstruidos del modelo.
    """
    # Crear conjuntos de aristas visibles e indirectas para cada vista
    plan_visible = {tuple(sorted(edge)) for edge in plan_proj.edges}
    elevation_visible = {tuple(sorted(edge)) for edge in elevation_proj.edges}
    section_visible = {tuple(sorted(edge)) for edge in section_proj.edges}

    plan_support = {tuple(sorted(edge)) for edge in plan_proj.support_edges}
    elevation_support = {tuple(sorted(edge)) for edge in elevation_proj.support_edges}
    section_support = {tuple(sorted(edge)) for edge in section_proj.support_edges}

    # Mapear puntos 3D a índices 2D en cada vista
    plan_index_map = {}
    elevation_index_map = {}
    section_index_map = {}

    for index_3d, point_3d in enumerate(points_3d):
        # Proyección en el plano XZ (planta)
        point_2d_plan = point_3d[[0, 2]]
        for index_2d, point_2d in enumerate(plan_proj.points):
            if np.linalg.norm(point_2d - point_2d_plan) < tolerance:
                plan_index_map[index_3d] = index_2d
                break

        # Proyección en el plano XY (alzado)
        point_2d_elevation = point_3d[[0, 1]]
        for index_2d, point_2d in enumerate(elevation_proj.points):
            if np.linalg.norm(point_2d - point_2d_elevation) < tolerance:
                elevation_index_map[index_3d] = index_2d
                break

        # Proyección en el plano YZ (perfil)
        point_2d_section = point_3d[[1, 2]]
        for index_2d, point_2d in enumerate(section_proj.points):
            if np.linalg.norm(point_2d - point_2d_section) < tolerance:
                section_index_map[index_3d] = index_2d
                break

    edges_3d = set()
    n_points = len(points_3d)

    # Comprobar todas las posibles parejas de vértices para formar aristas
    for u in range(n_points):
        for v in range(u + 1, n_points):
            # Comprobar que ambos puntos se mapearon correctamente en todas las vistas
            if not all(p in plan_index_map for p in (u, v)) or \
                    not all(p in elevation_index_map for p in (u, v)) or \
                    not all(p in section_index_map for p in (u, v)):
                continue

            # Obtener los índices 2D
            i_plan, j_plan = plan_index_map[u], plan_index_map[v]
            i_elev, j_elev = elevation_index_map[u], elevation_index_map[v]
            i_section, j_section = section_index_map[u], section_index_map[v]

            # Comprobar si la proyección es una arista (visible u oculta) o un punto en cada vista
            plan_edge = tuple(sorted((i_plan, j_plan)))
            elevation_edge = tuple(sorted((i_elev, j_elev)))
            section_edge = tuple(sorted((i_section, j_section)))

            # Arista directa (visible u oculta) en cada vista
            plan_is_direct = plan_edge in plan_visible
            elevation_is_direct = elevation_edge in elevation_visible
            section_is_direct = section_edge in section_visible

            # Comprobar si corresponde con un punto
            plan_is_point = (i_plan == j_plan)
            elevation_is_point = (i_elev == j_elev)
            section_is_point = (i_section == j_section)

            # Arista indirecta en cada vista
            plan_is_support = plan_edge in plan_support
            elevation_is_support = elevation_edge in elevation_support
            section_is_support = section_edge in section_support

            # La vista es compatible si es arista directa, punto o arista indirecta
            plan_valid = plan_is_point or plan_is_support
            elevation_valid = elevation_is_point or elevation_is_support
            section_valid = section_is_point or section_is_support

            # Contar las vistas válidas y en las que aparece la arista como visible
            all_valid_views = plan_valid and elevation_valid and section_valid
            edge_is_direct = plan_is_direct or elevation_is_direct or section_is_direct

            # Arista visible presente en todas las vistas
            if all_valid_views:
                edges_3d.add(tuple(sorted((u, v))))

    return np.array(sorted(edges_3d), dtype=int)


def check_projection_connectivity(projections):
    """
    Verifica la conectividad de las proyecciones.

    Para cada proyección se calcula el grado (número de conexiones) de cada punto. Se considera
    inconsistente si se encuentra un punto con únicamente una conexión (grado 1)

    :param projections: Conjunto de proyecciones a evaluar.
    :return:
        - bool: True si no se encuentra ninguna inconsistencia en la conectividad.
                False en caso contrario.
    """

    for proj in projections:
        # Si hay menos de 3 puntos, se omite la comprobación
        if len(proj.points) < 3:
            continue

        degrees = np.zeros(len(proj.points), dtype=int)

        # Contar conexiones para cada punto según las aristas
        for u, v in proj.edges:
            degrees[u] += 1
            degrees[v] += 1

        # Verificar si existe algún punto con una única conexión (grado 1)
        for idx, deg in enumerate(degrees):
            if deg == 1:
                print(f"Inconsistencia en {proj.name}. El punto {proj.points[idx]} tiene una única conexión.")
                return False
    return True


def check_model_consistency(points, edges):
    """
    Comprueba la consistencia global del modelo 3D reconstruido.

    Se calcula el grado de cada vértice y se comprueba que, para modelos de más de 4 vértices, cada
    vértice tenga al menos 3 conexiones.

    :param points: Matriz de puntos 3D.
    :param edges: Matriz de aristas.
    :return:
        - bool: True si el modelo es consistente.
                False en caso contrario.
    """
    if edges is None or len(edges) == 0:
        return True

    if len(points) < 3:
        return False

    # Calcular grado de cada vértice
    degrees = np.zeros(len(points), dtype=int)
    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1

    for i, deg in enumerate(degrees):
        if deg < 3:
            print(f"Inconsistencia global en el vértice {i} con coordenadas {points[i]} [grado {deg}]")
            return False
    return True


def check_projection_consistency(plan_proj, elevation_proj, section_proj, tolerance=1e-2):
    """
    Comprueba que las proyecciones sean consistentes entre sí y con el modelo 3D reconstruido.

    Se realizan las siguientes comprobaciones:
        1. Verificar la conectividad de cada proyección.
        2. Reconstruir el modelo 3D a partir de las proyecciones.
        3. Comprobar la consistencia global del modelo 3D.
        4. Reconstruir las proyecciones a partir del modelo 3D obtenido.
        5. Comparar los puntos de las proyecciones originales con los reconstruidos.

    :param plan_proj: Proyección en el plano XZ. (planta)
    :param elevation_proj: Proyección en el plano XY. (alzado)
    :param section_proj: Proyección en el plano YZ. (perfil)
    :param tolerance: Tolerancia para la comparación de coordenadas.
    :return:
        - bool: True si se cumplen todas las comprobaciones.
                False en caso contrario.
    """

    try:
        # 1. Verificar la conectividad de cada proyección
        if not check_projection_connectivity((plan_proj, elevation_proj, section_proj)):
            return False

        # 2. Reconstruir el modelo 3D a partir de las proyecciones
        points_3d = reconstruct_points_3d(plan_proj, elevation_proj, section_proj, tolerance)
        if len(points_3d) == 0:
            return False

        edges_3d = reconstruct_edges_3d(points_3d, plan_proj, elevation_proj, section_proj, tolerance)

        # 3. Verificar consistencia global del modelo 3D
        if not check_model_consistency(points_3d, edges_3d):
            return False

        # 4. Reconstruir las proyecciones a partir del modelo 3D obtenido
        recon_plan, recon_elevation, recon_section = project_cloud(points_3d, edges_3d)

        # 5. Comparar cada punto en la proyección original con algún punto en la reconstruida
        for proj_orig, proj_recon in [
            (plan_proj, recon_plan),
            (elevation_proj, recon_elevation),
            (section_proj, recon_section)
        ]:
            for point_orig in proj_orig.points:
                match_found = any(
                    abs(point_orig[0] - point_recon[0]) < tolerance and
                    abs(point_orig[1] - point_recon[1]) < tolerance
                    for point_recon in proj_recon.points
                )
                if not match_found:
                    return False
        return True

    except Exception as e:
        print(f"Error al verificar consistencia: {e}")
        return False


def build_projection_graph(proj: Projection) -> Graph:
    """
    Crea un grafo no dirigido a partir de los vértices y aristas de una proyección.

    :param proj: Proyección utilizada para crear el grafo.
    :return: Grafo con los vértices y conexiones de la proyección.
    """
    g = Graph()

    for i in range(len(proj.points)):
        g.add_vertex(i)

    for u, v in proj.edges:
        g.add_edge(u, v)
    return g


def canonical(cycle):
    """
    Rota la lista de vértices de un ciclo para que empiece por el índice mínimo.

    :param cycle: Lista con los índices del ciclo.
    :return: Tupla ordenada del ciclo.
    """
    i = cycle.index(min(cycle))
    return tuple(cycle[i:] + cycle[:i])


def fix_face_orientation(face, points):
    """
    Garantiza que la normal de la cara apunte al exterior del modelo.

    :param face: Tupla con los índices de los vértices que componen la cara.
    :param points: Array de puntos 3D del modelo.
    :return: Cara resultante con la orientación correcta.
    """
    c = points[list(face)].mean(axis=0)

    i, j, k = face[:3]
    normal = np.cross(points[j] - points[i], points[k] - points[i])

    if np.dot(normal, c - points.mean(axis=0)) < 0:
        face = face[::-1]
    return tuple(face)


def find_cycles(proj: Projection):
    """
    Encuentra todos los ciclos sin cuerdas (chordless cycles) del grafo de la proyección,
    que corresponden a las caras mínimas.

    :param proj: Proyección desde la que se extrae el grafo.
    :return: Lista de ciclos encontrados.
    """

    # Construir un grafo no dirigido
    graph = nx.Graph()
    graph.add_nodes_from(range(len(proj.points)))
    graph.add_edges_from([tuple(sorted(edge)) for edge in proj.support_edges])

    # Encontrar los ciclos
    try:
        cycles = nx.chordless_cycles(graph)
        return cycles
    except nx.NetworkXError as e:
        print(f"Error al encontrar los ciclos del grafo en la vista {proj.name}: {e}")
        return []


def find_cycles_old(proj: Projection, max_cycle_len=10):
    """
    Encuentra todos los ciclos del grafo de la proyección hasta una longitud dada.

    :param proj: Proyección desde la que se extrae el grafo.
    :param max_cycle_len: Longitud máxima del ciclo a buscar.
    :return: Lista de ciclos encontrados.
    """

    # Construir el grafo a partir de las aristas de la proyección
    graph = build_projection_graph(proj)
    n_points = len(proj.points)

    cycles = set()

    def dfs_recursive(start, current, path, visited):
        """
        Realiza un recorrido en profundidad de forma recursiva para encontrar todos los ciclos.

        :param start: Vértice de inicio del ciclo.
        :param current: Vértice actual de exploración.
        :param path: Camino actual recorrido.
        :param visited: Vértices visitados en el camino actual.
        """

        # Añadir el nodo actual al camino y marcarlo como visitado
        path.append(current)
        visited[current] = True

        # Recorrer los vecinos
        for neighbor in graph.adj.get(current, Vertex(current)).neighbors():
            w = neighbor.name

            # Si ha llegado al inicio y el camino tiene al menos 3 nodos, hay ciclo
            if w == start and len(path) > 2:
                if len(path) <= max_cycle_len:
                    cycles.add(canonical(path))
                continue

            # Búsqueda recursiva
            if not visited.get(w, False):
                dfs_recursive(start, w, path, visited)

        # Eliminar el nodo del camino al retroceder
        path.pop()
        visited[current] = False

    for v in range(n_points):
        visited_path = {}
        dfs_recursive(start=v, current=v, path=[], visited=visited_path)

    return [list(c) for c in cycles]


def is_face_connected(candidate_face, unique_edges):
    """
    Verifica que la cara esté formada por aristas 3D existentes.

    :param candidate_face: Tupla con los índices de los vértices que componen la cara.
    :param unique_edges: Array de aristas 3D del modelo.
    :return:
        - bool: True si la cara es conexa.
                False en caso contrario.
    """
    for i in range(len(candidate_face)):
        u = candidate_face[i]
        v = candidate_face[(i + 1) % len(candidate_face)]

        if tuple(sorted((u, v))) not in unique_edges:
            return False

    return True


def is_face_coplanar(face, points, tolerance=1e-2):
    """
    Verifica que todos los puntos de la cara se encuentren en un mismo plano.

    :param face: Tupla con los índices de los vértices que componen la cara.
    :param points: Array de puntos 3D del modelo.
    :param tolerance: Tolerancia para la comparación de coordenadas.
    :return:
        - bool: True si la cara es coplanar.
                False en caso contrario.
    """
    if len(face) < 4:
        return True

    p0, p1, p2 = points[face[:3]]
    normal = np.cross(p1 - p0, p2 - p0)
    norm = np.linalg.norm(normal)

    if norm < 1e-3:
        return False
    normal = normal / norm

    for idx in face[3:]:
        distance = abs(np.dot(points[idx] - p0, normal))
        if distance >= tolerance:
            return False

    return True


def is_face_geometry_valid(face, points, tolerance=1e-2):
    """
    Valida la geometría de la cara mediante la suma de ángulos.

    Una cara válida debe tener una suma de giros de 360 grados, mientras que
    una cara que se intersecta a sí misma será cercana a 0.

    :param face: Tupla con los índices de los vértices que componen la cara.
    :param points: Array de puntos 3D.
    :param tolerance: Tolerancia para la suma de ángulos.
    :return:
        - True si la geometría de la cara es válida.
        - False en caso contrario.
    """
    if len(face) < 3:
        return False

    face_points = points[face]

    # Proyectar la cara 3D a un plano 2D
    p0, p1, p2 = face_points[:3]
    normal = np.cross(p1 - p0, p2 - p0)
    if np.linalg.norm(normal) < 1e-9:
        return False

    # Descartar eje perpendicular
    normal_absolute = np.abs(normal)
    if normal_absolute[0] > normal_absolute[1] and normal_absolute[0] > normal_absolute[2]:
        points_2d = face_points[:, [1, 2]]
    elif normal_absolute[1] > normal_absolute[2]:
        points_2d = face_points[:, [0, 2]]
    else:
        points_2d = face_points[:, [0, 1]]

    # Calcular la suma total de los giros en cada vértice
    total_angle = 0.0
    for i in range(len(face)):
        p_prev = points_2d[i - 1]
        p_current = points_2d[i]
        p_next = points_2d[(i + 1) % len(face)]

        # Obtener ángulo de cada vector respecto al origen
        v1 = p_prev - p_current
        v2 = p_next - p_current
        angle = math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0])

        # Normalizar el ángulo a [-180, 180]
        if angle > math.pi:
            angle -= 2 * math.pi
        elif angle < -math.pi:
            angle += 2 * math.pi

        total_angle += angle

    return abs(abs(total_angle) - 2 * math.pi) < tolerance


def cycles_to_faces(proj: Projection, cycles, points_3d, edges_3d, matching_tolerance, geometry_tolerance):
    """
    Convierte los ciclos 2D de una proyección en caras 3D.

    :param proj: Proyección a procesar.
    :param cycles: Ciclos 2D de la proyección.
    :param points_3d: Array de puntos 3D del modelo.
    :param edges_3d: Array de aristas 3D del modelo.
    :param matching_tolerance: Tolerancia de emparejamiento para la comparación de coordenadas.
    :param geometry_tolerance: Tolerancia geométrica para la comparación de coordenadas.
    :return: Lista de caras 3D de la proyección.
    """
    axis = int(np.argmax(proj.normal))
    mask = tuple(i for i in range(3) if i != axis)

    unique_edges = {tuple(sorted(edge)) for edge in edges_3d}

    # Relacionar cada punto 2D con sus posibles puntos 3D
    candidates = [[] for _ in range(len(proj.points))]
    for idx_2d, p_2d in enumerate(proj.points):
        u, v = p_2d
        for idx_3d, p_3d in enumerate(points_3d):
            if (abs(p_3d[mask[0]] - u) < matching_tolerance and
                    abs(p_3d[mask[1]] - v) < matching_tolerance):
                candidates[idx_2d].append(idx_3d)

    # Buscar caras válidas
    faces = []
    unique_faces = set()

    # Backtracking recursivo
    def find_valid_faces_recursive(cycle_2d_indices, current_face_3d, depth):
        # Caso base: cara completa reconstruida
        if depth == len(cycle_2d_indices):
            # Validar arista de cierre
            first_vertex = current_face_3d[0]
            last_vertex = current_face_3d[-1]
            if tuple(sorted((first_vertex, last_vertex))) not in unique_edges:
                return

            # Validar coplanaridad
            if not is_face_coplanar(current_face_3d, points_3d, geometry_tolerance):
                return

            # Validar geometría
            if not is_face_geometry_valid(current_face_3d, points_3d, geometry_tolerance):
                return

            # Añadir la cara si es única
            key = tuple(sorted(current_face_3d))
            if key not in unique_faces:
                oriented_face = fix_face_orientation(current_face_3d, points_3d)
                faces.append(oriented_face)
                unique_faces.add(key)
            return

        # Intentar añadir el siguiente vértice
        current_vertex_2d_idx = cycle_2d_indices[depth]

        for candidate_3d_idx in candidates[current_vertex_2d_idx]:
            # Evitar índices repetidos en la misma cara
            if candidate_3d_idx in current_face_3d:
                continue

            # Si no es el primer vértice, comprobar conectividad con el anterior (poda)
            if depth > 0:
                prev_vertex_3d_idx = current_face_3d[-1]
                edge = tuple(sorted((prev_vertex_3d_idx, candidate_3d_idx)))

                # Comprobar si la arista existe para seguir o no por esa rama
                if edge not in unique_edges:
                    continue

            # Continuar la búsqueda recursiva
            find_valid_faces_recursive(cycle_2d_indices, current_face_3d + [candidate_3d_idx], depth + 1)

    # Iniciar la búsqueda para cada ciclo
    for cycle in cycles:
        if not cycle or len(cycle) < 3:
            continue

        find_valid_faces_recursive(cycle, [], 0)

    return faces


def filter_invalid_vertices(points_3d, edges_3d, min_degree=3, collinear_tolerance=1.0):
    """
    Filtra los vértices que no cumplen con un grado mínimo de conectividad.

    :param points_3d: Array de puntos 3D.
    :param edges_3d: Array de aristas 3D.
    :param min_degree: Grado mínimo que debe tener un vértice para ser conservado.
    :return:
        - filtered_points: Array de puntos filtrado.
        - filtered_edges: Array de aristas actualizado con los nuevos índices de los puntos.
    """
    if len(points_3d) <= min_degree or len(edges_3d) == 0:
        return points_3d, edges_3d

    # Crear grafo a partir de vértices y aristas
    g = Graph()
    for i in range(len(points_3d)):
        g.add_vertex(i)
    for u, v in edges_3d:
        g.add_edge(u, v)

    valid_indices = []
    for i in range(len(points_3d)):
        v = g.adj.get(i)
        if not v:
            continue

        # Obtener los vecinos del vértice
        neighbors = [neighbor.name for neighbor in v.neighbors()]
        if len(neighbors) < min_degree:
            continue

        # Calcular el grado de cada vértice
        degree = 0
        if len(neighbors) > 0:
            # Obtener direcciones de cada arista
            vectors = [points_3d[n] - points_3d[i] for n in neighbors]
            remaining = list(range(len(vectors)))

            while remaining:
                degree += 1
                idx = remaining.pop(0)
                vector = vectors[idx]

                # Conservar únicamente vectores no colineales
                remaining = [idx for idx in remaining if not is_collinear(vector, vectors[idx], collinear_tolerance)]

        if degree >= min_degree:
            valid_indices.append(i)

    if len(valid_indices) == len(points_3d):
        return points_3d, edges_3d

    # Traducir índices antiguos a nuevos
    old_to_new_indices = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_indices)}

    # Filtrar lista de puntos y remapear aristas
    filtered_points = points_3d[valid_indices]
    filtered_edges = []

    for u, v in edges_3d:
        if u in old_to_new_indices and v in old_to_new_indices:
            # Añadir la arista usando los índices nuevos
            new_u = old_to_new_indices[u]
            new_v = old_to_new_indices[v]
            filtered_edges.append((new_u, new_v))

    return filtered_points, np.array(filtered_edges, dtype=int)


def align_view(points, edges, angle_tolerance=3.0, line_tolerance=3.0):
    """
    Corrige los puntos de la vista para garantizar la ortogonalidad de las aristas.

    :param points: Array de puntos 2D de la vista.
    :param edges: Array de aristas que conectan los puntos.
    :param angle_tolerance: Tolerancia en grados para considerar aristas como horizontales o verticales.
    :param line_tolerance: Tolerancia en píxeles para agrupar puntos en la misma línea.
    :return: Lista de puntos alineados.
    """
    if points is None or len(points) == 0 or edges is None or len(edges) == 0:
        return points

    aligned_points = points.copy()

    horizontal = set()
    vertical = set()

    # Identificar aristas horizontales y verticales
    for u, v in edges:
        p1, p2 = points[u], points[v]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]

        angle = abs(math.degrees(math.atan2(dy, dx)))

        # Comprobar si es horizontal
        if min(angle, 180.0 - angle) <= angle_tolerance:
            horizontal.add(u)
            horizontal.add(v)

        # Comprobar si es vertical
        elif abs(angle - 90.0) <= angle_tolerance:
            vertical.add(u)
            vertical.add(v)

    # Agrupar los puntos por coordenada
    def align_axis(vertices, axis):
        if not vertices:
            return

        ordered_vertices = sorted(list(vertices), key=lambda i: aligned_points[i, axis])

        groups = []
        current_group = [ordered_vertices[0]]

        for i in range(1, len(ordered_vertices)):
            current_idx = ordered_vertices[i]
            prev_idx = current_group[-1]

            # Calcular distancia al último punto del grupo actual
            distance = abs(aligned_points[current_idx, axis] - aligned_points[prev_idx, axis])

            if distance <= line_tolerance:
                current_group.append(current_idx)
            else:
                groups.append(current_group)
                current_group = [current_idx]

        groups.append(current_group)

        for group in groups:
            mean = np.mean([aligned_points[j, axis] for j in group])
            for j in group:
                aligned_points[j, axis] = mean

    align_axis(vertical, axis=0)
    align_axis(horizontal, axis=1)

    return aligned_points


def rotate_points(points, angle):
    """
    Rota un conjunto de puntos alrededor del origen un ángulo en grados.

    :param points: Array de puntos 2D.
    :param angle: Ángulo de rotación en grados.
    :return: Array de puntos rotados.
    """
    if points is None or len(points) == 0:
        return points

    radians = math.radians(angle)
    rotation_matrix = np.array([
        [np.cos(radians), -np.sin(radians)],
        [np.sin(radians), np.cos(radians)]
    ], dtype=float)

    return np.dot(points, rotation_matrix.T)


def evaluate_rotation(points, edges):
    """
    Estima el ángulo de rotación principal de una vista y el grado de alineación de sus aristas.

    :param points: Array de puntos 2D.
    :param edges: Array de aristas.
    :return: Tupla con el ángulo de rotación en grados y el grado de alineación (0 a 1).
    """
    if points is None or len(points) == 0 or edges is None or len(edges) == 0:
        return 0.0, 0.0

    angles = []
    weights = []

    # Extraer ángulos y pesos (longitud) de cada arista
    for u, v in edges:
        p1, p2 = points[u], points[v]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]

        length = math.hypot(dx, dy)
        if length < 1e-9:
            continue

        angle = math.degrees(math.atan2(dy, dx))
        angles.append(angle)
        weights.append(length)

    if not angles:
        return 0.0, 0.0

    # Mapear los ángulos al rango [-360, 360]
    doubled_angles = np.radians(np.array(angles, dtype=float) * 2.0)

    # Normalizar los pesos
    weights = np.array(weights, dtype=float)
    weight_total = weights.sum()
    if weight_total <= 0.0:
        return 0.0, 0.0
    normalized_weights = weights / weight_total

    # Calcular el vector promedio ponderado
    mean_cos = float(np.sum(normalized_weights * np.cos(doubled_angles)))
    mean_sin = float(np.sum(normalized_weights * np.sin(doubled_angles)))

    # Calcular el grado de alineación (0 -> aleatoria, 1 -> perfecta)
    alignment_value = math.hypot(mean_cos, mean_sin)

    # Volver al espacio de ángulos original
    mean_doubled_angle = math.atan2(mean_sin, mean_cos)
    angle = math.degrees(mean_doubled_angle / 2.0)

    # Ajustar el ángulo para obtener la rotación más corta hacia el eje más cercano
    angle = (angle + 90.0) % 180.0 - 90.0
    if angle >= 45.0:
        angle -= 90.0
    elif angle < -45.0:
        angle += 90.0

    return angle, alignment_value


def align_view_auto(points, edges, angle_tolerance=5.0, line_tolerance=5.0, rotation_threshold=0.2):
    """
    Detecta y corrige automáticamente la rotación de una vista para alinearla con los ejes.

    :param points: Array de puntos 2D de la vista.
    :param edges: Array de aristas de la vista.
    :param angle_tolerance: Tolerancia en grados para considerar aristas como horizontales o verticales.
    :param line_tolerance: Tolerancia en píxeles para agrupar puntos en la misma línea.
    :param rotation_threshold: Umbral para aplicar la rotación.
    :return: Array de puntos con la alineación corregida.
    """
    if points is None or len(points) == 0 or edges is None or len(edges) == 0:
        return points

    angle, rotation_value = evaluate_rotation(points, edges)
    if rotation_value < rotation_threshold:
        return align_view(points, edges, angle_tolerance=angle_tolerance, line_tolerance=line_tolerance)

    points_rotated = rotate_points(points, -angle)
    points_aligned = align_view(points_rotated, edges, angle_tolerance=angle_tolerance, line_tolerance=line_tolerance)
    points_original = rotate_points(points_aligned, angle)

    return points_original


def main():
    return


if __name__ == '__main__':
    main()
