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

    def __repr__(self):
        return f'Project(\n\t{repr(self.points)},\n\t{repr(self.edges)},\n\t{repr(self.normal)},\n\t{repr(self.name)}\n)'


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
    :return: Array de aristas 3D reconstruidos del modelo.
    """
    plan_unique = {tuple(sorted(edge)) for edge in plan_proj.edges}
    elevation_unique = {tuple(sorted(edge)) for edge in elevation_proj.edges}
    section_unique = {tuple(sorted(edge)) for edge in section_proj.edges}

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

            # Comprobar si la proyección es un borde o un punto en cada vista
            plan_valid = (i_plan == j_plan) or (tuple(sorted((i_plan, j_plan))) in plan_unique)
            elevation_valid = (i_elev == j_elev) or (tuple(sorted((i_elev, j_elev))) in elevation_unique)
            section_valid = (i_section == j_section) or (tuple(sorted((i_section, j_section))) in section_unique)

            if plan_valid and elevation_valid and section_valid:
                edges_3d.add(tuple(sorted((u, v))))

    return np.array(list(edges_3d), dtype=int)


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
        for edge in proj.edges:
            degrees[edge[0]] += 1
            degrees[edge[1]] += 1

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
    graph.add_edges_from(proj.edges)

    # Encontrar los ciclos
    try:
        cycles = list(nx.chordless_cycles(graph))
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

    unique_edges = {tuple(sorted(e)) for e in edges_3d}

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

    for cycle in cycles:
        # Generar todas las listas de candidatos para los puntos del ciclo
        candidates_for_cycle = [candidates[idx_2d] for idx_2d in cycle]

        # Generar todas las combinaciones posibles de caras, utilizando el producto cartesiano
        for candidate_face in product(*candidates_for_cycle):
            # Descartar si tiene vértices repetidos
            if len(set(candidate_face)) != len(candidate_face):
                continue

            # Validar conectividad
            if not is_face_connected(candidate_face, unique_edges):
                continue

            # Validar coplanaridad
            if not is_face_coplanar(list(candidate_face), points_3d, geometry_tolerance):
                continue

            # Añadir la cara a la lista
            key = tuple(sorted(candidate_face))
            if key not in unique_faces:
                oriented_face = fix_face_orientation(list(candidate_face), points_3d)
                faces.append(oriented_face)

                unique_faces.add(key)

    return faces


def filter_invalid_vertices(points_3d, edges_3d, min_degree=3):
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

    # Calcular grado de cada vértice
    degrees = np.zeros(len(points_3d), dtype=int)
    for u, v in edges_3d:
        degrees[u] += 1
        degrees[v] += 1

    # Identificar los índices válidos.
    # Si eliminamos el vértice 2, el antiguo vértice 3 será ahora el 2 -> {0:0, 1:1, 3:2, 4:3}
    valid_indices = sorted([i for i, deg in enumerate(degrees) if deg >= min_degree])

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


def align_view(points, edges, tolerance=5.0):
    """
    Corrige los puntos de la vista para garantizar la ortogonalidad de las aristas.

    :param points: Array de puntos 2D de la vista.
    :param edges: Array de aristas que conectan los puntos.
    :param tolerance: Tolerancia en grados para considerar aristas como horizontales o verticales.
    :return: Lista de puntos alineados.
    """
    if len(points) == 0:
        return points

    candidates_x = [[] for _ in range(len(points))]
    candidates_y = [[] for _ in range(len(points))]

    # Identificar aristas horizontales y verticales
    for u, v in edges:
        p1, p2 = points[u], points[v]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]

        angle = math.degrees(math.atan2(dy, dx))

        # Comprobar si es horizontal
        if abs(angle) < tolerance or abs(abs(angle) - 180) < tolerance:
            average_y = (p1[1] + p2[1]) / 2.0
            candidates_y[u].append(average_y)
            candidates_y[v].append(average_y)

        # Comprobar si es vertical
        elif abs(abs(angle) - 90) < tolerance:
            average_x = (p1[0] + p2[0]) / 2.0
            candidates_x[u].append(average_x)
            candidates_x[v].append(average_x)

    # Corregir los puntos según lo calculado
    aligned_points = points.copy()
    for i in range(len(points)):
        if candidates_x[i]:
            aligned_points[i, 0] = np.mean(candidates_x[i])
        if candidates_y[i]:
            aligned_points[i, 1] = np.mean(candidates_y[i])

    return aligned_points


SEGMENT1 = (
    np.array([(0, 0, 0), (1, 1, 1)]),
    np.array([(0, 1)])
)
SEGMENT2 = (
    np.array([(0, 0, 0), (1, 1, 0)]),
    np.array([(0, 1)])
)
SEGMENT3 = (
    np.array([(0, 0, 0), (1, 0, 0)]),
    np.array([(0, 1)])
)

TRIANGLE1 = (
    np.array([(-1, 0, -1), (0, 1, 1), (1, 0, 0)]),
    np.array([(0, 1), (1, 2), (2, 0)])
)
TRIANGLE2 = (
    np.array([(-1, 0, 0), (0, 1, 1), (1, 0, 0)]),
    np.array([(0, 1), (1, 2), (2, 0)])
)
TRIANGLE3 = (
    np.array([(-1, 0, 0), (0, 1, 0), (1, 0, 0)]),
    np.array([(0, 1), (1, 2), (2, 0)])
)
TRIANGLE4 = (
    np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0)]),
    np.array([(0, 1), (1, 2), (2, 0)])
)

CUBE = (
    np.array([
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),  # Cara inferior
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)  # Cara superior
    ]),
    np.array([
        (0, 1), (1, 2), (2, 3), (3, 0),  # Aristas de la cara inferior
        (4, 5), (5, 6), (6, 7), (7, 4),  # Aristas de la cara superior
        (0, 4), (1, 5), (2, 6), (3, 7)  # Aristas laterales
    ])
)

ROTATED_CUBE = (
    np.array([
        (0, 0, 0), (np.sqrt(2) / 2, 0, -np.sqrt(2) / 2), (np.sqrt(2) / 2, 1, -np.sqrt(2) / 2), (0, 1, 0),
        (np.sqrt(2) / 2, 0, np.sqrt(2) / 2), (np.sqrt(2), 0, 0), (np.sqrt(2), 1, 0), (np.sqrt(2) / 2, 1, np.sqrt(2) / 2)
    ]),
    np.array([
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ])
)

PYRAMID = (
    np.array([
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),  # Base
        (0.5, 0.5, 1)  # Vértice superior (apex)
    ]),
    np.array([
        (0, 1), (1, 2), (2, 3), (3, 0),  # Aristas de la base
        (0, 4), (1, 4), (2, 4), (3, 4)  # Conexiones al vértice superior
    ])
)

IRREGULAR = (
    np.array([
        (0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0),  # Cara inferior
        (0.5, 0.5, 2), (1.5, 0.5, 2)  # Puntos superiores
    ]),
    np.array([
        (0, 1), (1, 2), (2, 3), (3, 0),  # Aristas de la cara inferior
        (0, 4), (1, 5), (2, 5), (3, 4),  # Aristas laterales
        (4, 5)  # Arista superior
    ])
)

INCONSISTENT = (
    np.array([
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
        (0.5, 0.5, 0.5)  # Punto inconsistente
    ]),
    np.array([
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
        (0, 8), (4, 8)  # Aristas conectadas al punto inconsistente
    ])
)

TRIANGULAR_PRISM = (
    np.array([
        [0, 0, 0],  # Vértices del triángulo inferior
        [1, 0, 0],
        [0.5, 1, 0],
        [0, 0, 1],  # Vértices del triángulo superior
        [1, 0, 1],
        [0.5, 1, 1]
    ]),
    np.array([
        (0, 1), (1, 2), (2, 0),  # Aristas del triángulo inferior
        (3, 4), (4, 5), (5, 3),  # Aristas del triángulo superior
        (0, 3), (1, 4), (2, 5)  # Aristas laterales que conectan ambos triángulos
    ])
)

HEXAGONAL_PRISM = (
    np.array([
        [math.cos(0), math.sin(0), 0],
        [math.cos(math.pi / 3), math.sin(math.pi / 3), 0],
        [math.cos(2 * math.pi / 3), math.sin(2 * math.pi / 3), 0],
        [math.cos(math.pi), math.sin(math.pi), 0],
        [math.cos(4 * math.pi / 3), math.sin(4 * math.pi / 3), 0],
        [math.cos(5 * math.pi / 3), math.sin(5 * math.pi / 3), 0],
        [math.cos(0), math.sin(0), 1],
        [math.cos(math.pi / 3), math.sin(math.pi / 3), 1],
        [math.cos(2 * math.pi / 3), math.sin(2 * math.pi / 3), 1],
        [math.cos(math.pi), math.sin(math.pi), 1],
        [math.cos(4 * math.pi / 3), math.sin(4 * math.pi / 3), 1],
        [math.cos(5 * math.pi / 3), math.sin(5 * math.pi / 3), 1],
    ]),
    np.array([
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),  # Aristas de la base inferior
        (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 6),  # Aristas de la base superior
        (0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11)  # Conexiones verticales
    ])
)

OCTAHEDRON = (
    np.array([
        [1, 0, 0],  # A (índice 0)
        [-1, 0, 0],  # B (índice 1)
        [0, 1, 0],  # C (índice 2)
        [0, -1, 0],  # D (índice 3)
        [0, 0, 1],  # E (índice 4, polo superior)
        [0, 0, -1]  # F (índice 5, polo inferior)
    ]),
    np.array([
        (4, 0), (4, 1), (4, 2), (4, 3),  # Conexiones del polo superior
        (5, 0), (5, 1), (5, 2), (5, 3),  # Conexiones del polo inferior
        (0, 2), (2, 1), (1, 3), (3, 0)  # Conexiones entre vértices equatoriales
    ])
)

DISCONNECTED_LINES = (
    np.array([
        [0, 0, 0],  # Primer segmento
        [1, 1, 1],
        [2, 2, 2],  # Segundo segmento
        [3, 3, 3]
    ]),
    np.array([
        (0, 1),  # Arista del primer segmento
        (2, 3)  # Arista del segundo segmento
    ])
)

HOUSE = (
    np.array([
        [0, 0, 0],  # Base inferior del cubo
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],  # Parte superior del cubo
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
        [0.5, 0.5, 1.5]  # Vértice del techo (pirámide sobre el cubo)
    ]),
    np.array([
        (0, 1), (1, 2), (2, 3), (3, 0),  # Aristas de la base inferior
        (4, 5), (5, 6), (6, 7), (7, 4),  # Aristas de la parte superior (cubo)
        (0, 4), (1, 5), (2, 6), (3, 7),  # Aristas verticales del cubo
        (8, 4), (8, 5), (8, 6), (8, 7)  # Aristas del techo
    ])
)


def main():
    shapes = [
        ("SEGMENT1", SEGMENT1),
        ("SEGMENT2", SEGMENT2),
        ("SEGMENT3", SEGMENT3),
        ("TRIANGLE1", TRIANGLE1),
        ("TRIANGLE2", TRIANGLE2),
        ("TRIANGLE3", TRIANGLE3),
        ("TRIANGLE4", TRIANGLE4),
        ("CUBE", CUBE),
        ("ROTATED_CUBE", ROTATED_CUBE),
        ("PYRAMID", PYRAMID),
        ("IRREGULAR", IRREGULAR),
        ("INCONSISTENT", INCONSISTENT),
        ("TRIANGULAR_PRISM", TRIANGULAR_PRISM),
        ("HEXAGONAL_PRISM", HEXAGONAL_PRISM),
        ("OCTAHEDRON", OCTAHEDRON),
        ("DISCONNECTED_LINES", DISCONNECTED_LINES),
        ("HOUSE", HOUSE)
    ]

    for name, shape in shapes:
        print(f"\n=== Procesando forma: {name} ===")

        # Obtener vértices y aristas de la forma
        points_3d, edges_3d = shape

        # Generar las proyecciones ortogonales
        plan, elevation, section = project_cloud(points_3d, edges_3d)

        # Verificar la conectividad de cada proyección
        if not check_projection_connectivity((plan, elevation, section)):
            print("Inconsistencia en la conectividad de las proyecciones.")
            continue

        plan_cycles = find_cycles(plan)
        elevation_cycles = find_cycles(elevation)
        section_cycles = find_cycles(section)

        all_faces = []
        all_faces += cycles_to_faces(plan, plan_cycles, points_3d, edges_3d)
        all_faces += cycles_to_faces(elevation, elevation_cycles, points_3d, edges_3d)
        all_faces += cycles_to_faces(section, section_cycles, points_3d, edges_3d)

        # Comprobar la consistencia global del modelo 3D reconstruido
        if not check_model_consistency(points_3d, edges_3d):
            print("Inconsistencia global en el modelo 3D reconstruido.")
            continue

        unique_faces = set()
        faces = []
        for face in all_faces:
            key = tuple(sorted(face))
            if key not in unique_faces:
                faces.append(face)
                unique_faces.add(key)

        # Visualizar el modelo 3D
        print(f"Puntos 3D: {len(points_3d)}, Aristas 3D: {len(edges_3d)}, Caras: {len(faces)}")
        plot_3d_mesh(points_3d, faces)


if __name__ == '__main__':
    main()
