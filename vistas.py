#
# Proyecta una nube de puntos 3D en sus proyecciones
#

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import math
from scipy.spatial import Delaunay, ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Projection:
    """Orthogonal projection of a point cloud"""

    def __init__(self, points: np.ndarray, edges: np.ndarray, normal: np.ndarray, name: str, index_map=None):
        self.points = points
        self.edges = edges
        self.normal = normal
        self.name = name
        self.index_map = index_map

    def __repr__(self):
        return f'Project(\n\t{repr(self.points)},\n\t{repr(self.edges)},\n\t{repr(self.normal)},\n\t{repr(self.name)}\n)'


def project_cloud_axis(points: np.ndarray, edges: np.ndarray, axis: int, name: str):
    """Project a 3D point cloud into a coordinate plane"""

    mask = tuple(k for k in range(points.shape[1]) if k != axis)

    # Project points
    projection = points[:, mask]

    # Remove duplicates and sort
    # projection, index_map = np.unique(projection, axis=0, return_inverse=True)
    index_map = None

    # Remap indices and remove invisible ones
    # remapped_edges = np.unique(np.sort(index_map[edges], axis=1), axis=0)
    # visible_edges = remapped_edges[remapped_edges[:, 0] != remapped_edges[:, 1]]
    visible_edges = edges

    # Normal to the projection plane
    normal = np.zeros((1, 3))
    normal[0, axis] = 1

    return Projection(projection, visible_edges, normal, name, index_map=index_map)


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
    poly3d = Poly3DCollection(mesh, facecolors='blue', alpha=0.3, edgecolors='none')
    ax.add_collection3d(poly3d)

    # Extraer y dibujar los bordes de la malla
    boundary_edges = extract_edges(faces)
    for edge in boundary_edges:
        p1 = points[edge[0]]
        p2 = points[edge[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='k', linewidth=2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Modelo 3D Delaunay')

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


def reconstruct_3d_from_projections(plan_proj, elevation_proj, section_proj, orig_edges, epsilon=1e-6):
    """
    Reconstruye el modelo 3D a partir de las proyecciones ortogonales.

    Para cada punto de la proyección de alzado se buscan candidatos en las otras proyecciones
    que tengan coordenadas compatibles y se reconstruye la coordenada Z promediando los valores
    de planta y perfil.

    :param plan_proj: Proyección en el plano XZ. (planta)
    :param elevation_proj: Proyección en el plano XY. (alzado)
    :param section_proj: Proyección en el plano YZ. (perfil)
    :param orig_edges: Aristas originales de la nube de puntos 3D.
    :param epsilon: Tolerancia para la comparación de coordenadas.
    :return: Tupla de puntos 3D reconstruidos y aristas correspondientes.
    """

    points_3d = []  # Lista para almacenar los puntos 3D reconstruidos
    mapping = {}  # Mapea el índice de la proyección de alzado al índice en points_3d

    # Conjuntos para evitar reutilizar el mismo punto candidato en otras proyecciones
    used_plan = set()
    used_section = set()

    # Para cada punto en la proyección de alzado (XY)
    for i, (x_elevation, y_elevation) in enumerate(elevation_proj.points):
        # Buscar candidatos en la proyección de planta que tengan una X similar
        candidates_plan = [
            j for j, (x_plan, z_plan) in enumerate(plan_proj.points)
            if abs(x_plan - x_elevation) < epsilon and j not in used_plan
        ]

        # Buscar candidatos en la proyección de perfil que tengan una Y similar
        candidates_section = [
            k for k, (y_section, z_section) in enumerate(section_proj.points)
            if abs(y_section - y_elevation) < epsilon and k not in used_section
        ]

        if not candidates_plan or not candidates_section:
            print(f"No se encontraron candidatos para el punto alzado {i}: ({x_elevation}, {y_elevation})")
            continue

        # En caso de múltiples candidatos, se selecciona el que minimice la diferencia
        j = min(candidates_plan, key=lambda idx: abs(plan_proj.points[idx][0] - x_elevation))
        k = min(candidates_section, key=lambda idx: abs(section_proj.points[idx][0] - y_elevation))

        # Obtener la coordenada Z de ambas proyecciones
        z_plan = plan_proj.points[j][1]
        z_section = section_proj.points[k][1]
        if abs(z_plan - z_section) > epsilon:
            print(f"Inconsistencia en Z para el punto alzado {i}: {z_plan} - {z_section}")
            continue

        # Reconstruir el punto 3D (x, y, z), donde z es el promedio de los dos valores
        z = (z_plan + z_section) / 2.0
        points_3d.append([x_elevation, y_elevation, z])
        mapping[i] = len(points_3d) - 1
        used_plan.add(j)
        used_section.add(k)

    points_3d = np.array(points_3d)

    # Reconstruir las aristas
    edges_3d = []
    for (i, j) in orig_edges:
        if i in mapping and j in mapping and mapping[i] != mapping[j]:
            edges_3d.append(tuple(sorted((mapping[i], mapping[j]))))

    # Eliminar aristas duplicadas
    edges_3d = np.array(list(set(edges_3d)))

    return points_3d, edges_3d


def reconstruct_mesh(points, epsilon=1e-6):
    """
    Reconstruye una malla a partir de la nube de puntos 3D.

    Se utiliza 'Convex Hull' si la nube tiene al menos 4 puntos y las varianzas en cada dimensión
    son significativas. En otro caso, se proyecta a 2D usando las dos dimensiones con mayor
    varianza y se aplica 'Delaunay'.

    :param points: Matriz de puntos 3D.
    :param epsilon: Tolerancia para considerar varianza significativa.
    :return:
        - faces: Conjunto de caras (índices) que forman la malla.
    """

    if points.shape[0] >= 4 and np.std(points, axis=0).min() >= epsilon:
        hull = ConvexHull(points)
        faces = hull.simplices
        return faces
    else:
        # Proyectar a 2D usando las dos dimensiones con mayor varianza
        variances = np.var(points, axis=0)
        dims = np.argsort(variances)[-2:]
        points_2d = points[:, dims]
        tri = Delaunay(points_2d)
        faces = tri.simplices
        return faces


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

    [ ESTA VERIFICACIÓN ASUME MODELOS SÓLIDOS CERRADOS ]

    :param points: Matriz de puntos 3D.
    :param edges: Matriz de aristas.
    :return:
        - bool: True si el modelo es consistente.
                False en caso contrario.
    """

    if len(points) < 3:
        return False

    degrees = np.zeros(len(points), dtype=int)
    for edge in edges:
        degrees[edge[0]] += 1
        degrees[edge[1]] += 1

    if len(points) > 4:
        for i, d in enumerate(degrees):
            if d < 3:
                print(f"Inconsistencia global en el vértice {i} con coordenadas {points[i]} [grado {d}]")
                return False
    return True


def check_projection_consistency(plan_proj, elevation_proj, section_proj, orig_edges, epsilon=1e-6):
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
    :param orig_edges: Aristas originales de la nube de puntos 3D.
    :param epsilon: Tolerancia para la comparación de coordenadas.
    :return:
        - bool: True si se cumplen todas las comprobaciones.
                False en caso contrario.
    """

    try:
        # 1. Verificar la conectividad de cada proyección
        if not check_projection_connectivity((plan_proj, elevation_proj, section_proj)):
            return False

        # 2. Reconstruir el modelo 3D a partir de las proyecciones
        points_3d, edges_3d = reconstruct_3d_from_projections(plan_proj, elevation_proj, section_proj, orig_edges)
        if len(points_3d) == 0:
            return False

        # 3. Verificar consistencia global del modelo 3D
        if not check_model_consistency(points_3d, edges_3d):
            return False

        # 4. Reconstruir las proyecciones a partir del modelo 3D obtenido
        recon_plan, recon_elevation, recon_section = project_cloud(points_3d, edges_3d)

        # 5. Verificar que el número de puntos coincide en cada proyección
        if (len(plan_proj.points) != len(recon_plan.points) or
                len(elevation_proj.points) != len(recon_elevation.points) or
                len(section_proj.points) != len(recon_section.points)):
            return False

        # Comparar cada punto en la proyección original con algún punto en la reconstruida
        for proj_orig, proj_recon in [
            (plan_proj, recon_plan),
            (elevation_proj, recon_elevation),
            (section_proj, recon_section)
        ]:
            for point_orig in proj_orig.points:
                match_found = any(
                    abs(point_orig[0] - point_recon[0]) < epsilon and
                    abs(point_orig[1] - point_recon[1]) < epsilon
                    for point_recon in proj_recon.points
                )
                if not match_found:
                    return False
        return True

    except Exception as e:
        print(f"Error al verificar consistencia: {e}")
        return False


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
        points, edges = shape

        # Generar las proyecciones ortogonales
        plan, elevation, section = project_cloud(points, edges)

        # Comprobar si las proyecciones son consistentes entre sí y con el modelo 3D
        if check_projection_consistency(plan, elevation, section, edges):
            # Reconstruir el modelo 3D a partir de las proyecciones
            reconstruct_points, reconstruct_edges = reconstruct_3d_from_projections(plan, elevation, section, edges)

            # Visualizar las proyecciones originales en 2D
            plot_projections((plan, elevation, section))

            # Visualizar el modelo 3D reconstruido
            plot_3d_model(reconstruct_points, reconstruct_edges)

            # Reconstruir y visualizar la malla 3D
            faces = reconstruct_mesh(reconstruct_points)
            plot_3d_mesh(reconstruct_points, faces)
        else:
            print(f"No se pudo reconstruir la forma {name} a partir de sus proyecciones.")


if __name__ == '__main__':
    main()
