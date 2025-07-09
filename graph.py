class Edge:
    def __init__(self, vertex, cost: float = 1.0):
        self.vertex = vertex
        self.cost = cost


class Vertex:
    def __init__(self, name):
        self.name = name
        self.edges = set()

    def add_edge(self, vertex, cost: float = 1.0):
        self.edges.add(Edge(vertex, cost))

    def neighbors(self):
        return (e.vertex for e in self.edges)


class Graph:
    def __init__(self):
        self.adj = {}

    def add_vertex(self, name):
        if name not in self.adj:
            self.adj[name] = Vertex(name)
        return self.adj[name]

    def add_edge(self, u, v, cost: float = 1.0):
        v1 = self.add_vertex(u)
        v2 = self.add_vertex(v)
        v1.add_edge(v2, cost)
        v2.add_edge(v1, cost)
