from paths import Paths
import numpy as np
import heapq
from resources.data_structures import Coords
import matplotlib.pyplot as plt

class Vertex:
    """Defines node in graph"""

    def __init__(self, node_id, coords):
        self.id = node_id
        self.adjacent = {}
        self.distance = float('inf')
        self.visited = False
        self.previous = None
        self.coords = coords

    def __str__(self):
        return str(self.coords)

    def __eq__(self, other):
        if isinstance(other, Coords):
            return self.coords == other
        else:
            return self.coords == other.coords

    def __key__(self):
        return self.coords

    def __hash__(self):
        return hash(self.__key__())

    def __lt__(self, other):
        return self.id < other.id

    def add_neighbor(self, neighbor, weight=0):
        if neighbor in self.adjacent:
            print("WARNING! neighbor already exists")

        self.adjacent[neighbor.id] = weight

    # def get_connections(self):
    #     return self.adjacent.keys()

    def get_id(self):
        return self.id

    def get_weight(self, neighbor):
        return self.adjacent[neighbor.id]

    def set_distance(self, dist):
        self.distance = dist

    def get_distance(self):
        return self.distance

    def set_previous(self, prev):
        self.previous = prev

    def set_visited(self):
        self.visited = True

    def reset(self):
        self.distance = float('inf')
        self.visited = False
        self.previous = None


class Graph:
    """Defines graph structure with vertices and weighted edges"""

    def __init__(self, paths_segments, obstacles_objects = None):
        self.vert_dict = {}
        self.num_vertices = 0
        self.paths_segments = paths_segments
        self.obstacles_objects = obstacles_objects

    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_vertex(self, node_id, coords):
        if node_id in self.vert_dict.keys():
            raise ValueError("node with given ID already exists!")

        self.num_vertices = self.num_vertices + 1
        new_vertex = Vertex(node_id, Coords(coords[0], coords[1]))

        if new_vertex in self.vert_dict.values():
            raise ValueError("node with given coordinates already exists!")

        self.vert_dict[node_id] = new_vertex

        return new_vertex

    def get_vertex(self, node_id):
        if node_id in self.vert_dict:
            return self.vert_dict[node_id]
        else:
            return None

    def add_edge(self, frm, to, cost=0.0):
        # Find node_ids to given coords
        frm_id = list(self.vert_dict.values())[list(self.vert_dict.values()).index(frm)].id
        to_id = list(self.vert_dict.values())[(list(self.vert_dict.values()).index(to))].id

        self.vert_dict[frm_id].add_neighbor(self.vert_dict[to_id], cost)
        self.vert_dict[to_id].add_neighbor(self.vert_dict[frm_id], cost)

    def get_vertices(self, return_coords=False):
        """Returns all vertices"""
        if not return_coords:
            return self.vert_dict
        return [v.coords for v in self.vert_dict.values()]

    def reset_graph(self):
        for vertex in self.vert_dict.values():
            vertex.reset()

    def get_path_from_to(self, frm, to, coords_result=True):
        self.reset_graph()
        path_list = dijkstra(self, frm, to)
        path_list.reverse()
        # for idx in range(len(path_list)-1):
        #     assert (path_list[idx+1] in self.vert_dict[path_list[idx]].adjacent.keys())
        if coords_result:
            return self.get_path_coords(path_list)
        return path_list

    def get_path_coords(self, path):
        path_coords = []
        for v_id in path:
            path_coords.append(self.vert_dict[v_id].coords)
        return path_coords

    def plot_segments(self, show_flag = False):
        for segment in self.paths_segments:
            xs, ys = zip(*segment)
            plt.plot(xs, ys, markerfacecolor='black', linestyle='dotted', linewidth=1, color='black')
        if show_flag:
            plt.show()

def get_graph_from_segments(paths=None):
    """Generates graph from given paths or default ones"""
    if not paths:
        # Obtain paths segments using default settings and obstacles (Madrid)
        # Otherwise do something like Paths(Obstacles(obstacles_list))
        paths = Paths()

    # Create graph object
    graph = Graph(paths.paths_segments, paths.obstacles_obj)

    # Add all vertices
    for idx, vertex_coords in enumerate(paths.all_vertices):
        graph.add_vertex(idx, vertex_coords)

    # Add all edges
    for segment in paths.paths_segments:
        graph.add_edge(Coords(segment[0][0], segment[0][1]), Coords(segment[1][0], segment[1][1]),
                       float(np.linalg.norm(segment[1] - segment[0])))
    return graph


def shortest(vert, path):
    """ make shortest path from v.previous"""
    if vert.previous:
        path.append(vert.previous.get_id())
        shortest(vert.previous, path)
    return


def dijkstra(graph, start_id, target_id):
    """Dijkstra's shortest path"""
    start = graph.get_vertex(start_id)
    target = graph.get_vertex(target_id)

    # Set the distance for the start node to zero
    start.set_distance(0)

    # Put tuple pair into the priority queue
    unvisited_queue = [(v.get_distance(), v) for v in graph]
    heapq.heapify(unvisited_queue)

    while len(unvisited_queue):
        # Pops a vertex with the smallest distance
        uv = heapq.heappop(unvisited_queue)
        current = uv[1]
        current.set_visited()

        # for next_v in v.adjacent:
        for next_v_id in current.adjacent:
            next_v = graph.vert_dict[next_v_id]
            # if visited, skip
            if next_v.visited:
                continue
            new_dist = current.get_distance() + current.get_weight(next_v)

            if new_dist < next_v.get_distance():
                next_v.set_distance(new_dist)
                next_v.set_previous(current)

            else:
                pass

        # Rebuild heap
        # 1. Pop every item
        while len(unvisited_queue):
            heapq.heappop(unvisited_queue)
        # 2. Put all vertices not visited into the queue
        unvisited_queue = [(v.get_distance(), v) for v in graph if not v.visited]
        heapq.heapify(unvisited_queue)

    path = [target.get_id()]
    shortest(target, path)
    return path


if __name__ == '__main__':
    g = get_graph_from_segments()
    # g.plot_segments(False)

    for vert in g.vert_dict.values():
        for vert_id in vert.adjacent.keys():
            xs = [vert.coords.x, g.vert_dict[vert_id].coords.x]
            ys = [vert.coords.y, g.vert_dict[vert_id].coords.y]
            plt.plot(xs, ys, markerfacecolor='black', linestyle='dotted', linewidth=1, color='black')
    plt.show()

