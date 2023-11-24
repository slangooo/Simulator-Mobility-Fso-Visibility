from icc_simulation import IccSimulator
from visibility_graph import Point, VisGraph, Edge
from copy import deepcopy
from resources.data_structures import Coords
import visibility_polygon
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np
from itertools import repeat, compress
from itertools import chain as iter_chain
from multiprocessing import Pool

N_HOPS_ALLOWED = 2
RIS_ACTIVE = True


class Wall:
    def __init__(self, start_coords, end_coords, ris_feasible, orientation):
        self.start_coords = start_coords
        self.end_coords = end_coords
        self.ris_installed = False
        self.ris_feasible = ris_feasible
        self.orientation = orientation


class RIS:
    def __init__(self, x, y, wall):
        self.x = x
        self.y = y
        self.wall = wall
        self.active_flag = RIS_ACTIVE


class WwrfSimulator(IccSimulator):
    def __init__(self, renewable_energy_flag):
        super().__init__(renewable_energy_flag)
        self.ris_list = []
        self.walls = []
        self.populate_walls()
        self.segments = self.get_obstacles_segments()
        self.vertices_list = []

    def build_graph(self):
        super().build_graph()
        self.graph_backup = deepcopy(self.graph)
        self.update_vertices_list()

    def update_vertices_list(self):
        self.vertices_list = [(item.x, item.y) for item in self.total_vertices]

    def reload_graph_from_backup(self):
        self.graph = deepcopy(self.graph_backup)

    def add_edges_to_visibility_graph(self, point_1, point_2, ris):
        if point_2 in self.graph.visgraph.get_adjacent_points(point_1):
            return
        _edge = Edge(point_1, point_2)
        _edge.ris_hop = ris
        self.graph.visgraph.add_edge(_edge)

    # for _pt in self.graph.graph.get_points():
    #     if _pt.x == 285:
    #         print(_pt.y)

    def get_obstacles_segments(self, flatten=True):
        segs = self._controller.mobility_model.graph.obstacles_objects.get_total_segments()
        return segs if not flatten else [item for sublist in segs for item in sublist]

    def populate_walls(self):
        segs = self.get_obstacles_segments(False)

        # remove boundaries
        for i in range(4):
            segs.pop(0)

        [min_x, max_x], [min_y, max_y] = self._controller.mobility_model.obstacles_objects.get_boundaries()

        margin_from_boundaries = 20
        min_x += margin_from_boundaries
        min_y += margin_from_boundaries
        max_x -= margin_from_boundaries
        max_y -= margin_from_boundaries

        for i in range(0, len(segs), 4):
            # min_bldg_x = min(segs[0][0], segs[1][0], segs[2][0])
            max_bldg_x = max(segs[i][0][0], segs[i][1][0], segs[i + 1][0][0], segs[i + 1][1][0])
            # min_bldg_y = min(segs[0][1], segs[1][1], segs[2][1])
            max_bldg_y = max(segs[i + 0][0][1], segs[i][1][1], segs[i + 1][0][1], segs[i + 1][1][1])
            for _seg in segs[i:i + 4]:
                if (_seg[0][0] != _seg[1][0]):
                    if (_seg[0][1] == max_bldg_y):
                        orientation = 'u'
                    else:
                        orientation = 'd'
                else:
                    if (_seg[0][0] == max_bldg_x):
                        orientation = 'r'
                    else:
                        orientation = 'l'

                if _seg[0][0] > min_x and _seg[0][0] < max_x and _seg[0][1] > min_y and _seg[0][1] < max_y \
                        and _seg[1][0] > min_x and _seg[1][0] < max_x and _seg[1][1] > min_y and _seg[1][1] < max_y:
                    self.walls.append(Wall(_seg[0], _seg[1], True, orientation))
                else:
                    self.walls.append(Wall(_seg[0], _seg[1], False, orientation))

    def install_ris_to_wall(self, idx):
        wall = self.walls[idx]
        if wall.ris_installed:
            print("RIS already installed here")
            return
        wall.ris_installed = True
        ris_x = (wall.start_coords[0] + wall.end_coords[0]) / 2
        ris_y = (wall.start_coords[1] + wall.end_coords[1]) / 2

        # # remove for harder
        shift = 0.01
        # n_idx = idx / 4
        # number_dec = n_idx - int(n_idx)
        # n_idx = number_dec * 4

        if (wall.orientation == 'l'):
            ris_x -= shift
        elif (wall.orientation == 'd'):
            ris_y -= shift
        elif (wall.orientation == 'r'):
            ris_x += shift
        else:
            ris_y += shift

        ris_obj = RIS(ris_x, ris_y, wall)
        self.ris_list.append(ris_obj)
        return ris_obj

    def get_ris_visibility(self, ris_obj):
        vis = visibility_polygon.get_visibility(self.segments, [ris_obj.x, ris_obj.y])
        return vis

    def get_vertices_inside_polygon(self, poly_in):
        pool = Pool(4)
        results = list(
            pool.starmap(self.check_if_point_in_polygon, zip(repeat(poly_in), self.vertices_list), chunksize=10))
        return list(compress(self.vertices_list, results))

    @staticmethod
    def check_if_point_in_polygon(input_poly: list, test_pt: tuple) -> bool:
        if visibility_polygon.point_in_polygon(input_poly, test_pt):
            return True
        else:
            return False

    def install_ris_and_update_visibility(self, wall_idx):
        if not self.walls[wall_idx].ris_feasible:
            return False
        new_ris = self.install_ris_to_wall(wall_idx)
        vis = self.get_ris_visibility(new_ris)
# _, ax = plt.subplots()
# rects_patches = self.plot_obstacles()
# for _patch in rects_patches:
#     ax.add_patch(_patch)
# ax.add_collection(PatchCollection([Polygon(np.array(vis), closed=True)], fc='g', ec='r'))
# bds = self._controller.mobility_model.graph.obstacles_objects.get_margin_boundary()
# ax.set_xlim(bds[0][0], bds[2][0])
# ax.set_ylim(bds[0][1], bds[2][1])
# plt.show()
#         bla = 5
        vertices_within_ris = self.get_vertices_inside_polygon(vis)
        self.make_vertices_visible(vertices_within_ris, new_ris)
        return True

    def make_vertices_visible(self, vertices_within_ris, ris):
        for i, pt_1 in enumerate(vertices_within_ris):
            for j, pt_2 in enumerate(vertices_within_ris):
                if i == j:
                    continue
                self.add_edges_to_visibility_graph(Point(pt_1[0], pt_1[1]), Point(pt_2[0], pt_2[1]), ris)

    def get_shortest_path_to_dest(self, point_dest):
        return self.graph.shortest_path \
            (Point(self.src[0], self.src[1]), point_dest, [])

    def get_n_hops_to_dest(self, dest):
        return len(self.get_shortest_path_to_dest(dest))

    def get_pts_reachable_with_n_hops(self, n_hops=N_HOPS_ALLOWED):
        # remove for harder
        pool = Pool(4)
        results = list(pool.imap(self.get_n_hops_to_dest, self.total_vertices, chunksize=10))
        flags = [i < n_hops + 2 for i in results]
        return list(compress(self.total_vertices, flags))

    def plot_points(self, points, ris_involved=False):
        for idx, _vertex in enumerate(points):
            if not idx:
                plt.plot(_vertex.x, _vertex.y, c=('green' if not ris_involved else 'red'), marker='o',
                         label=f'Max {N_HOPS_ALLOWED} hops' + \
                               (' and RIS' if ris_involved else ''),
                         linestyle='none')
            else:
                plt.plot(_vertex.x, _vertex.y, c=('green' if not ris_involved else 'red'), marker='o')

    def plot_wwrf_results(self):
        self.plot_points()
        self.plot_obstacles()
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()

        def flip(items, ncol):
            return iter_chain(*[items[i::ncol] for i in range(ncol)])

        plt.legend(flip(handles, 2), flip(labels, 2), loc='upper left', ncol=2)
        plt.show()

    def random_ris_installation(self, n_walls=20):
        random_idxs = np.random.permutation(len(self.walls))
        ctr, idx = 0, 0
        while (ctr < n_walls):
            if (self.install_ris_and_update_visibility(random_idxs[idx])):
                ctr += 1
            idx += 1

    def get_square_walls_idxs(self):
        selected_walls = []
        # xs = [147 + i * 400 for i in range(3)]
        # xs += [267 + i * 400 for i in range(3)]
        # ys = [267 + i * 570 for i in range(3)]
        # ys += [423 + i * 570 for i in range(3)]
        xs = [147 + 400]
        xs += [267 + 400]
        ys = [267 + 570]
        ys += [423 + 570]
        for idx, _wall in enumerate(self.walls):
            if (_wall.start_coords[0] in xs and _wall.end_coords[0] in xs and \
                    _wall.start_coords[1] in ys and _wall.end_coords[1] in ys):
                selected_walls.append(idx)
        return selected_walls

    def install_ris_square(self):
        idxs = self.get_square_walls_idxs()
        for _idx in idxs:
            self.install_ris_and_update_visibility(_idx)

    def plot_ris(self):
        for idx, _ris in enumerate(self.ris_list):
            xs, ys = zip(_ris.wall.start_coords, _ris.wall.end_coords)
            if not idx:
                plt.plot(xs, ys, c='black', label='RIS', linestyle='solid', linewidth=4)
            else:
                plt.plot(xs, ys, c='black', linestyle='solid', linewidth=4)



if __name__ == '__main__':
    simulator = WwrfSimulator(renewable_energy_flag=False)
    simulator.set_source_and_dest([600, 900], None)
    simulator.build_graph()

    # prev_n = 0
    # n_pts_per_n_hops = []
    # reachable_pts = simulator.get_pts_reachable_with_n_hops()
    # for i in range(8):
    #     n_pts_per_n_hops.append(len(simulator.get_pts_reachable_with_n_hops(i)) - prev_n)
    #     prev_n = n_pts_per_n_hops[-1]

    # simulator.install_ris_square()

    prev_n = 0
    n_pts_per_n_hops_ris = []
    # reachable_pts = simulator.get_pts_reachable_with_n_hops()
    # for i in range(8):
    #     n_pts_per_n_hops_ris.append(len(simulator.get_pts_reachable_with_n_hops(i)) - prev_n)
    #     prev_n = n_pts_per_n_hops_ris[-1]

    # simulator.plot_points(reachable_pts)
    # ax = plt.gca()
    # rects_patches = simulator.plot_obstacles()
    # for _patch in rects_patches:
    #     ax.add_patch(_patch)

    # With RIS
    simulator.install_ris_square()
    reachable_pts_ris = simulator.get_pts_reachable_with_n_hops()
    # reachable_pts_new = list(set(reachable_pts_ris) ^ set(reachable_pts))
    # simulator.plot_points(reachable_pts_new, True)
    # simulator.plot_endpoints()
    # simulator.plot_ris()
    #
    # handles, labels = ax.get_legend_handles_labels()
    # plt.legend(handles, labels, loc='upper left', ncol=2)
    # plt.show()

    # _, ax = plt.subplots()
    # simulator.plot_obstacles()
    # ax = plt.gca()
    # ax.add_collection(PatchCollection([Polygon(np.array(vis), closed=True)], fc='g', ec='r'))
    # ax.set_xlim(-20, 1200)
    # ax.set_ylim(-20, 1750)
    # plt.show()
    # simulator.set_source_and_dest([600, 900], [1000, 130])
    # simulator.build_graph()
    # simulator.plot_default_vertices()
    # simulator.plot_results()
