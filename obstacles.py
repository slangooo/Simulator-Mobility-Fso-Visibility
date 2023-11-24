import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from resources.data_structures import Obstacle
from resources.math_utils import line_point_angle, get_mid_azimuth, angle_in_range
from parameters import BS_HEIGHT, SUN_SEARCH_STEP, MAX_SUN_SEARCH_STEPS, BUILDING_EDGE_MARGIN, SUN_SEARCH_COUNT

BOUNDARY_MARGIN = 20
EXTEND_TIMES_FOUR = True

obstacles_madrid_list = [[9, 423, 129, 543, 52.5 + 60],
                         [9, 285, 129, 405, 49 + 60],
                         [9, 147, 129, 267, 42 + 60],
                         [9, 9, 129, 129, 45.5 + 60],
                         [147, 423, 267, 543, 31.5 + 60],
                         [147, 147, 267, 267, 52.5 + 60],
                         [147, 9, 267, 129, 28 + 60],
                         [297, 423, 327, 543, 31.5 + 60],
                         [297, 285, 327, 405, 45.5 + 60],
                         [297, 147, 327, 267, 38.5 + 60],
                         [297, 9, 327, 129, 42 + 60],
                         [348, 423, 378, 543, 45.5 + 60],
                         [348, 285, 378, 405, 49 + 60],
                         [348, 147, 378, 267, 38.5 + 60],
                         [348, 9, 378, 129, 42 + 60]]


# obstacles_madrid_list =  [[423, 9, 543, 129, 52.5],
#                           [285, 9, 405, 129, 49],
#                           [147, 0, 267, 140, 42],
#                           [9, 9, 129, 110, 45.5],
#                           [423, 147, 543, 267, 31.5],
#                           [147, 160, 267, 250, 52.5],
#                           [9, 125, 129, 267, 28],
#                           [423, 297, 543, 327, 31.5],
#                           [285, 220, 405, 327, 45.5],
#                           [147, 297, 267, 327, 38.5],
#                           [9, 297, 129, 327, 42],
#                           [423, 348, 543, 378, 45.5],
#                           [285, 348, 405, 378, 49],
#                           [147, 348, 267, 378, 38.5],
#                           [9, 348, 129, 378, 42]]

# obstacles_madrid_list = [[29, 420, 120, 520, 52.5],
#                          [20, 300, 100, 378, 49],
#                          [0, 150, 120, 265, 42],
#                          [9, 9, 140, 110, 45.5],
#                          [150, 425, 255, 532, 31.5],
#                          [135, 190, 280, 340, 52.5],
#                          [160, 9, 255, 129, 28],
#                          [270, 430, 327, 525, 31.5],
#                          [290, 315, 350, 395, 45.5],
#                          [295, 155, 327, 255, 38.5],
#                          [297, 40, 327, 129, 42],
#                          [360, 440, 395, 515, 45.5],
#                          [360, 295, 420, 395, 49],
#                          [352, 155, 412, 255, 38.5],
#                          [360, 9, 405, 120, 42]]

class Obstacles(object):
    obstaclesList = []

    def __init__(self, obstacles_data_list, vertices_format='axes'):
        for obstacle_id, obstacle in enumerate(obstacles_data_list):
            vertices = []
            if vertices_format == 'coordinates':
                for idx in range(0, (len(obstacle) - 1), 2):
                    vertices.append((obstacle[idx], obstacle[idx + 1]))
            elif vertices_format == 'axes':
                vertices = [(obstacle[0], obstacle[1]), (obstacle[0], obstacle[3]),
                            (obstacle[2], obstacle[3]), (obstacle[2], obstacle[1])]
            self.obstaclesList.append(Obstacle(obstacle_id, obstacle[-1], vertices))

    def get_total_vertices(self):
        total_vertices = []
        for obstacle in self.obstaclesList:
            total_vertices = total_vertices + obstacle.vertices
        return total_vertices

    def get_sunny_vertices(self, sun_azimuth, sun_elevation):
        total_vertices = []
        if sun_elevation < 0:
            return []
        for obstacle in self.obstaclesList:
            for _vertex in obstacle.vertices:
                sun_search_steps = MAX_SUN_SEARCH_STEPS
                [adj_1, adj_2] = obstacle.get_adjacent_vertices(_vertex)
                offset_azimuth, max_azimuth, min_azimuth = get_mid_azimuth(_vertex, adj_1, adj_2)

                direction1 = line_point_angle(length=SUN_SEARCH_STEP * (SUN_SEARCH_COUNT - 1),
                                              point=[_vertex[0], _vertex[1]],
                                              angle_x=max_azimuth)
                direction2 = line_point_angle(length=SUN_SEARCH_STEP * (SUN_SEARCH_COUNT - 1),
                                              point=[_vertex[0], _vertex[1]],
                                              angle_x=min_azimuth)
                x_2 = direction1[0] if direction1[0] != _vertex[0] else direction2[0]
                y_2 = direction1[1] if direction1[1] != _vertex[1] else direction2[1]

                direction1 = line_point_angle(length=SUN_SEARCH_STEP, point=[_vertex[0], _vertex[1]],
                                              angle_x=max_azimuth + 180)
                direction2 = line_point_angle(length=SUN_SEARCH_STEP, point=[_vertex[0], _vertex[1]],
                                              angle_x=min_azimuth + 180)

                x_1 = direction1[0] if direction1[0] != _vertex[0] else direction2[0]
                y_1 = direction1[1] if direction1[1] != _vertex[1] else direction2[1]

                x_ticks = np.roll(np.linspace(x_1, x_2, SUN_SEARCH_COUNT, endpoint=False), SUN_SEARCH_COUNT - 1)
                y_ticks = np.roll(np.linspace(y_1, y_2, SUN_SEARCH_COUNT, endpoint=False), SUN_SEARCH_COUNT - 1)

                idxes = np.array(np.meshgrid([np.arange(0, SUN_SEARCH_COUNT)],
                                             [np.arange(0, SUN_SEARCH_COUNT)])).T.reshape(-1, 2).T

                idxes = np.array(sorted(idxes.T.tolist(), key=lambda x: (x[0] + x[1] + ((x[0] * x[1]) == 0)) / 2))

                idxes = idxes[3:-1]
                # idxes = idxes[(idxes != 1).any(axis=1)]

                for idx_x, idx_y in idxes:
                    x_coord, y_coord = x_ticks[idx_x], y_ticks[idx_y]
                    not_valid = False
                    for blocking_obstacle in self.obstaclesList:
                        if blocking_obstacle.is_overlapping(x_coord, y_coord, BS_HEIGHT):
                            not_valid = True
                            break
                        if blocking_obstacle.is_blocking(x_coord, y_coord, sun_azimuth, sun_elevation,
                                                         reference_height=BS_HEIGHT):
                            not_valid = True
                            break
                    if not not_valid:
                        total_vertices.append([x_coord, y_coord])
                        break
        return total_vertices

    def get_total_edges(self):
        edges = []
        for obstacle in self.obstaclesList:
            obstacle_poly = obstacle.vertices + [obstacle.vertices[0]]
            for idx in range(len(obstacle.vertices)):
                edges.append([obstacle_poly[idx], obstacle_poly[idx + 1]])
        return edges

    def print_obstacles(self):
        for obstacle in self.obstaclesList:
            print(obstacle.id, ": ", obstacle.vertices, obstacle.height)

    def plot_obstacles(self, show_flag=False, fill_color='Gray'):
        if not fill_color:
            for obstacle in self.obstaclesList:
                xs, ys = zip(*obstacle.vertices + [obstacle.vertices[0]])
                plt.plot(xs, ys, c='dimgray')
        else:
            rects = []
            for obstacle in self.obstaclesList:
                xs, ys = zip(*obstacle.vertices + [obstacle.vertices[0]])
                xs = set(xs)
                ys = set(ys)
                corner = min(xs), min(ys)
                height = max(ys) - min(ys)
                width = max(xs) - min(xs)
                rects.append(Rectangle(corner, width, height, color=fill_color))
            return rects

        if show_flag:
            print("SHOWING")
            plt.show()

    def get_boundaries(self):
        x_min, x_max = self.obstaclesList[0].vertices[0][0], self.obstaclesList[0].vertices[0][0]
        y_min, y_max = self.obstaclesList[0].vertices[0][1], self.obstaclesList[0].vertices[0][1]
        for obstacle in self.obstaclesList:
            xs, ys = zip(*obstacle.vertices)
            x_min, x_max = min(x_min, min(xs)), max(x_max, max(xs))
            y_min, y_max = min(y_min, min(ys)), max(y_max, max(ys))
        return [[x_min, x_max], [y_min, y_max]]

    def get_margin_boundary(self, as_polygon=True):
        xs, ys = self.get_boundaries()
        x_min, x_max = xs[0] - BOUNDARY_MARGIN, xs[1] + BOUNDARY_MARGIN
        y_min, y_max = ys[0] - BOUNDARY_MARGIN, ys[1] + BOUNDARY_MARGIN
        if not as_polygon:
            return [x_min, x_max], \
                   [y_min, y_max]
        else:
            return (x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)

    def get_total_segments(self):
        segments = []
        bds = self.get_margin_boundary()
        for i in range(len(bds)):
            segments += [(bds[i], bds[(i + 1) % len(bds)])]

        for obstacle in self.obstaclesList:
            for i in range(len(obstacle.vertices)):
                segments += [(obstacle.vertices[i], obstacle.vertices[(i + 1) % len(obstacle.vertices)])]
        return segments


def get_madrid_buildings():
    # #For polish paper
    # for _bldg in obstacles_madrid_list:
    #     _bldg[0], _bldg[1], _bldg[2], _bldg[3] = _bldg[1], _bldg[0], _bldg[3], _bldg[2]
    # for _bldg in obstacles_madrid_list:
    #     _bldg[-1] += 6
    new_obstacles = obstacles_madrid_list.copy()
    if EXTEND_TIMES_FOUR:
        x_shift_step = 400 # Total dimensions for Madrid Grid is 387 m (east-west) and 552 m (south north).  The
                      # building height is uniformly distributed between 8 and 15 floors with 3.5 m per floor
        y_shift_step = 570
        for x_steps in range(0, 3):
            for y_steps in range(0, 3):
                extension = []
                for _bldg in obstacles_madrid_list:
                    extension.append([_bldg[0] + x_steps * x_shift_step, _bldg[1] + y_steps * y_shift_step,
                                      _bldg[2] + x_steps * x_shift_step, _bldg[3] + y_steps * y_shift_step, _bldg[4]])
                # for _bldg in obstacles_madrid_list:
                #     extension.append([_bldg[0], _bldg[1] + y_steps * y_shift_step,
                #                       _bldg[2], _bldg[3] + y_steps * y_shift_step, _bldg[4]])
                # for _bldg in obstacles_madrid_list:
                #     extension.append([_bldg[0] + x_steps * x_shift_step,
                #                       _bldg[1] + y_steps * y_shift_step, _bldg[2] + x_steps * x_shift_step,
                #                       _bldg[3] + y_steps * y_shift_step, _bldg[4]])
                new_obstacles += extension

    return Obstacles(new_obstacles)


if __name__ == '__main__':
    _rects = get_madrid_buildings().plot_obstacles(False, fill_color='gray')
    ax = plt.gca()
    for _rect in _rects:
        ax.add_patch(_rect)
    # xs = np.array([0, -5, 250.0])
    # ys = np.array([0.0, 420, 380.0])
    # plt.plot(xs, ys, c='red')
    # plt.plot([], [], c='red', label='Łącze FSO')
    # plt.plot(0, 0, c='blue', marker='s', label='MBS', linestyle='none')
    plt.plot(600, 900, c='blue', marker='s', label='MBS', linestyle='none')
    # plt.plot(250.0, 380.0, c='green', marker='o', label='Hotspot', linestyle='none')
    # plt.plot(200.0, 370.0, c='green', marker='o', label='Hotspot center', linestyle='none')
    plt.legend(loc="upper left")
    plt.show()
    # plt.savefig('plots/madrid_modified_plus_stations_polish_final.eps', format='eps')
