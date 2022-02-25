import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from resources.data_structures import Obstacle
from resources.math_utils import line_point_angle, get_mid_azimuth, angle_in_range
from parameters import BS_HEIGHT, SUN_SEARCH_STEP, MAX_SUN_SEARCH_STEPS, BUILDING_EDGE_MARGIN, SUN_SEARCH_COUNT

obstacles_madrid_list = [[29, 420, 120, 520, 52.5],
                         [20, 300, 100, 378, 49],
                         [0, 150, 120, 265, 42],
                         [9, 9, 140, 110, 45.5],
                         [150, 425, 255, 532, 31.5],
                         [135, 190, 280, 340, 52.5],
                         [160, 9, 255, 129, 28],
                         [270, 430, 327, 525, 31.5],
                         [290, 315, 350, 395, 45.5],
                         [295, 155, 327, 255, 38.5],
                         [297, 40, 327, 129, 42],
                         [360, 440, 395, 515, 45.5],
                         [360, 295, 420, 395, 49],
                         [352, 155, 412, 255, 38.5],
                         [360, 9, 405, 120, 42]]


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
                #######
                # if _vertex == (405, 9):
                #     bla = 5
                direction1 = line_point_angle(length=SUN_SEARCH_STEP*(SUN_SEARCH_COUNT-1), point=[_vertex[0], _vertex[1]],
                                                    angle_x=max_azimuth)
                direction2 = line_point_angle(length=SUN_SEARCH_STEP *(SUN_SEARCH_COUNT-1), point=[_vertex[0], _vertex[1]],
                                              angle_x=min_azimuth)
                x_2 = direction1[0] if direction1[0] != _vertex[0] else direction2[0]
                y_2 = direction1[1] if direction1[1] != _vertex[1] else direction2[1]

                direction1 = line_point_angle(length=SUN_SEARCH_STEP, point=[_vertex[0], _vertex[1]],
                                              angle_x=max_azimuth + 180)
                direction2 = line_point_angle(length=SUN_SEARCH_STEP, point=[_vertex[0], _vertex[1]],
                                              angle_x=min_azimuth + 180)

                x_1 = direction1[0] if direction1[0] != _vertex[0] else direction2[0]
                y_1 = direction1[1] if direction1[1] != _vertex[1] else direction2[1]

                x_ticks = np.roll(np.linspace(x_1, x_2, SUN_SEARCH_COUNT, endpoint=False), SUN_SEARCH_COUNT-1)
                y_ticks = np.roll(np.linspace(y_1, y_2, SUN_SEARCH_COUNT, endpoint=False), SUN_SEARCH_COUNT-1)

                # x_ticks = np.linspace(_vertex[0], x_2, SUN_SEARCH_COUNT, endpoint=False)
                # y_ticks = np.linspace(_vertex[1], y_2, SUN_SEARCH_COUNT, endpoint=False)

                # coords = np.array(np.meshgrid(x_ticks, y_ticks, sparse=False)).T.reshape(-1, 2)
                idxes = np.array(np.meshgrid([np.arange(0, SUN_SEARCH_COUNT)],
                                             [np.arange(0, SUN_SEARCH_COUNT)])).T.reshape(-1,2).T
                # idxes = idxes[:, idxes.sum(axis=0).argsort()].T
                idxes = np.array(sorted(idxes.T.tolist(), key = lambda x : (x[0] + x[1] + ((x[0] * x[1])==0))/2))

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
                ################
                # x_coord, y_coord = line_point_angle(length=BUILDING_EDGE_MARGIN, point=[_vertex[0], _vertex[1]],
                #                                     angle_x=offset_azimuth)
                # perp_to_sun = angle_in_range(sun_azimuth +
                #                              (90 if abs(sun_azimuth - 90 - offset_azimuth)
                #                                     > abs(sun_azimuth + 90 - offset_azimuth) else -90), 360)
                # if angle_in_range(max_azimuth-perp_to_sun, 360) < angle_in_range(max_azimuth-min_azimuth, 360):
                #     sun_search_azimuth = perp_to_sun
                # elif angle_in_range(perp_to_sun - max_azimuth, 360) < angle_in_range(min_azimuth - perp_to_sun, 360):
                #     sun_search_azimuth = max_azimuth
                # else:
                #     sun_search_azimuth = min_azimuth
                # location_updated = True
                # while location_updated:
                #     location_updated = False
                #     for blocking_obstacle in self.obstaclesList:
                #         if blocking_obstacle.is_overlapping(x_coord, y_coord, BS_HEIGHT):
                #             sun_search_steps = 0
                #         while blocking_obstacle.is_blocking(x_coord, y_coord, sun_azimuth, sun_elevation,
                #                                    reference_height=BS_HEIGHT) and sun_search_steps > 0:
                #             location_updated = True
                #             x_coord, y_coord = line_point_angle(length=SUN_SEARCH_STEP, point=[x_coord, y_coord],
                #                                                 angle_x=sun_search_azimuth)
                #             sun_search_steps -= 1
                #         if sun_search_steps == 0:
                #             location_updated = False
                #             break
                # if sun_search_steps > 0:
                #     total_vertices.append([x_coord, y_coord])
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

    def plot_obstacles(self, show_flag=False, fill_color=None):
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
                rects.append(Rectangle(corner, width, height,color=fill_color))
            return rects

        if show_flag:
            print("SHOWING")
            plt.show()


def get_madrid_buildings():
    return Obstacles(obstacles_madrid_list)


if __name__ == '__main__':
    _rects = get_madrid_buildings().plot_obstacles(False, fill_color='gray')
    ax = plt.gca()
    for _rect in _rects:
        ax.add_patch(_rect)
    plt.plot(0, 0, c='blue', marker='s', label='MBS', linestyle='none')
    plt.plot(200.0, 370.0, c='green', marker='o', label='Hotspot center', linestyle='none')
    plt.legend(loc="upper left")
    plt.show()