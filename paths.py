from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import matplotlib.pyplot as plt
import obstacles


class Paths(object):
    def __init__(self, obstacles_obj=None):
        if not obstacles_obj:
            self.obstacles_obj = obstacles.get_madrid_buildings()
        self.paths_segments = []
        self.obstacles_vertices = np.asarray(self.obstacles_obj.get_total_vertices())
        self.vor = Voronoi(self.obstacles_vertices)
        self.all_vertices = self.vor.vertices.tolist()
        self.generate_paths()

    def generate_paths(self, x_boundary=None, y_boundary=None):

        if not x_boundary or not y_boundary:
            max_x_boundary, max_y_boundary = np.max(self.obstacles_vertices, 0) + 30
            min_x_boundary, min_y_boundary = np.min(self.obstacles_vertices, 0) - 30
        else:
            min_x_boundary, max_x_boundary = x_boundary
            min_y_boundary, max_y_boundary = y_boundary

        vor = self.vor
        center = vor.points.mean(axis=0)
        ptp_bound = vor.points.ptp(axis=0)
        infinite_segments = []
        finite_segments = []
        for point_idx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
            simplex = np.asarray(simplex)
            if np.all(simplex >= 0):
                if vor.vertices[simplex[0]][0] < min_x_boundary or vor.vertices[simplex[0]][0] > max_x_boundary or \
                        vor.vertices[simplex[0]][1] < min_y_boundary or vor.vertices[simplex[0]][1] > max_y_boundary or\
                        vor.vertices[simplex[1]][0] < min_x_boundary or vor.vertices[simplex[1]][0] > max_x_boundary or\
                        vor.vertices[simplex[1]][1] < min_y_boundary or vor.vertices[simplex[1]][1] > max_y_boundary:
                    continue
                finite_segments.append(vor.vertices[simplex])
            else:
                i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

                t = vor.points[point_idx[1]] - vor.points[point_idx[0]]  # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal

                midpoint = vor.points[point_idx].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                if vor.furthest_site:
                    direction = -direction
                far_point = vor.vertices[i] + direction * ptp_bound.max()
                if vor.vertices[i][0] < min_x_boundary or vor.vertices[i][0] > max_x_boundary or \
                        vor.vertices[i][1] < min_y_boundary or vor.vertices[i][1] > max_y_boundary:
                    continue
                far_point[0] = max_x_boundary if far_point[0] > max_x_boundary else far_point[0]
                far_point[1] = max_y_boundary if far_point[1] > max_y_boundary else far_point[1]

                far_point[0] = min_x_boundary if far_point[0] < min_x_boundary else far_point[0]
                far_point[1] = min_y_boundary if far_point[1] < min_y_boundary else far_point[1]

                infinite_segments.append([vor.vertices[i], far_point])

                if far_point.tolist() not in self.all_vertices:
                    self.all_vertices.append(far_point.tolist())

        self.paths_segments = infinite_segments + finite_segments
        # voronoi_plot_2d(vor)
        return self.paths_segments

    def get_segments(self):
        if not self.paths_segments:
            return self.generate_paths()
        else:
            return self.paths_segments

    def plot_segments(self, show_flag=False):
        for segment in self.paths_segments:
            xs, ys = zip(*segment)
            plt.plot(xs, ys)
        if show_flag:
            plt.show()


if __name__ == '__main__':
    madrid_obstacles = obstacles.get_madrid_buildings()
    vertices_array = np.array(madrid_obstacles.get_total_vertices())
    P = Paths()
    print(len(P.all_vertices))
    P.generate_paths()
    print(len(P.all_vertices))
    P.plot_segments()
    P.obstacles_obj.plot_obstacles(True)
