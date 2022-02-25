from resources import plotting
import user_mobility
from numpy import loadtxt
from multiprocessing import Process, Manager
import multiprocessing as mp
import obstacles
import ctypes
import numpy as np
from datetime import date, datetime, time, timedelta
from parameters import STARTING_SOLAR_HOUR, STARTING_SOLAR_MINUTE, BS_HEIGHT, NUM_OF_USERS, MAX_HOUR_DAY, TIME_STEP


from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from scipy.spatial import ConvexHull, convex_hull_plot_2d, Delaunay
from resources.math_utils import get_azimuth, get_distance, angle_in_range, to_radians, rotate, line_point_angle

# https://keisan.casio.com/exec/system/1224682277
AZIMUTH_OFFSET = 0
# solar angles per 15 minutes
ELEVATION_ANGLES = [-22.95, -24.58, -26.07, -27.4, -28.56, -29.54, -30.34, -30.94, -31.33, -31.53, -31.51, -31.29,
                    -30.86, -30.23, -29.41, -28.4, -27.21, -25.86, -24.35, -22.7, -20.91, -19.0, -16.98, -14.86, -12.64,
                    -10.34, -7.97, -5.49, -2.57, 0.21, 2.56, 5.13, 7.79, 10.5, 13.26, 16.04, 18.85, 21.68, 24.52, 27.37,
                    30.22, 33.06, 35.9, 38.73, 41.53, 44.3, 47.02, 49.7, 52.3, 54.81, 57.21, 59.47, 61.55, 63.4, 64.98,
                    66.22, 67.07, 67.49, 67.45, 66.95, 66.02, 64.71, 63.07, 61.17, 59.06, 56.77, 54.34, 51.81, 49.19,
                    46.5, 43.76, 40.99, 38.18, 35.35, 32.51, 29.66, 26.8, 23.95, 21.11, 18.28, 15.48, 12.69, 9.94, 7.23,
                    4.58, 2.04, -0.28, -3.3, -6.06, -8.52, -10.89, -13.18, -15.39, -17.5, -19.51, -21.41, -23.18]
AZIMUTH_ANGLES = [323.39, 326.85, 330.43, 334.14, 337.95, 341.87, 345.88, 349.96, 354.1, 358.27, 2.45, 6.62, 10.75,
                  14.83, 18.82, 22.73, 26.53, 30.21, 33.78, 37.22, 40.54, 43.75, 46.83, 49.8, 52.67, 55.44, 58.13,
                  60.73, 63.26, 65.73, 68.14, 70.51, 72.85, 75.16, 77.45, 79.74, 82.03, 84.34, 86.68, 89.06, 91.5,
                  94.02, 96.64, 99.37, 102.26, 105.33, 108.62, 112.17, 116.04, 120.3, 125.02, 130.29, 136.18, 142.78,
                  150.14, 158.24, 166.98, 176.17, 185.48, 194.6, 203.24, 211.21, 218.42, 224.88, 230.65, 235.8, 240.43,
                  244.61, 248.42, 251.92, 255.16, 258.19, 261.04, 263.75, 266.35, 268.85, 271.28, 273.65, 275.98,
                  278.28, 280.57, 282.86, 285.15, 287.46, 289.8, 292.17, 294.59, 297.07, 299.61, 302.23, 304.93, 307.72,
                  310.6, 313.6, 316.7, 319.93, 323.27]

AZIMUTH_ANGLES = [x + AZIMUTH_OFFSET for x in AZIMUTH_ANGLES]


class ShadowedObstaclesMobility(user_mobility.ObstaclesMobilityModel):
    def __init__(self, step_size=None, number_of_users=NUM_OF_USERS, shadow_grid_flag=True):
        super().__init__(number_of_users=number_of_users)
        self.shadow_grid_flag = shadow_grid_flag
        self.sun_azimuth = None
        self.sun_elevation = None
        self.last_solar_timestamp = 0
        self.update_sun = True
        self.shadows_update_flag = None
        self.shadow_xs = None
        self.shadow_ys = None
        self.shadow_grid = None
        self.shadows_state = None
        self.shadow_building_mask = None
        self.base_stations_coords = None
        self.solar_time_step = 15  # minutes
        self.max_time_of_day = None
        self.last_time_difference = 0
        self.last_time_step = 0
        self.seconds_buffer = 0
        self.starting_solar_hour, self.starting_solar_minute = STARTING_SOLAR_HOUR, STARTING_SOLAR_MINUTE
        self.time_of_day = datetime.combine(date(2022, 7, 1),
                                            time(hour=self.starting_solar_hour, minute=self.starting_solar_minute))

        self.set_max_time_of_day(MAX_HOUR_DAY, 59)
        if step_size is None:
            step_size = [5, 5]

        self.step_size = step_size

        self.setup_shadow_grid()

        self.auxiliary_updates = [self.increment_time, self.calculate_shadows, self.update_shadows_shared_array,
                                  self.trigger_solar_update]

    # noinspection PyArgumentList
    def setup_shadow_grid(self):
        if not self.shadow_grid_flag:
            return
        min_x = np.concatenate(self.graph.paths_segments, 0).min(0)[0]
        max_x = np.concatenate(self.graph.paths_segments, 0).max(0)[0]
        min_y = np.concatenate(self.graph.paths_segments, 0).min(0)[1]
        max_y = np.concatenate(self.graph.paths_segments, 0).max(0)[1]

        x_steps = int(np.ceil((max_x - min_x) / self.step_size[0]))
        y_steps = int(np.ceil((max_y - min_y) / self.step_size[1]))

        self.shadow_xs = np.linspace(min_x, min_x + x_steps * self.step_size[0], x_steps, endpoint=False)
        self.shadow_ys = np.linspace(min_y, min_y + y_steps * self.step_size[1], y_steps, endpoint=False)
        # self.shadow_xs, self.shadow_ys = np.meshgrid(xs, ys, sparse=True)

        self.shadow_grid = np.full((self.shadow_xs.size, self.shadow_ys.size), True, dtype=bool)
        # self.shadow_building_mask = np.full((self.shadow_xs.size, self.shadow_ys.size), False, dtype=bool)
        #
        # for idx_x in range(self.shadow_xs.size):
        #     for idx_y in range(self.shadow_ys.size):
        #         for _obstacle in self.obstacles_objects.obstaclesList:
        #             if _obstacle.is_overlapping(self.shadow_xs[idx_x], self.shadow_ys[idx_y]):
        #                 self.shadow_building_mask[idx_x, idx_y] = True

        self.shadows_state = None
        self.calculate_shadows()
        self.shadows_state = mp.Array(ctypes.c_bool, self.shadow_xs.size * self.shadow_ys.size)

    def trigger_solar_update(self):
        if self.update_sun or abs(self.current_time - self.last_solar_timestamp) >= self.solar_time_step * 60:
            self.last_time_difference = self.current_time - self.last_solar_timestamp
            self.current_time = 0
            self.last_solar_timestamp = self.current_time
            self.update_sun = True

    def reset(self):
        self.current_time = 0
        self.update_sun = True
        self.plot_flag = False
        self.reset_users()

    def increment_time(self, minutes_to_add=None):
        self.seconds_buffer += self.last_time_step
        if minutes_to_add is None and self.seconds_buffer > 60:
            minutes_to_add = int(self.seconds_buffer / 60)
            self.seconds_buffer = self.seconds_buffer % 60

            time_of_day = self.time_of_day + timedelta(minutes=minutes_to_add)
            self.time_of_day = time_of_day

            if time_of_day > self.max_time_of_day:
                self.seconds_buffer = 0
                self.time_of_day = datetime.combine(self.time_of_day.date() + timedelta(days=1),
                                                    time(hour=self.starting_solar_hour, minute=self.starting_solar_minute))
                self.set_max_time_of_day(MAX_HOUR_DAY, 59)

    def set_max_time_of_day(self, max_hour, max_minutes):
        if max_hour > self.time_of_day.time().hour:
            self.max_time_of_day = datetime.combine(self.time_of_day.date(), time(hour=max_hour, minute=max_minutes))
        else:
            self.max_time_of_day = datetime.combine(self.time_of_day.date() + timedelta(days=1),
                                                    time(hour=max_hour, minute=max_minutes))

    def set_time_of_day(self, hours, minutes):
        self.time_of_day = datetime.combine(self.time_of_day.date(),
                                            time(hour=hours, minute=minutes))

    def get_time_index(self):
        """Return index to access ELEVATION and AZIMUTH arrays based on current time."""
        idx = int(self.time_of_day.hour * 4 + self.time_of_day.minute / 15)
        return idx

    def update_shadows_shared_array(self):
        if not self.update_sun:
            return
        if self.plot_flag and self.shadow_grid_flag:
            shadows_state_array = to_numpy_array(self.shadows_state, self.shadow_ys.size)
            shadows_state_array[:] = self.shadow_grid.copy()
        self.update_sun = False
        if self.shadows_update_flag is not None:
            self.shadows_update_flag.set()

    def update_sun_angles(self):
        time_idx = self.get_time_index()
        self.sun_azimuth = AZIMUTH_ANGLES[time_idx]
        self.sun_elevation = ELEVATION_ANGLES[time_idx]

    def force_shadow_grid_calculations(self):
        prev_flag = self.shadow_grid_flag
        self.shadow_grid_flag = True
        self.update_sun = True
        self.setup_shadow_grid()
        self.calculate_shadows()
        self.shadow_grid_flag = prev_flag

    def calculate_shadows(self):
        self.update_sun_angles()
        if not self.update_sun or not self.shadow_grid_flag:
            return
        if self.sun_elevation < 0:
            self.shadow_grid = np.full((self.shadow_xs.size, self.shadow_ys.size), True, dtype=bool)
            return

        self.shadow_grid = np.full((self.shadow_xs.size, self.shadow_ys.size), False, dtype=bool)
        # shadow_coords = np.array(np.meshgrid(self.shadow_xs, self.shadow_ys)).T.reshape(-1, 2)
        # unshadowed_points = shadow_coords[self.shadow_grid.reshape(-1).__invert__()]
        #
        # for _obstacle in self.obstacles_objects.obstaclesList:
        #     shadow_length = (_obstacle.height - BS_HEIGHT) / np.tan(np.deg2rad(self.sun_elevation))
        #     if shadow_length <=0:
        #         continue
        #     p0p = line_point_angle(shadow_length, _obstacle.vertices[0], angle_x=self.sun_azimuth + 180, angle_y=None,
        #                            rounding=False)
        #     p1p = line_point_angle(shadow_length, _obstacle.vertices[1], angle_x=self.sun_azimuth + 180, angle_y=None,
        #                            rounding=False)
        #     p2p = line_point_angle(shadow_length, _obstacle.vertices[2], angle_x=self.sun_azimuth + 180, angle_y=None,
        #                            rounding=False)
        #     p3p = line_point_angle(shadow_length, _obstacle.vertices[3], angle_x=self.sun_azimuth + 180, angle_y=None,
        #                            rounding=False)
        #     ch = ConvexHull(self.vertices + [p0p, p1p, p2p, p3p])


        for idx_x in range(self.shadow_xs.size):
            for idx_y in range(self.shadow_ys.size):
                # if self.shadow_building_mask[idx_x, idx_y]:
                #     self.shadow_grid[idx_x, idx_y] = True
                if self.is_shadowed(self.shadow_xs[idx_x], self.shadow_ys[idx_y]):
                    self.shadow_grid[idx_x, idx_y] = True
                else:
                    self.shadow_grid[idx_x, idx_y] = False

        # for idx_x in range(self.shadow_xs.size):
        #     for idx_y in range(self.shadow_ys.size):
        #         if self.shadow_grid[idx_x, idx_y]:
        #             plt.plot(self.shadow_xs[idx_x], self.shadow_ys[idx_y], 'ko')
        #         else:
        #             plt.plot(self.shadow_xs[idx_x], self.shadow_ys[idx_y], 'yo')
        # obs = obstacles.get_madrid_buildings()
        # obs.plot_obstacles(True)
        # plt.show()

    def is_shadowed(self, x_coord, y_coord, z_coords=BS_HEIGHT):
        if self.sun_elevation <= 0:
            return True
        for _obstacle in self.obstacles_objects.obstaclesList:
            if _obstacle.is_blocking(x_coord, y_coord, self.sun_azimuth, self.sun_elevation,
                                     reference_height=z_coords) or _obstacle.is_overlapping(x_coord, y_coord, z_coords):
                return True
        return False

    def generate_plot(self, plot_sleep=user_mobility.TIME_SLEEP):
        """If called, an animation of users mobility will be shown with shadows"""
        self.plot_flag = True
        self.plot_sleep = plot_sleep
        manager = Manager()
        self.users_coords = manager.list()
        self.shadows_update_flag = manager.Event()
        for _user in self.users:
            self.users_coords.append([_user.current_coords.x, _user.current_coords.y])

        obstacles_list = obstacles.get_madrid_buildings().obstaclesList

        self.update_shadows_shared_array()

        self.plotter_process = Process(target=plotter_func,
                                       args=(self.graph.paths_segments, obstacles_list, self.users_coords,
                                             self.shadow_xs, self.shadow_ys, self.shadows_state,
                                             self.shadows_update_flag, self.base_stations_coords
                                             if self.base_stations_coords else None))
        self.plotter_process.start()

    def generate_model_step(self, time_step=None):
        if time_step is not None:
            self.last_time_step = time_step
        else:
            self.last_time_step = 0

        super().generate_model_step(time_step)
        return [_user.current_coords for _user in self.users], self.shadow_grid, self.update_sun

    def get_grid_coordinates(self):
        return self.shadow_xs, self.shadow_ys

    def get_grid_steps(self):
        return self.step_size


def plotter_func(path_segments, obstacles_list, users_coords, shadow_xs, shadow_ys, shadows_state, shadows_update_flag,
                 base_stations_coords=None):
    plotter = plotting.Plotter()
    background_objects = []
    for segment in path_segments:
        xs, ys = zip(*segment)
        background_objects.append([xs, ys])
    plotter.set_fixed_background(background_objects, color='b', width=0.5, style='dotted')

    background_objects = []
    for _building in obstacles_list:
        xs, ys = zip(*_building.vertices + [_building.vertices[0]])
        background_objects.append([xs, ys])
    plotter.set_fixed_background(background_objects, color='k', width=1, style='dashed')

    plotter.set_users(users_coords)
    plotter.set_shadows(shadow_xs, shadow_ys, shadows_state, shadows_update_flag)
    if base_stations_coords:
        plotter.set_base_stations(base_stations_coords)
    plotter.start_plotter()


def to_numpy_array(mp_arr, y_size):
    _array = np.frombuffer(mp_arr.get_obj(), dtype=bool)
    return np.reshape(_array, (-1, y_size))


if __name__ == '__main__':
    # lines = loadtxt("scratch2", comments="#", delimiter="\n", unpack=False)
    # _list = lines.tolist()

    mobility_model = ShadowedObstaclesMobility()
    # mobility_model.generate_plot()
    mobility_model.generate_model_step(100)
