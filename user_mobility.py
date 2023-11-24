import graph_tools
from resources.data_structures import Coords
from random import randint
import numpy as np
from resources import plotting
import obstacles
from multiprocessing import Process, Manager
from time import sleep
from parameters import USER_SPEED, PAUSE_INTERVAL, NUM_OF_USERS, TIME_STEP, TIME_SLEEP


class User:
    def __init__(self, user_id, starting_vertex=None, initial_coords=None):
        if starting_vertex:
            self.last_waypoint = starting_vertex.id
            self.current_coords = starting_vertex.coords.copy()
        elif initial_coords is not None:
            self.current_coords = Coords(initial_coords[0], initial_coords[1])
        else:
            raise ValueError('No coordinates were provided!')

        self.id = user_id
        self.waypoints = []
        self.speed = np.random.uniform(USER_SPEED[0], USER_SPEED[1])
        self.remaining_pause_time = 0

    def set_waypoints(self, waypoints, last_waypoint_id):
        assert (waypoints[0] == self.current_coords)
        waypoints.pop(0)
        self.waypoints = waypoints
        self.last_waypoint = last_waypoint_id

    def update_location(self, delta_t):
        """Update location towards next waypoint or pause. Returns True when pause
         is done or last waypoint reached"""
        if not self.remaining_pause_time:
            if self.waypoints:
                if self.current_coords.update(self.waypoints[0], delta_t * self.speed):
                    self.current_coords = self.waypoints[0].copy()
                    self.waypoints.pop(0)
                return False
            else:
                self.remaining_pause_time = randint(PAUSE_INTERVAL[0], PAUSE_INTERVAL[1])
        else:
            self.remaining_pause_time -= delta_t
            if self.remaining_pause_time <= 0:
                self.remaining_pause_time = 0

        if not self.remaining_pause_time:
            assert (not self.waypoints)
            return True
        return False


class ObstaclesMobilityModel:
    def __init__(self, time_step=TIME_STEP, number_of_users=NUM_OF_USERS, graph=None):
        if not graph:
            # Obtains default Madrid graph
            self.graph = graph_tools.get_graph_from_segments()
        self.obstacles_objects = self.graph.obstacles_objects
        self.graph_vertices = self.graph.get_vertices()
        self.number_of_vertices = len(self.graph_vertices)
        self.users = []
        self.plotter_func = plotter_func
        self.current_time = 0
        self.plot_flag = False
        self.users_coords = None
        self.plot_sleep = 0
        self.plotter_process = None
        self.time_step = time_step
        self.auxiliary_updates = []
        self.base_stations_coords = None
        self.n_users = number_of_users
        self.reset_users()

    def reset_users(self):
        self.users = []
        for user_id in range(self.n_users):
            start_vertex_id = randint(0, self.number_of_vertices - 1)
            end_vertex_id = randint(0, self.number_of_vertices - 1)
            _user = User(user_id, self.graph_vertices[start_vertex_id])
            path_waypoints = self.graph.get_path_from_to(start_vertex_id, end_vertex_id)
            _user.set_waypoints(path_waypoints.copy(), end_vertex_id)
            self.users.append(_user)

    def generate_plot(self, plot_sleep=TIME_SLEEP):
        """If called, an animation of users mobility will be shown"""
        self.plot_flag = True
        self.plot_sleep = plot_sleep
        manager = Manager()
        self.users_coords = manager.list()
        for _user in self.users:
            self.users_coords.append([_user.current_coords.x, _user.current_coords.y])
        obstacles_list = obstacles.get_madrid_buildings().obstaclesList
        self.plotter_process = Process(target=self.plotter_func,
                                       args=(self.graph.paths_segments, obstacles_list, self.users_coords,
                                             self.base_stations_coords if self.base_stations_coords else None))
        self.plotter_process.start()

    def update_plot_users_coords(self):
        for idx, _user in enumerate(self.users):
            self.users_coords[idx] = [_user.current_coords.x, _user.current_coords.y]

    def update_plot(self):
        self.update_plot_users_coords()
        if self.plot_sleep:
            sleep(self.plot_sleep)

    def update_users_locations(self):
        for _user in self.users:
            if _user.update_location(self.time_step):
                end_vertex_id = randint(0, self.number_of_vertices - 1)
                path_waypoints = self.graph.get_path_from_to(_user.last_waypoint,
                                                             end_vertex_id)
                _user.set_waypoints(path_waypoints, end_vertex_id)

    def generate_model(self, duration=None):
        while duration is None or self.current_time < duration:
            self.current_time += self.time_step
            self.update_users_locations()
            for _update in self.auxiliary_updates:
                _update()
            if self.plot_flag:
                self.update_plot()

        if self.plot_flag:
            self.plotter_process.join()

    def generate_model_step(self, time_step=None):
        if time_step is not None:
            self.time_step = time_step
        self.current_time += self.time_step
        self.update_users_locations()
        for _update in self.auxiliary_updates:
            _update()
        if self.plot_flag:
            self.update_plot()
        return [_user.current_coords for _user in self.users]

    def get_obstacles(self):
        return self.obstacles_objects.obstaclesList

    def add_base_stations_to_plot(self, base_stations_coords):
        self.base_stations_coords = base_stations_coords


def plotter_func(path_segments, obstacles_list, users_coords, base_stations_coords):
    plotter = plotting.Plotter()
    background_objects = []
    for segment in path_segments:
        xs, ys = zip(*segment)
        background_objects.append([xs, ys])
    plotter.set_fixed_background(background_objects, color='g', width=0.5, style='dotted')

    background_objects = []
    for _building in obstacles_list:
        xs, ys = zip(*_building.vertices + [_building.vertices[0]])
        background_objects.append([xs, ys])
    plotter.set_fixed_background(background_objects, color='k', width=1, style='dashed')

    plotter.set_users(users_coords)
    if base_stations_coords:
        plotter.set_base_stations(base_stations_coords)
    plotter.start_plotter()


if __name__ == '__main__':
    mobility_model = ObstaclesMobilityModel()
    mobility_model.generate_plot()
    mobility_model.generate_model()
