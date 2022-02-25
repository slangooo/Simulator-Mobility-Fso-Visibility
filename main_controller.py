import numpy.random

from solar_user_mobility import ShadowedObstaclesMobility
import numpy as np
from multiprocessing import Manager
from base_stations import BaseStation
from parameters import NUMBER_OF_BS, NUM_OF_USERS, BS_HEIGHT, TIME_STEP, MACRO_BS_LOCATION, STARTING_DAY, STARTING_MONTH
from user_equipment import UserEquipment
from resources.data_structures import Coords3D
from irradiation import PowerManager
import matplotlib.pyplot as plt
import time
from pysolar.solar import get_altitude, get_azimuth, get_position
from datetime import timezone
import types


class SimulationController:
    def __init__(self, plot_flag=False, renewable_energy_flag=True, shadow_grid_flag=True):
        self.n_of_bs = NUMBER_OF_BS
        self.n_of_users = NUM_OF_USERS
        self.users_equips = []
        self.base_stations = []
        self.mobility_model = None
        self.grid_xs, self.grid_ys = None, None
        self.shadow_grid = None
        self.min_xy, self.max_xy = None, None
        self.grid_steps = None
        self.solar_update_flag = False
        self.plot_flag = plot_flag
        self.base_stations_shared_coords = []
        self.users_coords = []
        self.power_manager = None
        self.constant_irradiation = None
        self.sin_solar_angle = None
        self.macro_bs = None
        self.skip_bs_shadow_calculations = False
        self.last_time_step = TIME_STEP
        self.renewable_energy_flag = renewable_energy_flag
        self.shadow_grid_flag = shadow_grid_flag

    def generate_environment_model(self, generate_random_bs=True):
        # initiate model
        self.initiate_mobility_model_and_grid()

        # Initiate base stations
        if not self.base_stations and generate_random_bs:
            self.generate_random_base_stations()

        if self.plot_flag:
            self.add_base_stations_to_plot()

        # Initiate users
        self.generate_ue_for_users()
        self.update_ue_available_base_stations()

        # Power manager
        self.init_power_manager()

        self.calculate_shadow_status_for_all_bs()

    def simulate_time_step(self, time_step=None):
        self.users_coords, self.shadow_grid, self.solar_update_flag = self.mobility_model.generate_model_step(time_step)
        self.last_time_step = time_step if time_step is not None else TIME_STEP
        if self.solar_update_flag:
            self.update_irradiation_parameters()
        if self.mobility_model.n_users:
            self.update_ue_locations_from_mobility_model()

        if self.base_stations:
            self.calculate_shadow_status_for_all_bs()
            self.update_base_stations_energy()
            self.update_ue_available_base_stations()
            if self.mobility_model.n_users:
                self.calculate_sinr_for_ues()
                self.associate_ues_to_bs_highest_sinr()

    def set_bs_locations(self, locations_array):
        for _bs, new_location in zip(self.base_stations, locations_array):
            _bs.set_coords(new_location[0], new_location[1], (new_location[2] if new_location.size > 2 else None))
        if self.plot_flag:
            self.update_bs_shared_coords()

    def update_ue_locations_from_mobility_model(self):
        for idx, _user_coords in enumerate(self.users_coords):
            self.users_equips[idx].coords = _user_coords

    def generate_random_base_stations(self):
        self.base_stations_shared_coords = []
        for base_id in range(self.n_of_bs):
            initial_coords = np.random.uniform(low=self.min_xy, high=self.max_xy, size=2)
            initial_coords = np.append(initial_coords, BS_HEIGHT)
            self.base_stations.append(BaseStation(base_id, initial_coords))
            while self.base_stations[-1].is_within_obstacle(self.mobility_model.get_obstacles()):
                initial_coords = np.random.uniform(low=self.min_xy, high=self.max_xy, size=2)
                initial_coords = np.append(initial_coords, BS_HEIGHT)
                self.base_stations[-1].coords = Coords3D(initial_coords[0], initial_coords[1], initial_coords[2])

    def generate_random_bs_destination(self):
        initial_coords = np.random.uniform(low=self.min_xy, high=self.max_xy, size=2)
        initial_coords = np.append(initial_coords, 0)  # 0 height
        _bs = BaseStation(-3, initial_coords)
        while _bs.is_within_obstacle(self.mobility_model.get_obstacles()):
            initial_coords = np.random.uniform(low=self.min_xy, high=self.max_xy, size=2)
            initial_coords = np.append(initial_coords, BS_HEIGHT)
            _bs.coords = Coords3D(initial_coords[0], initial_coords[1], initial_coords[2])

        destination = _bs.coords.copy()
        del _bs
        # return destination
        return Coords3D(200, 350, BS_HEIGHT)

    def check_bs_obstacle_collision(self, base_station):
        return base_station.is_within_obstacle(self.mobility_model.get_obstacles())

    def add_base_stations_to_plot(self, generate_plot_flag=False):
        self.base_stations_shared_coords = Manager().list()
        for _bs in self.base_stations:
            self.base_stations_shared_coords.append([_bs.coords.x, _bs.coords.y])
        self.mobility_model.add_base_stations_to_plot(self.base_stations_shared_coords)
        if generate_plot_flag:
            self.mobility_model.generate_plot()

    def reset_model(self):
        for _bs in self.base_stations:
            del _bs
        self.base_stations = []
        self.mobility_model.reset()
        self.simulate_time_step()
        self.mobility_model.calculate_shadows()
        self.calculate_shadow_status_for_all_bs()
        self.set_macro_bs(MACRO_BS_LOCATION[0], MACRO_BS_LOCATION[1])

    def generate_mobility_plot(self):
        self.add_base_stations_to_plot()
        self.mobility_model.generate_plot()

    def update_bs_shared_coords(self):
        for idx, _bs in enumerate(self.base_stations):
            self.base_stations_shared_coords[idx] = [_bs.coords.x, _bs.coords.y]

    def initiate_mobility_model_and_grid(self):
        self.mobility_model = ShadowedObstaclesMobility(number_of_users=self.n_of_users,
                                                        shadow_grid_flag=self.shadow_grid_flag)


        ############
        #TODO: Refactor and clean after implementing irradiation
        def update_sun_angles(self):
            date = self.time_of_day.replace(tzinfo=timezone.utc, microsecond=0)
            self.sun_azimuth, self.sun_elevation = np.round(get_position(40.418725331325, -3.704271435627907, date,
                                                                elevation=820 + BS_HEIGHT), decimals=2)

        self.mobility_model.update_sun_angles = types.MethodType(update_sun_angles, self.mobility_model)
        ##########


        if not self.shadow_grid_flag:
            return
        self.grid_xs, self.grid_ys = self.mobility_model.get_grid_coordinates()
        self.min_xy = [self.grid_xs.min(), self.grid_ys.min()]
        self.max_xy = [self.grid_xs.max(), self.grid_ys.max()]
        self.grid_steps = self.mobility_model.get_grid_steps()
    def generate_ue_for_users(self):
        for _user in self.mobility_model.users:
            self.users_equips.append(UserEquipment(_user.id, initial_coords=_user.current_coords))

    def update_ue_available_base_stations(self):
        [ue.set_available_base_stations(self.base_stations) for ue in self.users_equips]

    def calculate_shadow_status_for_all_bs(self):
        if not self.skip_bs_shadow_calculations:
            for _bs in self.base_stations:
                _bs.is_shadowed_flag = self.mobility_model.is_shadowed(_bs.coords.x, _bs.coords.y, _bs.coords.z)
                # _bs.is_shadowed(self.shadow_grid, self.grid_steps, self.grid_xs, self.grid_ys)

    def calculate_sinr_for_ues(self):
        [ue.get_received_sinr() for ue in self.users_equips]

    def associate_ues_to_bs_highest_sinr(self):
        ue_associations = [ue.associate_to_higesht_sinr() for ue in self.users_equips]
        for _bs in self.base_stations:
            _bs.n_served_users = ue_associations.count(_bs.base_station_id)

    def get_sinrs(self):
        return np.asarray([ue.received_sinr for ue in self.users_equips])

    def get_snrs(self):
        return np.asarray([ue.received_snr for ue in self.users_equips])

    def get_associations(self):
        return np.asarray([ue.serving_bs_id for ue in self.users_equips])

    def get_bs_locations(self):
        return np.asarray([_bs.coords.np_array() for _bs in self.base_stations])

    def get_date_time(self):
        return self.mobility_model.time_of_day

    def init_power_manager(self):
        # TODO: own implementation
        self.power_manager = PowerManager()
        self.power_manager.set_manager_default_values()
        self.update_irradiation_parameters()

    def update_irradiation_parameters(self):
        # TODO: own implementation
        _date = self.get_date_time()
        _date = _date.replace(day=STARTING_DAY, month=STARTING_MONTH)
        self.constant_irradiation = self.power_manager.caluclate_constant_solar_irradiation_power(_date)
        self.sin_solar_angle = self.power_manager.calculate_sin_solar_altitude(_date)

        # current_date = _date.replace(tzinfo=timezone.utc, microsecond=0)
        # sun_azimuth, sun_elevation = get_position(self.power_manager.coordinates[1], self.power_manager.coordinates[0], current_date,
        #                                           elevation=self.power_manager.terrain_height + self.power_manager.UAV_height)



    def calculate_energy_level(self, base_station):
        irradiation_magnitude = (1 - numpy.random.uniform(0.0, 0.2)) * \
                                self.constant_irradiation
        if self.renewable_energy_flag:
            produced_energy = self.power_manager.calculate_energy_produced(self.last_time_step,
                                                                           irradiation_magnitude, self.sin_solar_angle)
        else:
            produced_energy = 0

        consumed_energy = self.power_manager.calculate_energy_consumption(self.last_time_step, base_station.t_power)
        income_energy_joules = (produced_energy if not base_station.is_shadowed_flag else 0) - consumed_energy

        base_station.update_energy(joules=income_energy_joules)

    def update_base_stations_energy(self):
        [self.calculate_energy_level(_bs) for _bs in self.base_stations]

    def set_macro_bs(self, macro_x, macro_y, macro_z=BS_HEIGHT):
        self.macro_bs = BaseStation(-1, coords=(macro_x, macro_y, macro_z), backhaul_bs_id=-2)
        return self.macro_bs

    def get_bs_with_id(self, target_id):
        if target_id == -1:
            return self.macro_bs, -1
        for idx, _bs in enumerate(self.base_stations):
            if _bs.base_station_id == target_id:
                return _bs, idx
        return None, None

    def clear_bs(self):
        self.base_stations = []
        self.update_ue_available_base_stations()

    def add_bs_station(self, bs_x, bs_y, bs_z=BS_HEIGHT, backhaul_bs_id=None, check_if_shadowed=False):
        if self.base_stations:
            base_id = self.base_stations[-1].base_station_id + 1
        else:
            base_id = 0
        base_station = BaseStation(base_id, coords=(bs_x, bs_y, bs_z), backhaul_bs_id=backhaul_bs_id)
        if check_if_shadowed:
            base_station.is_shadowed(self.shadow_grid, self.grid_steps, self.grid_xs, self.grid_ys)

        self.base_stations.append(base_station)
        self.update_ue_available_base_stations()
        if self.plot_flag:
            self.add_base_stations_to_plot()
        return base_station.base_station_id

    def is_backhaul_blocked_with_id(self, base_station_id):
        bs, _ = self.get_bs_with_id(base_station_id)
        backhaul_bs, _ = self.get_bs_with_id(bs.backhaul_bs_id)
        return bs.is_blocked(backhaul_bs.coords, self.mobility_model.get_obstacles())

    def is_backhaul_blocked(self, base_station):
        backhaul_bs, _ = self.get_bs_with_id(base_station.backhaul_bs_id)
        return base_station.is_blocked(backhaul_bs.coords, self.mobility_model.get_obstacles())

    def get_last_bs_id(self):
        if not self.base_stations:
            return -1
        else:
            return self.base_stations[-1].base_station_id


if __name__ == '__main__':
    t1 = time.time()
    _controller = SimulationController()
    _controller.generate_environment_model(generate_random_bs=False)
    _controller.simulate_time_step(None)
    sunny_vertices = _controller.mobility_model.graph.obstacles_objects.get_sunny_vertices \
        (_controller.mobility_model.sun_azimuth, _controller.mobility_model.sun_elevation)
    last_id = 0
    # for _vertex in sunny_vertices:
    #     last_id = _controller.add_bs_station(_vertex[0], _vertex[1], backhaul_bs_id=last_id)

    xs, ys = np.meshgrid(_controller.mobility_model.shadow_xs, _controller.mobility_model.shadow_ys)


    def shadow_color(flag):
        if flag:
            return 'black'
        else:
            return 'yellow'


    vf = np.vectorize(shadow_color)

    shadow_colors = np.array(list(map(vf, _controller.mobility_model.shadow_grid))).flatten()

    _controller.mobility_model.graph.obstacles_objects.plot_obstacles(False)
    plt.scatter(xs.transpose(), ys.transpose(), s=0.5, alpha=1, linewidths=0.1, c=shadow_colors)

    # for _vertex in sunny_vertices:
    #     plt.scatter(_vertex[0], _vertex[1], c='black')

    plt.show()
    print(time.time() - t1)
    # _controller.generate_mobility_plot()

    # _controller.set_macro_bs(0, 400)
    # last_id = _controller.add_bs_station(0, 0, backhaul_bs_id=-1)
    # last_id = _controller.add_bs_station(100, 100, backhaul_bs_id=last_id)
    # last_id = _controller.add_bs_station(130, 100, backhaul_bs_id=last_id)
    # last_id = _controller.add_bs_station(135, 100, backhaul_bs_id=last_id)
    # _controller.generate_mobility_plot()
    # _controller.get_date_time()
    # _controller.simulate_time_step(None)
    # _controller.simulate_time_step(None)
    # _controller.simulate_time_step(None)
    # _controller.add_bs_station(0, 0)
    # _controller.is_backhaul_blocked(0)
    # _controller.is_backhaul_blocked(3)
    # [_bs.energy_level for _bs in _controller.base_stations]
    # [_bs.is_shadowed_flag for _bs in _controller.base_stations]

    # new_locations = np.vstack((np.random.uniform(low=_controller.min_xy, high=_controller.max_xy,
    #                                              size=(len(_controller.base_stations),2)).T,
    #                            np.zeros(len(_controller.base_stations)))).T
    #
    # new_locations = np.random.uniform(low=_controller.min_xy, high=_controller.max_xy,
    #                                              size=(len(_controller.base_stations),2))
    #
    # _controller.set_bs_locations(new_locations)
    # print(_controller.base_stations_shared_coords)
