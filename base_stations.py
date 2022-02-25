import numpy as np
from resources.data_structures import Coords3D
from user_equipment import UserEquipment

from parameters import BS_DEFAULT_TRANSMISSION_POWER, BS_TOTAL_BANDWIDTH_DEFAULT, BS_EXPECTED_SPECTRAL_EFFICIENCY, \
    BS_HEIGHT, BS_STARTING_ENERGY, BS_COVERAGE_RADIUS, BS_MAX_ENERGY

from solar_user_mobility import ShadowedObstaclesMobility
from multiprocessing import Manager

from resources.math_utils import get_azimuth


class BaseStation:
    def __init__(self, station_id, coords=(0, 0, BS_HEIGHT), t_power=BS_DEFAULT_TRANSMISSION_POWER,
                 backhaul_bs_id=None):
        self.base_station_id = station_id
        self.coords = Coords3D(coords[0], coords[1], coords[2])
        self.t_power = t_power
        self.total_bandwidth = BS_TOTAL_BANDWIDTH_DEFAULT
        self.spectral_efficiency = BS_EXPECTED_SPECTRAL_EFFICIENCY
        self.energy_level = BS_STARTING_ENERGY
        self.n_served_users = None
        self.is_shadowed_flag = False
        self.is_serving_flag = True
        self.backhaul_bs_id = backhaul_bs_id
        self.coverage_radius = BS_COVERAGE_RADIUS
        self.recharge_count = 0

    def energy_empty(self):
        self.energy_level = BS_STARTING_ENERGY + self.energy_level
        self.is_serving_flag = True
        self.recharge_count += 1

    def update_energy(self, joules=0, wh=0):
        if wh != 0:
            self.energy_level += wh
        elif joules != 0:
            self.energy_level += joules / 3600

        self.energy_level = min(self.energy_level, BS_MAX_ENERGY)
        if self.energy_level <= 0:
            self.energy_empty()

    def update_location(self, new_x, new_y, energy_consumed=0):
        self.set_coords(new_x, new_y)
        self.energy_level -= energy_consumed
        if self.energy_level <= 0:
            self.energy_empty()

    def is_shadowed(self, shadow_grid, steps_size, shadow_xs, shadow_ys):
        if self.coords.x < shadow_xs[0] or self.coords.y < shadow_ys[0] or \
                self.coords.x > shadow_xs[-1] or self.coords.y > shadow_ys[-1]:
            self.is_shadowed_flag = False
            return self.is_shadowed_flag
            # raise ValueError('Base station outside boundary!')
        x_idx = int(np.round((self.coords.x - shadow_xs[0]) / steps_size[0]))
        y_idx = int(np.round((self.coords.y - shadow_ys[0]) / steps_size[1]))
        self.is_shadowed_flag = shadow_grid[x_idx, y_idx]
        return self.is_shadowed_flag

    def is_within_obstacle(self, obstacles_list):
        for _obstacle in obstacles_list:
            if _obstacle.is_overlapping(self.coords.x, self.coords.y, self.coords.z):
                return True
        return False

    def set_coords(self, new_x, new_y, new_z=None):
        if new_z is None:
            new_z = self.coords.z
        self.coords = Coords3D(new_x, new_y, new_z)

    def is_blocked(self, dest_coords, obstacles_list):
        if dest_coords.z >= self.coords.z:
            src_coords = self.coords
        else:
            src_coords, dest_coords = dest_coords, self.coords
        azimuth_to_dest = get_azimuth(src_coords.x, src_coords.y, dest_coords.x, dest_coords.y)
        max_distance = src_coords.get_distance_to(dest_coords)
        elevation_to_dest = np.degrees(np.arcsin((dest_coords.z - src_coords.z) / max_distance))
        for _obstacle in obstacles_list:
            if _obstacle.is_blocking(self.coords.x, self.coords.y, azimuth_to_dest, elevation_to_dest,
                                     reference_height=src_coords.z, max_distance=max_distance):
                return True
        return False


if __name__ == '__main__':
    n_base_stations = 5
    n_users = 50
    xy_min = [0, 0]
    xy_max = [500, 500]
    users = []
    users_equipment = []
    base_stations = []
    # for user_id in range(n_users):
    #     initial_coords = np.random.uniform(low=xy_min, high=xy_max, size=2)
    #     # print(initial_coords)
    #     users.append(User(user_id, initial_coords=initial_coords))
    #     users_equipment.append(UserEquipment(user_id, initial_coords=initial_coords))

    for base_id in range(n_base_stations):
        initial_coords = np.random.uniform(low=xy_min, high=xy_max, size=2)
        initial_coords = np.append(initial_coords, BS_HEIGHT)
        base_stations.append(BaseStation(base_id, initial_coords))

    [ue.set_available_base_stations(base_stations) for ue in users_equipment]
    [ue.get_received_sinr() for ue in users_equipment]
    [ue.associate_to_higesht_sinr() for ue in users_equipment]

    mobility_model = ShadowedObstaclesMobility()
    base_stations_shared_coords = Manager().list()
    for _bs in base_stations:
        base_stations_shared_coords.append([_bs.coords.x, _bs.coords.y])
    mobility_model.add_base_stations_to_plot(base_stations_shared_coords)
    mobility_model.generate_plot()
    shadow_xs, shadow_ys = mobility_model.get_grid_coordinates()
    grid_x_step, grid_y_step = mobility_model.get_grid_steps()
    for _user in mobility_model.users:
        users_equipment.append(UserEquipment(_user.id, initial_coords=_user.current_coords))

    step_count = 0
    n_of_steps = 6000
    while step_count <= n_of_steps:
        results = mobility_model.generate_model_step()
        for idx, _user_coords in enumerate(results[0]):
            users_equipment[idx].coords = _user_coords
        shadow_grid = results[1]
        for _bs in base_stations:
            _bs.is_shadowed(shadow_grid, [grid_x_step, grid_y_step], shadow_xs, shadow_ys)
            _bs.is_within_obstacle(mobility_model.get_obstacles())
