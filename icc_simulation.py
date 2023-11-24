from main_controller import SimulationController
from visibility_graph import Point, VisGraph, Edge
import numpy as np
import matplotlib.pyplot as plt
from parameters import STARTING_SOLAR_HOUR, MAX_HOUR_DAY, STARTING_DAY, STARTING_SOLAR_MINUTE, STARTING_MONTH, BS_HEIGHT
from solar_user_mobility import ELEVATION_ANGLES
from pysolar.solar import get_altitude
from datetime import datetime, timedelta, timezone
from itertools import chain as iter_chain


class IccSimulator:
    def __init__(self, renewable_energy_flag):
        self._controller = SimulationController(renewable_energy_flag=renewable_energy_flag, shadow_grid_flag=False)
        self._controller.generate_environment_model(generate_random_bs=False)
        self._controller.simulate_time_step(None)
        self.current_day = self._controller.get_date_time().date()
        self.shadow_color_vectorized = np.vectorize(shadow_color)
        self.sunny_vertices = []
        self.selected_vertices = []
        self.shadow_colors = []
        self.obstacles_polys = []
        self.valuable_edges = []
        self.graph = None
        self.shortest_path = None
        self.src = None #[0.0, 0.0]
        self.dest = None #[250.0, 380.0]
        # self.dest = [200.0, 370.0]
        self.generate_obstacles_polygons()
        self.n_return = 0
        self.n_arrival = 0
        self.dumb_returns = 0
        self.returns_per_hour = np.zeros((24, 1), dtype=np.float32)
        self.arrivals_per_hour = np.zeros((24, 1), dtype=np.float32)
        self.returns_per_hour_final = np.zeros((24, 1), dtype=np.float32)
        self.arrivals_per_hour_final = np.zeros((24, 1), dtype=np.float32)
        self.average_energy_per_hour = np.zeros((24, 1), dtype=np.float32)
        self.average_energy_per_hour_final = np.zeros((24, 1), dtype=np.float32)
        self.n_bs_per_hour = np.zeros((24, 1), dtype=np.float32)
        self.n_bs_per_hour_final = np.zeros((24, 1), dtype=np.float32)
        self.n_days = 0
        self.initial_step_flag = True
        self.initial_results_flag = True
        self.renewable_energy_flag = renewable_energy_flag
        self.update_total_vertices()
        # self.build_graph()

    def simulate_step(self, t_step=1):
        self._controller.simulate_time_step(t_step)
        if self.renewable_energy_flag and self._controller.solar_update_flag or self.initial_step_flag:
            self.build_graph()
            n_arrival, n_return = simulator.update_bs_locations()
            if self.initial_step_flag:
                n_arrival, n_return = 0, 0
            self.n_arrival += n_arrival
            self.n_return += n_return
            self.dumb_returns += n_return
            self.initial_step_flag = False
        for _bs in self._controller.base_stations:
            if _bs.recharge_count > 0:
                self.n_arrival += _bs.recharge_count
                self.n_return += _bs.recharge_count
                _bs.recharge_count = 0

        _hour = self._controller.get_date_time().time().hour
        self.returns_per_hour[_hour] += self.n_return
        self.arrivals_per_hour[_hour] += self.n_arrival
        self.n_bs_per_hour[_hour] = len(self._controller.base_stations)
        self.average_energy_per_hour[_hour] = sum([_bs.energy_level for _bs in self._controller.base_stations]) / \
                                              self.n_bs_per_hour[_hour]
        self.n_arrival = 0
        self.n_return = 0

        if self._controller.get_date_time().date() != self.current_day:
            self.initial_results_flag = False
            self.n_days += 1
            self.returns_per_hour_final = self.returns_per_hour_final + (self.returns_per_hour -
                                                                         self.returns_per_hour_final) / self.n_days
            self.arrivals_per_hour_final = self.arrivals_per_hour_final + (self.arrivals_per_hour -
                                                                           self.arrivals_per_hour_final) / self.n_days
            self.n_bs_per_hour_final = self.n_bs_per_hour_final + (self.n_bs_per_hour - self.n_bs_per_hour_final
                                                                   ) / self.n_days
            self.average_energy_per_hour_final = self.average_energy_per_hour_final + \
                                                 (self.average_energy_per_hour - self.average_energy_per_hour_final) \
                                                 / self.n_days

            self.returns_per_hour = np.zeros((24, 1), dtype=np.float32)
            self.arrivals_per_hour = np.zeros((24, 1), dtype=np.float32)
            self.average_energy_per_hour = np.zeros((24, 1), dtype=np.float32)
            self.n_bs_per_hour = np.zeros((24, 1), dtype=np.float32)
            self.current_day = simulator._controller.get_date_time().date()

        if self.initial_results_flag:
            np.copyto(self.returns_per_hour_final, self.returns_per_hour)
            np.copyto(self.arrivals_per_hour_final, self.arrivals_per_hour)
            np.copyto(self.n_bs_per_hour_final, self.n_bs_per_hour)
            np.copyto(self.average_energy_per_hour_final, self.average_energy_per_hour)

    def calculate_selected_vertices(self):
        if self.renewable_energy_flag:
            self.sunny_vertices = self._controller.mobility_model.graph.obstacles_objects.get_sunny_vertices \
                (self._controller.mobility_model.sun_azimuth, self._controller.mobility_model.sun_elevation)
        self.selected_vertices = []
        for _vertex in self.sunny_vertices:
            self.selected_vertices.append(Point(_vertex[0], _vertex[1]))
        if self.dest:
            self.selected_vertices.append(Point(self.dest[0],self.dest[1]))
        if self.src:
            self.selected_vertices.append(Point(self.src[0],self.src[1]))
        self.update_total_vertices()
        return self.selected_vertices

    def set_source_and_dest(self, src=None, dest=None):
        if src is not None:
            self.src = src
        if dest is not None:
            self.dest = dest
        self.calculate_selected_vertices()

    def update_total_vertices(self):
        self.total_vertices = [self.selected_vertices] + self.obstacles_polys
        self.total_vertices = [item for sublist in self.total_vertices for item in sublist]
        self.n_vertices = len(self.total_vertices)

    def generate_obstacles_polygons(self):
        polys = []
        for _obstacle in self._controller.mobility_model.get_obstacles():
            poly = []
            for _vertex in _obstacle.vertices:
                poly.append(Point(_vertex[0], _vertex[1]))
            polys.append(poly)
        self.obstacles_polys = polys

    def build_graph(self):
        self.calculate_selected_vertices()
        input_polys = self.obstacles_polys + [[_vertex] for _vertex in self.selected_vertices]
        self.graph = VisGraph()
        self.graph.build(input_polys, [])
        self.valuable_edges = []
        for _edge in iter(self.graph.visgraph.get_edges()):
            if _edge.p1 in self.selected_vertices and _edge.p2 in self.selected_vertices:
                self.valuable_edges.append(_edge)
        self.get_shortest_sunny_path()

    def get_shortest_sunny_path(self):
        if not self.dest:
            return None

        if self.renewable_energy_flag:
            self.shortest_path = self.graph.shortest_path \
                (Point(self.src[0], self.src[1]), Point(self.dest[0], self.dest[1]), self.selected_vertices)
        else:
            self.shortest_path = self.graph.shortest_path \
                (Point(self.src[0], self.src[1]), Point(self.dest[0], self.dest[1]), [])

        self.shortest_path = self.shortest_path[1:-1]
        return self.shortest_path

    def init_stations_in_shortest_route(self):
        last_id = self._controller.get_last_bs_id()
        for _location in self.shortest_path:
            last_id = self._controller.add_bs_station(_location.x, _location.y, backhaul_bs_id=last_id)

    def update_bs_locations(self):
        """Returns number of trip flights (redundant drone returning or new drone needed coming)"""
        current_bs_locations = [Point(_bs.coords.x, _bs.coords.y) for _bs in self._controller.base_stations]
        locations_to_satisfy = self.shortest_path.copy()

        bs_update_indexes = [i for i, x in enumerate(current_bs_locations) if x not in locations_to_satisfy]
        locations_to_satisfy = [_loc for _loc in locations_to_satisfy if _loc not in current_bs_locations]

        n_return = 0
        n_new = 0

        for i in range(len(bs_update_indexes) - len(locations_to_satisfy)):
            self._controller.base_stations.pop(bs_update_indexes.pop(-1))
            n_return += 1

        for _loc in locations_to_satisfy:
            if bs_update_indexes:
                self._controller.base_stations[bs_update_indexes.pop(0)].update_location(_loc.x, _loc.y)
            else:
                n_new += 1
                self._controller.add_bs_station(_loc.x, _loc.y, backhaul_bs_id=0)

        return n_new, n_return

    def get_shadow_meshgrid(self):
        xs, ys = np.meshgrid(self._controller.mobility_model.shadow_xs, self._controller.mobility_model.shadow_ys)
        return xs, ys

    def plot_shadows(self):
        if self._controller.mobility_model.shadow_grid is None:
            self._controller.mobility_model.force_shadow_grid_calculations()

        self.shadow_colors = np.array(
            list(map(self.shadow_color_vectorized, self._controller.mobility_model.shadow_grid))).flatten()
        shadow_xs, shadow_ys = self.get_shadow_meshgrid()
        plt.scatter(shadow_xs.transpose(), shadow_ys.transpose(), s=30, alpha=0.5, linewidths=0.1, c=self.shadow_colors)

    def plot_sunny_vertices(self):
        for idx, _vertex in enumerate(self.sunny_vertices):
            if not idx:
                plt.plot(_vertex[0], _vertex[1], c='red', marker='*', label='Nasłoneczniony punkt', linestyle='none')
            else:
                plt.plot(_vertex[0], _vertex[1], c='red', marker='*')

    def plot_obstacles(self):
        return self._controller.mobility_model.graph.obstacles_objects.plot_obstacles(False)

    def plot_valuable_edges(self):
        for _edge in self.valuable_edges:
            xs = [_edge.p1.x, _edge.p2.x]
            ys = [_edge.p1.y, _edge.p2.y]
            plt.plot(xs, ys)

    def plot_shortest_path(self):
        for _hop in self.shortest_path:
            plt.plot(_hop.x, _hop.y, c='red', marker='*', markeredgecolor='black', markersize=10)
        plt.plot([], [], linestyle='none', c='red', marker='*', markeredgecolor='black', markersize=10, label='DRS')
        path = [Point(self.src[0], self.src[1])] + self.shortest_path + [Point(self.dest[0], self.dest[1])]
        for idx in range(len(path) - 1):
            xs = [path[idx].x, path[idx + 1].x]
            ys = [path[idx].y, path[idx + 1].y]
            if not idx:
                plt.plot(xs, ys, c='red', label='mmWave link')
            else:
                plt.plot(xs, ys, c='red')

    def plot_default_vertices(self):
        for idx, _vertex in enumerate(self._controller.mobility_model.graph.obstacles_objects.get_total_vertices()):
            plt.plot(_vertex[0], _vertex[1], c='red', marker='.')

    def plot_endpoints(self):
        # plt.scatter(self.src[0], self.src[1], c='blue', marker='s')
        # plt.scatter(self.dest[0], self.dest[1], c='blue', marker='s')

        plt.plot(self.src[0], self.src[1], c='blue', marker='s', linestyle='none', label='MBS')
        if self.dest:
            plt.plot(self.dest[0], self.dest[1], c='green', marker='o', linestyle='none', label='Hotspot')

    def plot_results(self, rects_patches=None):
        plt.clf()
        rects_patches = self.plot_obstacles()
        # self.plot_shadows()
        self.plot_sunny_vertices()
        self.plot_shortest_path()
        self.plot_endpoints()
        # self.plot_valuable_edges()
        ax = plt.gca()
        if rects_patches:
            ax = plt.gca()
            for _patch in rects_patches:
                ax.add_patch(_patch)
        handles, labels = ax.get_legend_handles_labels()

        def flip(items, ncol):
            return iter_chain(*[items[i::ncol] for i in range(ncol)])

        plt.legend(flip(handles, 2), flip(labels, 2), loc='upper left', ncol=2)
        # plt.legend(loc="upper left")
        # plt.savefig('plots/night.eps', format='eps')
        plt.show()

    def print_energy_levels(self):
        [print(_bs.energy_level) for _bs in self._controller.base_stations]


def shadow_color(flag):
    if flag:
        return 'black'
    else:
        return 'yellow'


if __name__ == '__main__':
    n_days_to_simulate = 200
    t_step = 60 * 15

    simulator = IccSimulator(renewable_energy_flag=False)
    simulator.set_source_and_dest([600, 900])
    # simulator.build_graph()
    # simulator.plot_results()

    simulator.plot_endpoints()
    simulator.plot_default_vertices()
    rects_patches =simulator.plot_obstacles()
    ax = plt.gca()
    if rects_patches:
        ax = plt.gca()
        for _patch in rects_patches:
            ax.add_patch(_patch)
    handles, labels = ax.get_legend_handles_labels()
    def flip(items, ncol):
        return iter_chain(*[items[i::ncol] for i in range(ncol)])
    plt.legend(flip(handles, 2), flip(labels, 2), loc='upper left', ncol=2)
    plt.show()

    # # simulator.build_graph()
    # # simulator.init_stations_in_shortest_route()
    # # simulator.plot_results()
    # while (simulator.n_days < n_days_to_simulate):
    #     simulator.simulate_step(t_step)
    #     # print(simulator.arrivals_per_hour.transpose())
    #     # print(simulator.returns_per_hour.transpose())
    #     # print(simulator.arrivals_per_hour_final.transpose())
    #     # print(simulator.returns_per_hour_final.transpose())
    #     print(simulator.n_days)
    #     print(simulator._controller.get_date_time())
    #     simulator.print_energy_levels()
    # simulator_with_energy = simulator
    #
    # np.savetxt('results/arrivals_per_hour.txt', simulator.arrivals_per_hour_final)
    # np.savetxt('results/returns_per_hour.txt', simulator.returns_per_hour_final)
    # np.savetxt('results/energy_per_hour.txt', simulator.average_energy_per_hour_final)
    # np.savetxt('results/n_bs_per_hour.txt', simulator.n_bs_per_hour_final)

    # # NO SOLAR PANELS
    # simulator = IccSimulator(renewable_energy_flag=False)
    # while simulator.n_days < n_days_to_simulate:
    #     simulator.simulate_step(t_step)
    #     print(simulator.n_days)
    #     print(simulator._controller.get_date_time())
    #     simulator.print_energy_levels()
    #
    # np.savetxt('results/arrivals_per_hour_no_solar.txt', simulator.arrivals_per_hour_final)
    # np.savetxt('results/returns_per_hour_no_solar.txt', simulator.returns_per_hour_final)
    # np.savetxt('results/energy_per_hour_no_solar.txt', simulator.average_energy_per_hour_final)
    # np.savetxt('results/n_bs_per_hour_no_solar.txt', simulator.n_bs_per_hour_final)
    #
    # [Point(_bs.coords.x, _bs.coords.y) for _bs in simulator._controller.base_stations]
    # simulator.update_bs_locations()
    # simulator._controller.clear_bs()
    # [print(_bs.energy_level) for _bs in simulator._controller.base_stations]

    # # #####results plotting
    #
    # arrivals_per_hour_energy = np.loadtxt(
    #     "C:\\Users\\user\\PycharmProjects\\obstacleMobilityModel\\results\\arrivals_per_hour.txt")
    # arrivals_per_hour = np.loadtxt(
    #     "C:\\Users\\user\\PycharmProjects\\obstacleMobilityModel\\results\\arrivals_per_hour_no_solar.txt")
    #
    # returns_per_hour_energy = np.loadtxt(
    #     "C:\\Users\\user\\PycharmProjects\\obstacleMobilityModel\\results\\returns_per_hour.txt")
    # returns_per_hour = np.loadtxt(
    #     "C:\\Users\\user\\PycharmProjects\\obstacleMobilityModel\\results\\returns_per_hour_no_solar.txt")
    #
    # n_bs_per_hour_energy = np.loadtxt(
    #     "C:\\Users\\user\\PycharmProjects\\obstacleMobilityModel\\results\\n_bs_per_hour.txt")
    # n_bs_per_hour = np.loadtxt(
    #     "C:\\Users\\user\\PycharmProjects\\obstacleMobilityModel\\results\\n_bs_per_hour_no_solar.txt")
    #
    # avg_energy_per_hour = np.loadtxt('C:\\Users\\user\\PycharmProjects\\obstacleMobilityModel\\results\\energy_per_hour_no_solar.txt')
    # avg_energy_per_hour_energy = np.loadtxt('C:\\Users\\user\\PycharmProjects\\obstacleMobilityModel\\results\\energy_per_hour.txt')
    #
    # elev_angles = []
    # date = datetime(year=2022, day=STARTING_DAY, minute=0, month=STARTING_MONTH, hour=00, tzinfo=timezone.utc)
    # for i in range(24):
    #     elev_angles.append(max(0, np.round(get_altitude(40.418725331325, -3.704271435627907, date,
    #                                                     elevation=820 + BS_HEIGHT), decimals=2)))
    #     date = date + timedelta(hours=1)
    #
    # # plt.plot(simulator.arrivals_per_hour_final)
    # # plt.plot(simulator_with_energy.arrivals_per_hour_final)
    #
    # a = np.asarray([i for i in range(24)], dtype=np.int32)
    # fig, ax = plt.subplots(figsize=(10, 4))
    # ax2 = ax.twinx()
    # width = 0.2
    # ax.bar(a.transpose(), arrivals_per_hour.flatten(), color='r', width=width)
    # ax.bar(a.transpose() + width, arrivals_per_hour_energy.flatten(), color='y', width=width)
    # ax2.plot(elev_angles, color='b')
    # ax.set_xticks(a)
    # ax.set_xticklabels(a, rotation=65)
    # ax.legend(['Without RES', 'With RES'], loc="lower left", prop={'size': 15})
    # ax.set_xlabel('hour', fontsize=15)
    # ax.set_ylabel('Number of new DBS arrivals', fontsize=15)
    # ax.tick_params(axis='both', which='major', labelsize=12)
    # ax2.tick_params(axis='both', which='major', labelsize=12)
    # ax2.set_ylabel('Solar elevation angle', color='b', fontsize=15)
    # fig.tight_layout()
    # plt.show()
    # # plt.savefig('C:\\Users\\user\\PycharmProjects\\obstacleMobilityModel\\plots\\arrivals.eps', format='eps')
    # a = np.asarray([i for i in range(24)], dtype=np.int32)
    # fig, ax = plt.subplots(figsize=(10, 4))
    # ax2 = ax.twinx()
    # width = 0.2
    # ax.bar(a.transpose(), returns_per_hour.flatten(), color='r', width=width)
    # ax.bar(a.transpose() + width, returns_per_hour_energy.flatten(), color='y', width=width)
    # ax2.plot(elev_angles, color='b')
    # ax.set_xticks(a)
    # ax.set_xticklabels(a, rotation=65)
    # ax.legend(['Without RES', 'With RES'], loc="lower left", prop={'size': 15})
    # ax.set_xlabel('hour', fontsize=15)
    # ax.set_ylabel('Number of DBS returns', fontsize=15)
    # ax.tick_params(axis='both', which='major', labelsize=12)
    # ax2.tick_params(axis='both', which='major', labelsize=12)
    # ax2.set_ylabel('Solar elevation angle', color='b', fontsize=15)
    # fig.tight_layout()
    # plt.show()
    # # plt.savefig('C:\\Users\\user\\PycharmProjects\\obstacleMobilityModel\\plots\\returns.eps', format='eps')
    # a = np.asarray([i for i in range(24)], dtype=np.int32)
    # fig, ax = plt.subplots(figsize=(10, 4))
    # ax2 = ax.twinx()
    # width = 0.2
    # ax.bar(a.transpose(), (avg_energy_per_hour*n_bs_per_hour).flatten(), color='r', width=width)
    # ax.bar(a.transpose() + width, (avg_energy_per_hour_energy*n_bs_per_hour_energy).flatten(), color='y', width=width)
    # ax2.plot(elev_angles, color='b')
    # ax.set_xticks(a)
    # ax.set_xticklabels(a, rotation=65)
    # ax.legend(['Without RES', 'With RES'])
    # ax.set_xlabel('hour')
    # ax.set_ylabel('Average energy level per DBS')
    # ax2.set_ylabel('Solar elevation angle', color='b')
    # fig.tight_layout()
    # # plt.show()

##################POLISH
#     arrivals_per_hour_energy = np.loadtxt(
#         "C:\\Users\\user\\PycharmProjects\\obstacleMobilityModel\\results_good_weather\\arrivals_per_hour.txt")
#     arrivals_per_hour = np.loadtxt(
#         "C:\\Users\\user\\PycharmProjects\\obstacleMobilityModel\\results\\arrivals_per_hour_no_solar.txt")
#
#     returns_per_hour_energy = np.loadtxt(
#         "C:\\Users\\user\\PycharmProjects\\obstacleMobilityModel\\results_good_weather\\returns_per_hour.txt")
#     returns_per_hour = np.loadtxt(
#         "C:\\Users\\user\\PycharmProjects\\obstacleMobilityModel\\results\\returns_per_hour_no_solar.txt")
#
#     n_bs_per_hour_energy = np.loadtxt(
#         "C:\\Users\\user\\PycharmProjects\\obstacleMobilityModel\\results\\n_bs_per_hour.txt")
#     n_bs_per_hour = np.loadtxt(
#         "C:\\Users\\user\\PycharmProjects\\obstacleMobilityModel\\results\\n_bs_per_hour_no_solar.txt")
#
#     avg_energy_per_hour = np.loadtxt('C:\\Users\\user\\PycharmProjects\\obstacleMobilityModel\\results\\energy_per_hour_no_solar.txt')
#     avg_energy_per_hour_energy = np.loadtxt('C:\\Users\\user\\PycharmProjects\\obstacleMobilityModel\\results\\energy_per_hour.txt')
#
#     elev_angles = []
#     date = datetime(year=2022, day=STARTING_DAY, minute=0, month=STARTING_MONTH, hour=00, tzinfo=timezone.utc)
#     for i in range(24):
#         elev_angles.append(max(0, np.round(get_altitude(40.418725331325, -3.704271435627907, date,
#                                                         elevation=820 + BS_HEIGHT), decimals=2)))
#         date = date + timedelta(hours=1)
#
#     arrivals_per_hour_energy_moderate =  np.loadtxt(
#         "C:\\Users\\user\\PycharmProjects\\obstacleMobilityModel\\results_moderate_weather\\arrivals_per_hour.txt")
#     arrivals_per_hour_energy_bad = np.loadtxt(
#         "C:\\Users\\user\\PycharmProjects\\obstacleMobilityModel\\results_bad_weather\\arrivals_per_hour.txt")
#
# # (1-arrivals_per_hour_energy.sum()/arrivals_per_hour.sum())*arrivals_per_hour.sum()*222
# # (1 - arrivals_per_hour_energy_moderate.sum() / arrivals_per_hour.sum()) * arrivals_per_hour.sum() * 222
# # (1 - arrivals_per_hour_energy_bad.sum() / arrivals_per_hour.sum()) * arrivals_per_hour.sum() * 222
#
#
#
#     a = np.asarray([i for i in range(24)], dtype=np.int32)
#     fig, ax = plt.subplots(figsize=(10, 4))
#     ax2 = ax.twinx()
#     width = 0.2
#     ax.bar(a.transpose(), arrivals_per_hour.flatten(), color='r', width=width)
#     ax.bar(a.transpose() + width, arrivals_per_hour_energy.flatten(), color='y', width=width)
#     ax.bar(a.transpose() + 2*width,  arrivals_per_hour_energy_moderate.flatten(), color='g', width=width)
#     ax.bar(a.transpose() + 3 * width, arrivals_per_hour_energy_bad.flatten(), color='c', width=width)
#
#     ax2.plot(elev_angles, color='b')
#     ax.set_xticks(a)
#     ax.set_xticklabels(a, rotation=65)
#     ax.legend(['Bez OZE', 'OZE/Dobre warunki pogodowe','OZE/Umiarkowane warunki pogodowe', 'OZE/Złe warunki pogodowe'], loc="lower left", prop={'size': 11})
#     ax.set_xlabel('Godzina', fontsize=15)
#     ax.set_ylabel('Liczba przylotów DRS', fontsize=15)
#     ax.tick_params(axis='both', which='major', labelsize=12)
#     ax2.tick_params(axis='both', which='major', labelsize=12)
#     ax2.set_ylabel('Kąt elewacji Słońca', color='b', fontsize=15)
#     fig.tight_layout()
#     plt.show()
#
#     returns_per_hour_energy_moderate = np.loadtxt(
#         "C:\\Users\\user\\PycharmProjects\\obstacleMobilityModel\\results_moderate_weather\\returns_per_hour.txt")
#     returns_per_hour_energy_bad = np.loadtxt(
#         "C:\\Users\\user\\PycharmProjects\\obstacleMobilityModel\\results_bad_weather\\returns_per_hour.txt")
#
#
#     a = np.asarray([i for i in range(24)], dtype=np.int32)
#     fig, ax = plt.subplots(figsize=(10, 4))
#     ax2 = ax.twinx()
#     width = 0.2
#     ax.bar(a.transpose(), returns_per_hour.flatten(), color='r', width=width)
#     ax.bar(a.transpose() + width, returns_per_hour_energy.flatten(), color='y', width=width)
#     ax.bar(a.transpose() + 2*width,  returns_per_hour_energy_moderate.flatten(), color='g', width=width)
#     ax.bar(a.transpose() + 3 * width, returns_per_hour_energy_bad.flatten(), color='c', width=width)
#     ax2.plot(elev_angles, color='b')
#     ax.set_xticks(a)
#     ax.set_xticklabels(a, rotation=65)
#     ax.legend(['Bez OZE', 'OZE/Dobre warunki pogodowe','OZE/Umiarkowane warunki pogodowe', 'OZE/Złe warunki pogodowe'], loc="lower left", prop={'size': 11})
#     ax.set_xlabel('Godzina', fontsize=15)
#     ax.set_ylabel('Liczba powrotów DRS', fontsize=15)
#     ax.tick_params(axis='both', which='major', labelsize=12)
#     ax2.tick_params(axis='both', which='major', labelsize=12)
#     ax2.set_ylabel('Kąt elewacji Słońca', color='b', fontsize=15)
#     fig.tight_layout()
#     plt.show()
