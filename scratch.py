from icc_simulation import *
from visibility_graph import edge_distance
from obstacles import get_madrid_buildings
if __name__ =='__main__':
    simulator = IccSimulator(renewable_energy_flag=True)
    simulator.build_graph()
    simulator.init_stations_in_shortest_route()
    rects_patches = get_madrid_buildings().plot_obstacles(False, fill_color='gray')
    simulator.plot_results(rects_patches)
    simulator.get_shortest_sunny_path()
    simulator._controller.base_stations.__len__()

    dist = 0
    for idx, _p in enumerate(simulator.get_shortest_sunny_path()):
        if idx == 0:
            dist += edge_distance(Point(0, 0), _p)
        else:
            dist += edge_distance(simulator.get_shortest_sunny_path()[idx-1], _p)
        if idx == len(simulator.shortest_path) -1:
            dist += edge_distance(_p, Point(200.0, 370.0))


    print(simulator._controller.base_stations.__len__())


####
arrivals_per_hour_energy = np.loadtxt("C:\\Users\\user\\PycharmProjects\\obstacleMobilityModel\\results\\arrivals_per_hour.txt")
arrivals_per_hour = np.loadtxt("C:\\Users\\user\\PycharmProjects\\obstacleMobilityModel\\results\\arrivals_per_hour_no_solar.txt")

returns_per_hour_energy = np.loadtxt("C:\\Users\\user\\PycharmProjects\\obstacleMobilityModel\\results\\returns_per_hour.txt")
returns_per_hour = np.loadtxt("C:\\Users\\user\\PycharmProjects\\obstacleMobilityModel\\results\\returns_per_hour_no_solar.txt")

n_bs_per_hour_energy = np.loadtxt("C:\\Users\\user\\PycharmProjects\\obstacleMobilityModel\\results\\n_bs_per_hour.txt")
n_bs_per_hour = np.loadtxt("C:\\Users\\user\\PycharmProjects\\obstacleMobilityModel\\results\\n_bs_per_hour_no_solar.txt")



