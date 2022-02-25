import numpy as np
from resources.data_structures import Coords
from parameters import *
from channel_model import PlosModel



class UserEquipment:
    def __init__(self, user_id, initial_coords=None):
        self.user_id = user_id
        if not isinstance(initial_coords, Coords):
            self.coords = Coords(initial_coords[0], initial_coords[1])
        else:
            self.coords = initial_coords

        self.noise_power = UE_NOISE
        self.frequency = UE_FREQ_DEFAULT
        self.bandwidth = UE_BANDWIDTH_DEFAULT
        self.required_rate = UE_RATE_DEFAULT

        self.serving_bs_id = None
        self.distance_to_server = None
        self.max_idx_sinr = None
        self.max_idx_snr = None
        self.received_snr = None
        self.received_sinr = None
        self.los_flag = UE_LOS_DEFAULT

        self.stations_list = []
        self.received_powers = []
        self.received_sinrs = []

    def set_available_base_stations(self, stations_list):
        self.stations_list = stations_list

    def get_received_power(self):
        self.received_powers = np.empty(len(self.stations_list))
        for idx, bs in enumerate(self.stations_list):
            path_loss = PlosModel.get_path_loss(bs.coords, self.coords)
            if bs.is_serving_flag:
                self.received_powers[idx] = (self.bandwidth / bs.total_bandwidth * bs.t_power * 10 ** (-path_loss / 10))
            else:
                self.received_powers[idx] = 0

    def get_received_sinr(self):
        self.get_received_power()
        interferences = np.empty(len(self.received_powers))
        for idx, _ in enumerate(self.received_powers):
            interferences[idx] = np.sum(np.delete(self.received_powers, idx))
        self.received_sinrs = self.received_powers / (interferences+10**(self.noise_power/10))
        self.max_idx_sinr = self.received_sinrs.argmax()

    def associate_to_higesht_sinr(self):
        self.serving_bs_id = self.stations_list[self.max_idx_sinr].base_station_id
        self.received_sinr = self.received_sinrs[self.max_idx_sinr]
        self.received_snr = self.received_powers[self.max_idx_sinr]/10**(self.noise_power/10)
        return self.serving_bs_id

    def associate_to_higesht_snr(self):
        snrs = self.received_powers/(10**(self.noise_power/10))
        self.max_idx_snr = snrs.argmax()
        self.serving_bs_id = self.stations_list[self.max_idx_snr].base_station_id
        self.received_sinr = self.received_sinrs[self.max_idx_snr]
