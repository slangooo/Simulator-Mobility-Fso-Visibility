import numpy as np
from scipy.constants import speed_of_light
from parameters import *
from resources.data_structures import *


class PlosModel:
    """The model as defined in https://ieeexplore.ieee.org/document/6863654"""

    @staticmethod
    def get_path_loss(bs_coords, ue_coords, frequency=UE_FREQ_DEFAULT, average_los_loss=PLOS_AVG_LOS_LOSS,
                      average_nlos_loss=PLOS_AVG_NLOS_LOSS):
        """Return path loss in dB"""
        distance_2d = ue_coords.get_distance_to(bs_coords, flag_2d=True)
        distance_3d = np.sqrt(distance_2d ** 2 + bs_coords.z ** 2)
        los_probability = PlosModel.get_los_probability(bs_coords.z, distance_2d)
        path_loss = 20 * np.log10(
            4 * np.pi * frequency * distance_3d / speed_of_light) + los_probability * average_los_loss +\
            (1 - los_probability) * average_nlos_loss
        return path_loss

    @staticmethod
    def get_los_probability(height, distance_2d, a_param=PLOS_A_PARAM, b_param=PLOS_B_PARAM):
        return 1 / (1 + a_param * np.exp(-b_param * (180 / np.pi * np.arctan(height / distance_2d) - a_param)))


if __name__ == '__main__':
    print(PlosModel.get_path_loss(Coords3D(0, 0, 0), Coords(0, 0)))
