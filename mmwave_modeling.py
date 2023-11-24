import numpy as np

MAXIMUM_PL = 95 #dB

def get_ris_path_loss(d_p1_r, d_r_p2, active_ris=False):
    if not active_ris:
        return 39 + 10 * np.log10(3**2*(d_p1_r + d_r_p2)**2.13)
    else:
        return 39 + 10 * np.log10(3**2*(d_p1_r + d_r_p2)**2.13) - 10

def get_simple_pl(distance):
    return 39 + 10 * 2.13 * np.log10(distance)
