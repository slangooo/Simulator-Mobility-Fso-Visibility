import numpy as np
# from math import degrees, atan2
from numpy import degrees, arctan2

def get_azimuth(center_x, center_y, x, y):
    angle = degrees(arctan2(y - center_y, x - center_x))
    bearing = (angle + 360) % 360
    return bearing


def get_mid_azimuth(p_origin, p1, p2):
    azim1 = get_azimuth(p_origin[0], p_origin[1], p1[0], p1[1])
    azim2 = get_azimuth(p_origin[0], p_origin[1], p2[0], p2[1])
    azim11 = angle_in_range(azim1 + 180, 360)
    azim22 = angle_in_range(azim2 + 180, 360)
    if angle_in_range(azim2 - azim1, 360) >= 180:
        return angle_in_range(angle_in_range(azim2 - azim1, 360) / 2 + azim1, 360), azim22, azim11
    elif angle_in_range(azim1 - azim2, 360) >= 180:
        return angle_in_range(angle_in_range(azim1 - azim2, 360) / 2 + azim2, 360), azim11, azim22
    else:
        raise ValueError("Ensure that polygon is convex!")


def rotate(l, n):
    return l[n:] + l[:n]


def to_radians(angle):
    return np.pi * angle / 180


def angle_in_range(angle, range):
    """Limit angle to given range."""
    angle = angle % range
    return (angle + range) % range


def line_point_angle(length, point, angle_x=None, angle_y=None, rounding=True):
    """
    Representing equation of line.

    Parameters
    ----------
    angle_y : angle in degrees from y axis
    angle_x : angle in degrees from x axis
    """
    dest = [0, 0]
    if angle_x is not None:
        if rounding:
            dest[0] = point[0] + length * np.round(np.cos(angle_x / 180 * np.pi), decimals=5)
            dest[1] = point[1] + length * np.round(np.sin(angle_x / 180 * np.pi), decimals=5)
        else:
            dest[0] = point[0] + length * np.cos(angle_x / 180 * np.pi)
            dest[1] = point[1] + length * np.sin(angle_x / 180 * np.pi)
        return dest

    elif angle_y is not None:
        dest[0] = point[0] + length * np.round(np.sin(angle_x / 180 * np.pi), decimals=5)
        dest[1] = point[1] + length * np.round(np.cos(angle_x / 180 * np.pi), decimals=5)
        return dest
    else:
        raise ValueError('No angle was provided!')


def get_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def lin2db(linear_input):
    return 10 * np.log10(linear_input)
