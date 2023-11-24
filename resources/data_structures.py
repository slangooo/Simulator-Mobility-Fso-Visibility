from dataclasses import dataclass
import numpy as np
from resources.math_utils import get_azimuth, get_distance, angle_in_range, to_radians, rotate, line_point_angle
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from scipy.spatial import ConvexHull, convex_hull_plot_2d, Delaunay
from scipy.spatial.distance import cdist


def point_inside_prlgm(x, y, poly):
    inside = False
    xb = poly[0][0] - poly[1][0]
    yb = poly[0][1] - poly[1][1]
    xc = poly[2][0] - poly[1][0]
    yc = poly[2][1] - poly[1][1]
    xp = x - poly[1][0]
    yp = y - poly[1][1]
    d = xb * yc - yb * xc
    if (d != 0):
        oned = 1.0 / d
        bb = (xp * yc - xc * yp) * oned
        cc = (xb * yp - xp * yb) * oned
        inside = (bb >= 0) & (cc >= 0) & (bb <= 1) & (cc <= 1)
    return inside


@dataclass
class Obstacle:
    __slots__ = ["id", "height", "vertices"]
    id: int
    height: float
    vertices: list

    def get_adjacent_vertices(self, _vertex=None, idx=None):
        if _vertex:
            idx = self.vertices.index(_vertex)
            return [self.vertices[(idx + 1) % len(self.vertices)],
                    self.vertices[(idx - 1 + len(self.vertices)) % len(self.vertices)]]
        else:
            return [(idx + 1) % len(self.vertices),
                    (idx - 1 + len(self.vertices)) % len(self.vertices)]

    def is_overlapping(self, x, y, ref_height=0):
        if ref_height > self.height:
            return False
        point = Point(x, y)
        polygon = Polygon(self.vertices)
        return polygon.contains(point)

    def is_intersecting(self, src_coords, dest_coords):
        # TODO: not tested
        def on_segment(p, q, r):
            if ((q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and
                    (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))):
                return True
            return False

        def orientation(p, q, r):
            # to find the orientation of an ordered triplet (p,q,r)
            # function returns the following values:
            # 0 : Collinear points
            # 1 : Clockwise points
            # 2 : Counterclockwise
            val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))
            if val > 0:
                # Clockwise orientation
                return 1
            elif val < 0:
                # Counterclockwise orientation
                return 2
            else:
                # Collinear orientation
                return 0

        def do_intersect(p1, q1, p2, q2):
            # Find the 4 orientations required for
            # the general and special cases
            o1 = orientation(p1, q1, p2)
            o2 = orientation(p1, q1, q2)
            o3 = orientation(p2, q2, p1)
            o4 = orientation(p2, q2, q1)
            # General case
            if (o1 != o2) and (o3 != o4):
                return True
            # Special Cases
            # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
            if (o1 == 0) and on_segment(p1, p2, q1):
                return True
            # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
            if (o2 == 0) and on_segment(p1, q2, q1):
                return True
            # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
            if (o3 == 0) and on_segment(p2, p1, q2):
                return True
            # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
            if (o4 == 0) and on_segment(p2, q1, q2):
                return True
            # If none of the cases
            return False

        for p1, p2 in zip(self.vertices, rotate(self.vertices, 1)):
            if do_intersect(Coords(p1[0], p1[1]), Coords(p2[0], p2[1]), src_coords, dest_coords):
                return True
        return False

    def is_blocking(self, src_x, src_y, azimuth_to_dest, elevation_to_dest, reference_height=0, max_distance=None):
        if elevation_to_dest < 0:
            return True
        ##########
        shadow_length = (self.height - reference_height) / np.tan(np.deg2rad(elevation_to_dest))

        # _vertices = np.array(self.vertices)
        # src = np.array([[src_x, src_y]])
        # mix_idx = cdist(src, _vertices).argmin()
        # idxs = self.get_adjacent_vertices(idx=mix_idx)
        # idxs.insert(1,mix_idx)
        # _vertices = _vertices[idxs]
        # p0p = line_point_angle(shadow_length, _vertices[0], angle_x=azimuth_to_dest + 180, angle_y=None,
        #                        rounding=False)
        # p1p = line_point_angle(shadow_length, _vertices[1], angle_x=azimuth_to_dest + 180, angle_y=None,
        #                        rounding=False)
        # p2p = line_point_angle(shadow_length, _vertices[2], angle_x=azimuth_to_dest + 180, angle_y=None,
        #                        rounding=False)
        # projections = [p0p, p1p, p2p]
        # plgrm1 = [_vertices[0], p0p, p1p, _vertices[1]]
        # plgrm2 = [_vertices[1], p1p, p2p, _vertices[2]]
        # if point_inside_prlgm(src_x,src_y,plgrm1):
        #     return True
        # elif point_inside_prlgm(src_x,src_y,plgrm2):
        #     return True
        # return False

        p0p = line_point_angle(shadow_length, self.vertices[0], angle_x=azimuth_to_dest + 180, angle_y=None,
                               rounding=False)
        p1p = line_point_angle(shadow_length, self.vertices[1], angle_x=azimuth_to_dest + 180, angle_y=None,
                               rounding=False)
        p2p = line_point_angle(shadow_length, self.vertices[2], angle_x=azimuth_to_dest + 180, angle_y=None,
                               rounding=False)
        p3p = line_point_angle(shadow_length, self.vertices[3], angle_x=azimuth_to_dest + 180, angle_y=None,
                               rounding=False)
        projections = [p0p, p1p, p2p, p3p]
        for p1, p2, p1p, p2p in zip(self.vertices, rotate(self.vertices, 1), projections, rotate(projections, 1)):
            if point_inside_prlgm(src_x, src_y, [p1, p1p, p2p, p2]):
                return True

        # ch = ConvexHull(self.vertices + [p0p, p1p, p2p, p3p])
        # 1st way
        # ch = Delaunay(ch.points)
        # if ch.find_simplex(point) >= 0:
        #     return True
        #
        # #2nd way
        # polygon = Polygon(ch.points)
        # if polygon.contains(point):
        #         return True

        return False
        #########
        # for p1, p2 in zip(self.vertices, rotate(self.vertices, 1)):
        #     azimuth_p_1 = get_azimuth(src_x, src_y, p1[0], p1[1])
        #     azimuth_p_2 = get_azimuth(src_x, src_y, p2[0], p2[1])
        #
        #     if azimuth_p_1 < azimuth_p_2:
        #         p1, p2 = p2, p1
        #         azimuth_p_1 = get_azimuth(src_x, src_y, p1[0], p1[1])
        #         azimuth_p_2 = get_azimuth(src_x, src_y, p2[0], p2[1])
        #
        #     if azimuth_p_1 - azimuth_p_2 <= 180:
        #         az_start, az_end = azimuth_p_2, azimuth_p_1
        #     else:
        #         az_start, az_end = azimuth_p_1, azimuth_p_2
        #     if angle_in_range(azimuth_to_dest + 360 - az_start, 360) > angle_in_range(az_end + 360 - az_start, 360):
        #         continue
        #
        #     azimuth_to_dest = angle_in_range(azimuth_to_dest, 360)
        #
        #     if azimuth_p_2 <= azimuth_to_dest <= azimuth_p_1:
        #         # if abs(azimuth_p_2 - azimuth_p_1) > 90:
        #         #     temp_p1_az = azimuth_p_1 - 180
        #         #     temp_p2_az = (azimuth_p_2 - 180)  if azimuth_p_2 > 180 else azimuth_p_2
        #         #     temp_dest_az = (azimuth_to_dest - 180) if azimuth_to_dest > 180 else azimuth_to_dest
        #         #     if azimuth_p_2 > azimuth_to_dest or azimuth_p_1 < azimuth_to_dest:
        #         #         continue
        #
        #         azimuth_1_p = get_azimuth(p1[0], p1[1], src_x, src_y)
        #         azimuth_1_2 = get_azimuth(p1[0], p1[1], p2[0], p2[1])
        #
        #         angle_1 = angle_in_range(azimuth_1_2 - azimuth_1_p, 180)
        #         angle_p = angle_in_range(azimuth_p_1 - azimuth_to_dest, 180)
        #         angle_2 = 180 - angle_1 - angle_p
        #         side_2 = get_distance(p1, [src_x, src_y])
        #         side_1 = np.sin(to_radians(angle_1)) * side_2 / np.sin(to_radians(angle_2))
        #
        #         elevation_length = side_1 / np.cos(to_radians(elevation_to_dest))
        #
        #         if max_distance is not None and elevation_length > max_distance:
        #             continue
        #
        #         h_to_block = elevation_length * np.sin(to_radians(elevation_to_dest))
        #
        #         if h_to_block <= self.height - reference_height or h_to_block < 0:
        #             return True
        # return False


@dataclass
class Coords:
    __slots__ = ["x", "y"]
    x: float
    y: float

    def __hash__(self):
        return hash((self.x, self.y))

    def __str__(self):
        return f'{{x: {str(self.x)}, y:{str(self.y)}}}'

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def set(self, new_coords: 'Coords'):
        self.x = new_coords.x
        self.y = new_coords.y

    def np_array(self):
        return np.asarray((self.x, self.y))

    def update_coords_from_array(self, np_array):
        self.x = np_array[0]
        self.y = np_array[1]

    def update(self, destination, distance):
        """Update coordinates towards the destination (x,y) with given distance.
            Return True if arrived to destination """
        if self == destination:
            return True

        coord_array = self.np_array()
        direction = - coord_array + destination.np_array()
        remaining_distance = np.linalg.norm(direction)
        direction /= np.linalg.norm(direction)

        if remaining_distance <= distance:
            return True

        else:
            coord_array += direction * distance
            self.update_coords_from_array(coord_array)
            return False

    def get_distance_to(self, other_coords, flag_2d=False):
        if isinstance(other_coords, Coords):
            return np.sqrt((self.x - other_coords.x) ** 2 + (self.y - other_coords.y) ** 2)
        elif isinstance(other_coords, Coords3D):
            return np.sqrt(((self.x - other_coords.x) ** 2 + (self.y - other_coords.y) ** 2
                            + ((self.z - other_coords.z) ** 2 if not flag_2d else 0)))
        elif isinstance(other_coords, tuple) or isinstance(other_coords, tuple):
            squared_sum = (other_coords[0] - self.x) ** 2 + (other_coords[1] - self.y) ** 2
            if len(other_coords) > 2 and not flag_2d:
                squared_sum += (other_coords[2] - self.z) ** 2
            return np.sqrt(squared_sum)
        else:
            raise ValueError('Unidentified input format!')

    def copy(self):
        return Coords(self.x, self.y)


@dataclass
class Coords3D:
    __slots__ = ["x", "y", "z"]
    x: float
    y: float
    z: float

    def set(self, new_coords: 'Coords3D'):
        self.x = new_coords.x
        self.y = new_coords.y
        self.z = new_coords.z

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __str__(self):
        return f'{{x: {str(self.x)}, y:{str(self.y)}, z:{str(self.z)}}}'

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def get_distance_to(self, other_coords, flag_2d=False):
        if isinstance(other_coords, Coords):
            return np.sqrt((self.x - other_coords.x) ** 2 + (self.y - other_coords.y) ** 2 +
                           (self.z ** 2 if not flag_2d else 0))
        elif isinstance(other_coords, Coords3D):
            return np.sqrt(
                (self.x - other_coords.x) ** 2 + (self.y - other_coords.y) ** 2 +
                (((self.z - other_coords.z) ** 2) if not flag_2d else 0))
        elif isinstance(other_coords, tuple) or isinstance(other_coords, tuple):
            squared_sum = (other_coords[0] - self.x) ** 2 + (other_coords[1] - self.y) ** 2
            if len(other_coords) > 2 and not flag_2d:
                squared_sum += (other_coords[2] - self.z) ** 2
            return np.sqrt(squared_sum)
        else:
            raise ValueError('Unidentified input format!')

    def np_array(self):
        return np.asarray((self.x, self.y, self.z))

    def copy(self):
        return Coords3D(self.x, self.y, self.z)

    def set(self, new_x, new_y, new_z=None):
        if new_z is None:
            new_z = self.z
        self.x = new_x
        self.y = new_y
