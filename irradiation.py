# This is the solar irradiation power script for ICC 2022 workshop. Copyright © 2022 Adam Samorzewski.

import math
import random
from datetime import date, datetime, timezone
from scipy import ndimage
import numpy
from parameters import BS_HEIGHT, STARTING_DAY, STARTING_SOLAR_HOUR, STARTING_SOLAR_MINUTE, STARTING_MONTH
from resources.math_utils import to_radians, angle_in_range
from pysolar.solar import get_altitude, get_azimuth, get_position

# Class for manager responsible for evaluating process of the solar irradiation
class PowerManager:

    # Function 1: Default manager object initialization (input: -, output: new object of GenerationManager class)
    def __init___(self):

        # Location coordinates (set for P(0.0, 0.0) point on the map)
        self.coordinates = [0.0, 0.0]  # [°W, °N]

        # Difference between Universal Coordinated Time (UTC) and Local Time (LT) (set for UTC time zone)
        self.UTC = 0  # [h]

        # Height above sea level (set for sea level height)
        self.terrain_height = 0  # [m]

        # Height of UAV's hovering (set for ground level height)
        self.UAV_height = BS_HEIGHT  # [m]

    # Function 2: Manager object initialization with parameters specifying
    # (input: UAV's location coordinates, the Local Time difference from the UTC
    # and total UAV's height, output: new object of GenerationManager class)
    def __init___(self, location, h_t, h_uav, diff_utc):

        # Location coordinates (set for passed location value)
        self.coordinates = location  # [°W, °N]

        # Difference between Universal Coordinated Time (UTC) and Local Time (LT) (set for passed UTC value)
        self.UTC = diff_utc  # [h]

        # Height above sea level (set for passed ground level hight value)
        self.terrain_height = h_t  # [m]

        # Height of UAV's hovering (set for passed flight level height value)
        self.UAV_height = h_uav  # [m]

    # Function 3: The setup of default values for manager object parameters (input: -, output: -)
    def set_manager_default_values(self):

        # Location coordinates (set for Prta del Sol, 7, 28013 Madrid, Spain - Kilometre Zero - from Google Maps)
        self.coordinates = [-3.704271435627907, 40.418725331325]  # [°W, °N]

        # Difference between Universal Coordinated Time (UTC) and Local Time (LT) (set for Madrid, Spain)
        self.UTC = 1  # [h]

        # Height above sea level (set for Madrid, Spain) -> Source: https://en.wikipedia.org/wiki/Madrid
        self.terrain_height = 820  # [m]

        # Height of UAV's hovering
        self.UAV_height = BS_HEIGHT  # [m]

    # Function 4: Setter for coordinates values of the manager object
    # (input: UAV's location coordinates array, output: -)
    def set_coordinates(self, location):

        # Set the value for the manager object parameter
        self.coordinates = location  # [°W, °N]

    # Function 5: Setter for UTC difference value of the manager object
    # (input: the Local Time difference from the UTC, output: -)
    def set_time_zone_utc(self, diff_utc):

        # Set the value for the manager object parameter
        self.UTC = diff_utc  # [h]

    # Function 6: Setter for height above sea level value of the manager object
    # (input: height above sea level, output: -)
    def set_terrain_height_above_sea_level(self, h_t):

        # Set the value for the manager object parameter
        self.terrain_height = h_t  # [m]

    # Function 7: Setter for UAV's hovering hight level value of the manager
    # object (input: UAV's hovering height level, output: -)
    def set_uav_flight_height_level(self, h_uav):

        # Set the value for the manager object parameter
        self.UAV_height = h_uav  # [m]

    # Function 8: Getter for coordinates values of the manager object
    # (input -, output: UAV's location coordinates array)
    def get_coordinates(self):

        # Return value of the manager object parameter
        return self.coordinates

    # Function 9: Getter for UTC difference value of the manager object
    # (input -, output: the Local Time difference from the UTC)
    def get_time_zone_utc(self):

        # Return value of the manager object parameter
        return self.UTC

    # Function 10: Getter for height above sea level value of the manager object
    # (input -, output: height above sea level)
    def get_terrain_height_above_sea_level(self):

        # Return value of the manager object parameter
        return self.terrain_height

    # Function 11: Getter for UAV's hovering hight level value of the manager object
    # (input -, output: UAV's hovering height level)
    def get_uav_flight_height_level(self):

        # Return value of the manager object parameter
        return self.UAV_height

    # # Function 12: Day difference from the beginning of the current year
    # # (input: current date object, output: difference in days)
    # def get_day_difference_from_start_of_the_year(self, current_date):
    #
    #     # Getting year from the current date
    #     current_year = int(current_date.strftime("%Y"))
    #
    #     # Setting start date of the year
    #     start_of_the_year = date(current_year, 1, 1)
    #
    #     # Calculating the difference of the current moment from the beginning of the year
    #     difference = (current_date.date() - start_of_the_year).days
    #
    #     # Return calculated value
    #     return difference

    # Function 13: Day difference from the vernal equinox
    # (input: current date object, output: difference in days)
    def get_day_difference_from_vernal_equinox(self, current_date):

        # Getting day from the current date
        current_day = int(current_date.strftime("%d"))

        # Getting month from the current date
        current_month = int(current_date.strftime("%m"))

        # Getting year from the current date
        current_year = int(current_date.strftime("%Y"))

        # Setting vernal equinox date parameters on 20th March current year
        vernal_equinox_day = 20  # th
        vernal_equinox_month = 3  # March
        vernal_equinox_year = current_year

        # If current day is before vernal equinox, then set its year on previous one
        if current_month <= vernal_equinox_month and current_day < vernal_equinox_day:
            vernal_equinox_year -= 1

        # Setting vernal equinox date
        vernal_equinox_date = date(vernal_equinox_year, vernal_equinox_month, vernal_equinox_day)

        # Calculating the difference of the current moment from the vernal equinox
        difference = abs((vernal_equinox_date - current_date.date()).days)

        # Return calculated value
        return difference

    # Function 14: Solar hour angle calculation (input: current date object, output: angle in degrees)
    def calculate_solar_hour_angle(self, current_date):

        # Source: https://www.pveducation.org/pvcdrom/properties-of-sunlight/solar-time

        # Longitude of considered area
        longitude = self.coordinates[0]  # [°]

        # Difference between Universal Coordinated Time (UTC) and Local Time (LT)
        UTC = self.UTC  # [h] (+ East, - West)

        # Local Standard Time Meridian (LTSM)
        LSTM = 15 * UTC  # [°]

        # Number of days since the start of the year (Function 12)
        # days = self.get_day_difference_from_start_of_the_year(current_date)

        days = (current_date - datetime(year=current_date.year, month=1, day=1)).days

        # B factor for EoT
        B = (360 / 365) * (days - 81)

        # Equation of Time (EoT)
        EoT = 9.87 * math.sin(2 * to_radians(B)) - 7.53 * math.cos(to_radians(B)) - 1.5 * math.sin(to_radians(B))

        # Time Correction Factor (TC)
        TC = 4 * (longitude - LSTM) + EoT

        # Local Time (LT)
        LT = current_date.hour + UTC + (current_date.minute / 60)  # [h]

        # Local Solar Time (LST)
        LST = LT + (TC / 60)  # [h]

        # Solar Hour Angle (HRA)
        HRA = 15 * (LST - 12)  # [°]

        # Return calculated value
        return HRA

    # Function 15: Sine of the solar altitude calculation
    # (input: current date object, output: sine of the solar altitude)
    def calculate_sin_solar_altitude(self, current_date):

        # Source: https://ieeexplore.ieee.org/document/9214030

        # Latitude of considered area
        latitude = to_radians(self.coordinates[1])  # [°]

        # Calculating the UAV's latitude sine
        sin_L = math.sin(latitude)

        # Calculating the UAV's latitude cosine
        cos_L = math.cos(latitude)

        # Calculating the solar hour angle (Function 14)
        H = to_radians(self.calculate_solar_hour_angle(current_date))

        # Calculating cosine of the solar hour angle
        cos_H = math.cos(H)

        # Number of days since the vernal equinox (Function 13)
        days = self.get_day_difference_from_vernal_equinox(current_date)

        # Declination angle
        D = to_radians(23.5 * math.sin(to_radians(360 * days / 365)))  # # [°?]

        # Calculating the declination angle sine
        sin_D = math.sin(D)

        # Calculating the declination angle cosine
        cos_D = math.cos(D)

        # Calculating the solar altitude angle sine
        sin_altitude = cos_D * cos_H * cos_L + sin_L * sin_D

        # numpy.rad2deg(math.asin(sin_altitude))
        #
        # sun_azimuth = numpy.rad2deg(math.acos((sin_D * math.cos(latitude) - cos_H * cos_D * math.sin(latitude)) /math.sin(numpy.deg2rad(90 - numpy.rad2deg(math.asin(sin_altitude))))))

        # Return calculated value
        return sin_altitude

    # Function 16: Current height pressure to sea level pressure ratio calculation (output: pressure ratio)
    def calculate_atmospheric_pressure_factor(self):

        # Source: https://en.wikipedia.org/wiki/Atmospheric_pressure

        # Total height level of UAV's hovering
        height = self.terrain_height + self.UAV_height  # [m]

        # Constant-pressure specific heat
        c_p = 1004.68506  # [J/(kg*K)]

        # Earth-surface gravitational acceleration
        g = 9.80665  # [m/s^2]

        # Sea level standard temperature
        T_o = 288.16  # [K]

        # Molar mass of dry air
        M = 0.02896968  # [kg/mol]

        # Universal gas constant
        R_o = 8.314462618  # [J/mol*K]

        # Sea level standard atmospheric pressure
        sea_level_pressure = 101325  # [Pa]

        # Temperature lapse rate (for dry air)
        L = g / c_p  # [K/m]

        # Determine height atmospheric pressure
        defined_height_level_pressure = sea_level_pressure * (1 - ((L * height) / T_o)) ** ((g * M) / (R_o * L))  # [Pa]

        # UAV's height level pressure to sea level pressure ratio calculation
        pressure_factor = defined_height_level_pressure / sea_level_pressure

        # Return calculated value
        return pressure_factor

    # Function 17: Calculation of constant solar irradiation per unit
    # area (input: current date object, output: power in Watts)
    def caluclate_constant_solar_irradiation_power(self, current_date):

        # Source: https://ieeexplore.ieee.org/document/9214030

        # Mean solar irradiation outside the atmosphere
        I_o = 1353  # [W/m^2]

        # Efficiency of used solar cell type
        n_sc = 0.2  # typical: 15 % - 20 %

        # Area covered by the solar panels
        S_c = 1  # [m^2]

        # Calculating sine of the solar altitude angle (Function 15)
        sin_A = self.calculate_sin_solar_altitude(current_date)

        # Aforementioned value at sea level
        m_o = math.sqrt(1229 + (614 * sin_A) ** 2) - 614 * sin_A

        # Calculating UAV's height level pressure to sea level pressure ratio (Function 5)
        pressure_factor = self.calculate_atmospheric_pressure_factor()

        # Aforementioned value at UAV's height level
        m_z = m_o * pressure_factor

        # Atmospheric transmittance
        T_atm = 0.5 * (math.exp(-0.65 * m_z) + math.exp(-0.095 * m_z))

        # Power of solar irradiation per unit area
        I_b = T_atm * I_o  # [W/m^2]

        # Produced power of the system (UAV)
        produced_power = I_b * n_sc * S_c * sin_A  # [W]

        # Return calculated value
        return I_b

    # Function 18: Generation of solar irradiation random factor
    # (input: mean value and deviation, output: gaussian random number)
    def generate_random_solar_irradiation_power(self, mu, sigma):

        # Solar irradiation factor
        IF = random.gauss(mu, sigma)

        # Return calculated value
        return IF

    # Function 19: Filtering solar irradiation map (input: map array, size
    # of the uniform filter and the mode of filtering, output: filtered map array)
    def filter_calculated_irradiation_values(self, array, size, mode):

        # Source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.uniform_filter.html

        # Filtering map array values with multidimensional uniform filter
        filtered_array = ndimage.uniform_filter(array, size=size, mode=mode)

        # Return calculated values
        return filtered_array

    # Function 20: Determining the matrix of solar irradiation values in particular map points
    # (input: matrix size, filter size, filtering mode, date object of the current moment)
    def define_irradiation_map(self, array_M, array_N, filter_size, filter_mode, current_date):

        # Calculating costant solar irradiation value
        constant_irradiation = self.caluclate_constant_solar_irradiation_power(current_date)

        # Creating matrix of ones with MxN size
        array = numpy.ones((array_M, array_N), dtype=float) * constant_irradiation

        # For every row in the matrix do...
        for row in array:

            # For every element with particular index in the row do...
            for index, element in enumerate(row):
                # Generating random irradiation factor (absolute value)
                random_irradiation_factor = self.generate_random_solar_irradiation_power(0, 1)

                # Multiplying particular element in row by determined factor value
                row[index] = element * (1 + random_irradiation_factor)

        # Filtering created irradiation matrix
        irradiation_array = self.filter_calculated_irradiation_values(array, filter_size, filter_mode)

        # Return calculated values
        return irradiation_array

    def calculate_energy_consumption(self, time_step, transmit_power):
        uav_mass = 4  # [kg]
        uav_propeller_radius = 0.25  # [m]
        uav_number_of_propellers = 4
        air_density = 1.225
        g = 9.80665
        P_hov = math.sqrt(
            ((uav_mass * g) ** 3) / (
                    2 * math.pi * (uav_propeller_radius ** 2) * uav_number_of_propellers * air_density))
        return (P_hov + transmit_power) * time_step

    def calculate_energy_produced(self, time_step, irradiation_magnitude, sin_solar_angle):
        solar_panel_efficiency_factor = 0.2  # typical: 0.15 (15 %) - 0.20 (20 %)
        solar_panel_area = 1  # [m^2]
        produced_power = irradiation_magnitude * solar_panel_efficiency_factor * \
                         solar_panel_area * sin_solar_angle  # [W]
        return time_step * produced_power


if __name__ == '__main__':
    power_manager = PowerManager()
    power_manager.set_manager_default_values()
    _date = datetime(2022, 7, 1, 13, 0)
    constant_irradiation = power_manager.caluclate_constant_solar_irradiation_power(_date)
    sin_solar_angle = power_manager.calculate_sin_solar_altitude(_date)
    irradiation_magnitude = power_manager.generate_random_solar_irradiation_power(0, 1) + constant_irradiation
    print(power_manager.calculate_energy_produced(1, irradiation_magnitude, sin_solar_angle))
    print(power_manager.calculate_energy_consumption(1, 0))
