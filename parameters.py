#Users mobility
NUM_OF_USERS = 1
USER_SPEED = [1.5, 1.7]
PAUSE_INTERVAL = [0, 60]
TIME_STEP = 60*15  # Between subsequent users mobility model updates
TIME_SLEEP = 0.5  # Sleep between updates to allow plotting to keep up

#DBS - UE
NUMBER_OF_BS = 5



STARTING_DAY = 1
STARTING_MONTH = 7
STARTING_SOLAR_HOUR = 3  # 24 format
STARTING_SOLAR_MINUTE = 0
MAX_HOUR_DAY = 23

UE_NOISE = -110  # dBW
UE_FREQ_DEFAULT = 1e6  # Hz
UE_BANDWIDTH_DEFAULT = 500e3  # Hz
UE_RATE_DEFAULT = 1e6  # bps
UE_LOS_DEFAULT = False

# Channel model PLOS
PLOS_AVG_LOS_LOSS = 1
PLOS_AVG_NLOS_LOSS = 20
PLOS_A_PARAM = 9.61
PLOS_B_PARAM = 0.16

# BS
BS_HEIGHT = 20
BS_TOTAL_BANDWIDTH_DEFAULT = 20e6
BS_EXPECTED_SPECTRAL_EFFICIENCY = 2
BS_DEFAULT_TRANSMISSION_POWER = 0.2 #W
BS_STARTING_ENERGY = 222
BS_MAX_ENERGY = 222
BS_COVERAGE_RADIUS = 20

MACRO_BS_LOCATION = [0,0]

FSO_MAX_DISTANCE = 200 #m

SUN_SEARCH_STEP = 7 #m
SUN_SEARCH_COUNT = 5
MAX_SUN_SEARCH_STEPS = 10
BUILDING_EDGE_MARGIN = 1 #m across each axis
SHADOWED_EDGE_PENALTY = 100

NO_SUN = True