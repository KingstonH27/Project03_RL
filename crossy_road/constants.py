# crossy_road/constants.py
import pygame

# -------------------------------------------------
#  Screen & timing
# -------------------------------------------------
WIN_WIDTH = 640
WIN_HEIGHT = 800
FPS = 120

# -------------------------------------------------
#  World geometry
# -------------------------------------------------
LANE_HEIGHT = 80                     # one logical “tile” high
NUM_LANES = WIN_HEIGHT // LANE_HEIGHT
CAR_HEIGHT = LANE_HEIGHT - 20        # vertical size for cars & trains

# -------------------------------------------------
#  Player
# -------------------------------------------------
PLAYER_SIZE = 50
PLAYER_RENDER_SIZE = LANE_HEIGHT     # render the 32px sprite at one movement-tile size
PLAYER_COLOR = (255, 215, 0)        # bright yellow
PLAYER_ART_SMOOTHING = 0.28         # visual follow speed; lower values create more lag
PLAYER_HITBOX_SCALE = 1.0            # 1.0 matches the player art; lower values shrink it

# -------------------------------------------------
#  Cars
# -------------------------------------------------
CAR_MIN_WIDTH = 80
CAR_MAX_WIDTH = 180
CAR_MIN_SPEED = 2
CAR_MAX_SPEED = 8
CAR_SPEED = 5                         # every standard car moves at this speed
TRAFFIC_START_LANES_ABOVE_CAMERA = 3  # keep this many bottom rows clear of new traffic

# -------------------------------------------------
#  Trains
# -------------------------------------------------
TRAIN_PROBABILITY = 0.12            # base chance a traffic lane spawns a train
TRAIN_SPEED_FACTOR = 0.5            # trains are about half as fast as cars
TRAIN_SPEED = 14                     # every train uses this fast, fixed speed

# -------------------------------------------------
#  Game‑wide parameters
# -------------------------------------------------
MAX_LIVES = 1
FOLLOW_DISTANCE = 3 * LANE_HEIGHT    # keep player three lanes from the bottom

CAMERA_SMOOTHING = 0.12

# -------------------------------------------------
#  Biome handling
# -------------------------------------------------
BIOME_CHANGE_EVERY = 15 * LANE_HEIGHT   # change scenery every 15 lanes

# -------------------------------------------------
#  Map generation
# -------------------------------------------------
# Each newly revealed map row is picked from this list.  ``probability`` is a
# relative weight, so the values do not need to add up to one.  Add new row
# types here (and handle their behaviour in LaneManager) to extend the map.
ROW_TYPES = [
    {"name": "grass", "probability": 0.40, "has_traffic": False,
     "colour_key": "grass_color"},
    {"name": "road_right", "probability": 0.20, "has_traffic": True,
     "obstacle_type": "car", "direction": 1, "colour_key": "road_color"},
    {"name": "road_left", "probability": 0.20, "has_traffic": True,
     "obstacle_type": "car", "direction": -1, "colour_key": "road_color"},
    {"name": "track_right", "probability": 0.10, "has_traffic": True,
     "obstacle_type": "train", "direction": 1, "colour_key": "track_color"},
    {"name": "track_left", "probability": 0.10, "has_traffic": True,
     "obstacle_type": "train", "direction": -1, "colour_key": "track_color"},
]

# -------------------------------------------------
#  Colours (fallback)
# -------------------------------------------------
BG_COLOR = (0, 0, 0)

# -------------------------------------------------
#  Biome definitions (road/grass colours + traffic params)
# -------------------------------------------------
DEFAULT_BIOME = {
    "name": "Default",
    "road_color": (50, 50, 50),
    "grass_color": (34, 139, 34),
    "track_color": (95, 95, 95),
    "car_min": CAR_MIN_SPEED,
    "car_max": CAR_MAX_SPEED,
    "train_prob": TRAIN_PROBABILITY,
}
BIOMES = [
    DEFAULT_BIOME,
    {
        "name": "Desert",
        "road_color": (80, 70, 50),
        "grass_color": (210, 180, 140),
        "track_color": (100, 85, 60),
        "car_min": 3,
        "car_max": 7,
        "train_prob": 0.07,
    },
    {
        "name": "Forest",
        "road_color": (46, 79, 45),
        "grass_color": (34, 85, 34),
        "track_color": (75, 75, 65),
        "car_min": 2,
        "car_max": 5,
        "train_prob": 0.12,
    },
    {
        "name": "City",
        "road_color": (60, 60, 60),
        "grass_color": (20, 100, 20),
        "track_color": (85, 85, 85),
        "car_min": 4,
        "car_max": 9,
        "train_prob": 0.10,
    },
]

# -------------------------------------------------
#  Observation helper – how many lanes ahead we expose to the AI
# -------------------------------------------------
VIEW_LANES_AHEAD = 6
SENSING_GRID_ROWS = 5
SENSING_GRID_COLUMNS = 5
SENSING_RADIUS_TILES = max(SENSING_GRID_ROWS, SENSING_GRID_COLUMNS) // 2
SENSING_SIZE_TILES = SENSING_GRID_ROWS
