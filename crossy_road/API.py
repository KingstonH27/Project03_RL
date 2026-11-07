
"""Control and observation helpers for agents; policy decisions belong to callers."""

import numpy as np

from crossy_road import Action
from crossy_road.constants import (
    LANE_HEIGHT,
    PLAYER_RENDER_SIZE,
    SENSING_RADIUS_TILES,
    SENSING_GRID_COLUMNS,
    SENSING_GRID_ROWS,
    SENSING_SIZE_TILES,
)


_DIRECTION_ACTIONS = [
    Action.LEFT,
    Action.RIGHT,
    Action.UP,
    Action.DOWN,
    Action.NOOP
]

_DIRECTION_NAMES = {
    "left": Action.LEFT,
    "right": Action.RIGHT,
    "up": Action.UP,
    "down": Action.DOWN,
    "noop": Action.NOOP,
}

SENSING_TILE_TYPES = (
    "land",
    "road_right",
    "road_left",
    "train_track_right",
    "train_track_left",
    "unknown_ground",
)
SENSING_VEHICLE_TYPES = ("car", "train")


def get_player_data(env):
    """
    Return the observation plus player position, tile types, and vehicle presence.
    """

    observation = env.get_observation()
    player = observation["player"]

    half_width = SENSING_GRID_COLUMNS * LANE_HEIGHT / 2
    half_height = SENSING_GRID_ROWS * LANE_HEIGHT / 2

    player_center_x = player["world_x"] + PLAYER_RENDER_SIZE / 2
    player_center_y = player["world_y"] + PLAYER_RENDER_SIZE / 2

    left = player_center_x - half_width
    right = player_center_x + half_width
    grid_top = player_center_y + half_height
    grid_bottom = player_center_y - half_height

    nearby_obstacles = []

    sensing_tiles = [
        ["unknown_ground" for _ in range(SENSING_GRID_COLUMNS)]
        for _ in range(SENSING_GRID_ROWS)
    ]
    sensing_vehicles = [
        [
            [False for _ in SENSING_VEHICLE_TYPES]
            for _ in range(SENSING_GRID_COLUMNS)
        ]
        for _ in range(SENSING_GRID_ROWS)
    ]

    terrain_by_lane = {}

    for map_row in observation["map_rows"]:
        obstacle_type = map_row["obstacle_type"]
        direction = map_row["direction"]

        if obstacle_type == "car":
            tile_type = "road_right" if direction > 0 else "road_left"
        elif obstacle_type == "train":
            tile_type = "train_track_right" if direction > 0 else "train_track_left"
        else:
            tile_type = "land"

        terrain_by_lane[map_row["lane"]] = tile_type

    for row in range(SENSING_GRID_ROWS):
        cell_top = grid_top - row * LANE_HEIGHT
        cell_bottom = cell_top - LANE_HEIGHT

        lane = int(((cell_top + cell_bottom) / 2) // LANE_HEIGHT)
        tile_type = terrain_by_lane.get(lane, "unknown_ground")

        for column in range(SENSING_GRID_COLUMNS):
            sensing_tiles[row][column] = tile_type

    for obstacle in observation["obstacles"]:
        obstacle_left = obstacle["world_x"]
        obstacle_right = obstacle_left + obstacle["width"]

        obstacle_top = obstacle["world_y"] - obstacle["height"] / 2
        obstacle_bottom = obstacle["world_y"] + obstacle["height"] / 2

        if (
            obstacle_right >= left
            and obstacle_left <= right
            and obstacle_bottom >= grid_bottom
            and obstacle_top <= grid_top
        ):
            nearby_obstacles.append({
                **obstacle,
                "relative_x": obstacle_left - player["world_x"],
                "relative_y": obstacle["world_y"] - player["world_y"],
            })

            for row in range(SENSING_GRID_ROWS):
                cell_top = grid_top - row * LANE_HEIGHT
                cell_bottom = cell_top - LANE_HEIGHT

                for column in range(SENSING_GRID_COLUMNS):
                    cell_left = left + column * LANE_HEIGHT
                    cell_right = cell_left + LANE_HEIGHT

                    if (
                        obstacle_right >= cell_left
                        and obstacle_left <= cell_right
                        and obstacle_bottom >= cell_bottom
                        and obstacle_top <= cell_top
                    ):
                        vehicle_index = SENSING_VEHICLE_TYPES.index(obstacle["type"])
                        sensing_vehicles[row][column][vehicle_index] = True

    # NEW: Convert position to a numeric list
    player_position = [
        player["world_x"],
        player["world_y"]
    ]

    return {
        **observation,
        "current_score": observation["score"],
        "player_position": player_position,
        "nearby_obstacles": nearby_obstacles,
        "sensing_tiles": sensing_tiles,
        "sensing_vehicles": sensing_vehicles,
        "sensing_tile_types": SENSING_TILE_TYPES,
        "sensing_vehicle_types": SENSING_VEHICLE_TYPES,
        "sensing_radius_tiles": SENSING_RADIUS_TILES,
        "sensing_size_tiles": SENSING_SIZE_TILES,
        "sensing_grid_rows": SENSING_GRID_ROWS,
        "sensing_grid_columns": SENSING_GRID_COLUMNS,
    }


def move_player(env, direction):
    """
    Submit the direction with the highest Q-value,
    or accept a direct Action or direction string.
    """

    if isinstance(direction, (list, np.ndarray)):
        if len(direction) != len(_DIRECTION_ACTIONS):
            raise ValueError("Expected exactly 5 Q-values")

        direction = int(np.argmax(direction))
        action = _DIRECTION_ACTIONS[direction]

    elif isinstance(direction, Action):
        action = direction

    elif isinstance(direction, str):
        action = _DIRECTION_NAMES.get(direction.lower())

        if action is None:
            raise ValueError(f"Invalid direction: {direction}")

    else:
        raise TypeError(
            "direction must be a list, NumPy array, Action, or string"
        )

    return env.step(action)
