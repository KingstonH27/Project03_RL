"""Control and observation helpers for agents; policy decisions belong to callers."""

from crossy_road import Action
from crossy_road.constants import (
    LANE_HEIGHT,
    PLAYER_RENDER_SIZE,
    SENSING_RADIUS_TILES,
    SENSING_SIZE_TILES,
)


_DIRECTION_ACTIONS = {
    "left": Action.LEFT,
    "right": Action.RIGHT,
    "up": Action.UP,
    "down": Action.DOWN,
    "noop": Action.NOOP,
}

SENSING_GRID_CODES = {
    0: "land",
    1: "road",
    2: "train_track",
    3: "unknown_ground",
    4: "car_on_land",
    5: "car_on_road",
    6: "car_on_train_track",
    7: "car_on_unknown_ground",
    8: "train_on_land",
    9: "train_on_road",
    10: "train_on_train_track",
    11: "train_on_unknown_ground",
}


def get_player_data(env):
    """Return the observation plus a numeric terrain/vehicle 5x5 sensing grid.

    Screen visibility does not determine sensing: obstacles present in the game
    observation are included whenever their world-space bounds overlap the square.
    Grid rows run from the player's upper/front side toward the lower/back side;
    columns run left to right. Values are decoded by ``SENSING_GRID_CODES``.
    """
    observation = env.get_observation()
    player = observation["player"]
    half_size = SENSING_SIZE_TILES * LANE_HEIGHT / 2
    player_center_x = player["world_x"] + PLAYER_RENDER_SIZE / 2
    player_center_y = player["world_y"] + PLAYER_RENDER_SIZE / 2
    left = player_center_x - half_size
    right = player_center_x + half_size
    grid_top = player_center_y + half_size
    grid_bottom = player_center_y - half_size

    nearby_obstacles = []
    sensing_grid = [
        [3 for _ in range(SENSING_SIZE_TILES)]
        for _ in range(SENSING_SIZE_TILES)
    ]
    terrain_by_lane = {}
    for map_row in observation["map_rows"]:
        obstacle_type = map_row["obstacle_type"]
        terrain_by_lane[map_row["lane"]] = (
            1 if obstacle_type == "car"
            else 2 if obstacle_type == "train"
            else 0
        )

    for row in range(SENSING_SIZE_TILES):
        cell_top = grid_top - row * LANE_HEIGHT
        cell_bottom = cell_top - LANE_HEIGHT
        lane = int(((cell_top + cell_bottom) / 2) // LANE_HEIGHT)
        terrain_code = terrain_by_lane.get(lane, 3)
        for column in range(SENSING_SIZE_TILES):
            sensing_grid[row][column] = terrain_code

    for obstacle in observation["obstacles"]:
        obstacle_left = obstacle["world_x"]
        obstacle_right = obstacle_left + obstacle["width"]
        obstacle_top = obstacle["world_y"] - obstacle["height"] / 2
        obstacle_bottom = obstacle["world_y"] + obstacle["height"] / 2
        if (obstacle_right >= left and obstacle_left <= right
                and obstacle_bottom >= grid_bottom
                and obstacle_top <= grid_top):
            nearby_obstacles.append({
                **obstacle,
                "relative_x": obstacle_left - player["world_x"],
                "relative_y": obstacle["world_y"] - player["world_y"],
            })
            for row in range(SENSING_SIZE_TILES):
                cell_top = grid_top - row * LANE_HEIGHT
                cell_bottom = cell_top - LANE_HEIGHT
                for column in range(SENSING_SIZE_TILES):
                    cell_left = left + column * LANE_HEIGHT
                    cell_right = cell_left + LANE_HEIGHT
                    if (obstacle_right >= cell_left
                            and obstacle_left <= cell_right
                            and obstacle_bottom >= cell_bottom
                            and obstacle_top <= cell_top):
                        vehicle_flag = 8 if obstacle["type"] == "train" else 4
                        sensing_grid[row][column] |= vehicle_flag

    return {
        **observation,
        "current_score": observation["score"],
        "player_position": {
            "x": player["world_x"],
            "y": player["world_y"],
        },
        "nearby_obstacles": nearby_obstacles,
        "sensing_grid": sensing_grid,
        "sensing_grid_codes": SENSING_GRID_CODES,
        "sensing_radius_tiles": SENSING_RADIUS_TILES,
        "sensing_size_tiles": SENSING_SIZE_TILES,
    }


def move_player(env, direction):
    """Submit a direction or Action to the game and return its step result.

    The game determines whether the requested move is legal; this API does not
    choose a direction or decide what the agent should do.
    """
    if isinstance(direction, Action):
        action = direction
    elif isinstance(direction, str):
        action = _DIRECTION_ACTIONS.get(direction.lower())
        if action is None:
            raise ValueError(f"Invalid direction: {direction}")
    else:
        raise TypeError("direction must be an Action or direction string")
    return env.step(action)
