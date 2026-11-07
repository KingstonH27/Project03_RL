# crossy_road/lane_manager.py
import random
from pathlib import Path

import pygame
from .constants import (
    WIN_WIDTH,
    LANE_HEIGHT,
    TRAFFIC_START_LANES_ABOVE_CAMERA,
    BIOME_CHANGE_EVERY,
    DEFAULT_BIOME,
    BIOMES,
    ROW_TYPES,
)
from .entities import Car, Train


ROAD_ASSET_DIR = Path(__file__).resolve().parent / "assets" / "Roads"


class LaneManager:
    """
    Handles:
        • background drawing for generated map rows
        • biome switching (colours & traffic parameters)
        • spawning of cars and trains
    """
    def __init__(self):
        self.all_obstacles = pygame.sprite.Group()   # cars + trains
        self.spawn_timers = {}                       # lane_index → frames until next spawn
        self.rows = {}                               # lane_index → generated row definition
        self.current_biome = DEFAULT_BIOME
        self.road_tiles = self._load_road_tiles()
        self._force_road_next = False

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------
    def reset(self):
        self.all_obstacles.empty()
        self.spawn_timers.clear()
        self.rows.clear()
        self.current_biome = DEFAULT_BIOME
        self._force_road_next = False

    def ensure_rows_through(self, last_lane):
        """Generate rows through ``last_lane`` without advancing the simulation."""
        self._ensure_rows(last_lane)

    # ------------------------------------------------------------------
    #  Biome handling
    # ------------------------------------------------------------------
    def _choose_biome(self, player_world_y):
        """Select a biome based on how far the player has travelled."""
        idx = (player_world_y // BIOME_CHANGE_EVERY) % len(BIOMES)
        self.current_biome = BIOMES[int(idx)]

    def _biome_for_lane(self, lane):
        """Return the biome that applies where this map row was created."""
        idx = (lane * LANE_HEIGHT // BIOME_CHANGE_EVERY) % len(BIOMES)
        return BIOMES[int(idx)]

    def _create_row(self, lane):
        """Generate and retain the next row of the procedural map."""
        was_forced_road = self._force_road_next
        candidates = ROW_TYPES
        if was_forced_road:
            candidates = [
                row for row in ROW_TYPES
                if row.get("obstacle_type") == "car"
            ]
        row_type = random.choices(
            candidates,
            weights=[row["probability"] for row in candidates],
            k=1,
        )[0]
        # A normally selected road reserves one following road row.  That
        # forced row clears the reservation, so non-road rows may resume.
        self._force_road_next = (
            not was_forced_road and row_type.get("obstacle_type") == "car"
        )
        biome = self._biome_for_lane(lane)
        row = {
            "type": row_type,
            "biome": biome,
        }
        # Choose traffic speed when the road is created, rather than when each
        # car spawns.  This keeps every car in a road moving together while
        # allowing separate roads to have different speeds.
        if row_type.get("obstacle_type") == "car":
            row["vehicle_speed"] = random.randint(
                biome["car_min"], biome["car_max"]
            )
        self.rows[lane] = row

    @staticmethod
    def _load_road_tiles():
        """Load the optional road-edge tiles, scaled to the map row height."""
        filenames = {
            "top": "TopRoad.png",
            "middle": "MiddleRoad.png",
            "bottom": "BottomRoad.png",
        }
        tiles = {}
        for name, filename in filenames.items():
            path = ROAD_ASSET_DIR / filename
            if not path.is_file():
                continue
            try:
                tile = pygame.image.load(str(path))
                tiles[name] = pygame.transform.scale(tile, (LANE_HEIGHT, LANE_HEIGHT))
            except pygame.error:
                continue
        return tiles

    def _is_road(self, lane):
        """Whether a row uses car-road artwork rather than grass or tracks."""
        return lane >= 0 and self.rows[lane]["type"].get("obstacle_type") == "car"

    def _road_tile_kind(self, lane):
        """Pick edge artwork from the road rows immediately above and below."""
        has_road_above = self._is_road(lane + 1)
        has_road_below = self._is_road(lane - 1)
        if not has_road_above:
            return "top"
        if not has_road_below:
            return "bottom"
        return "middle"

    def _draw_road_tile(self, surface, lane, top_screen_y):
        """Tile the appropriate road-cap or middle texture across one road row."""
        tile = self.road_tiles.get(self._road_tile_kind(lane))
        if tile is None:
            return
        for x in range(0, WIN_WIDTH, tile.get_width()):
            surface.blit(tile, (x, top_screen_y))

    def _ensure_rows(self, last_lane):
        """Create each missing map row once, so its type never changes."""
        for lane in range(max(0, last_lane) + 1):
            if lane not in self.rows:
                self._create_row(lane)

    # ------------------------------------------------------------------
    #  Update – move existing sprites, spawn new ones, keep timers tidy
    # ------------------------------------------------------------------
    def update(self, camera, player_world_y):
        """Advance timers, possibly spawn new obstacles, and move everything."""
        # 1️⃣  Choose biome
        self._choose_biome(player_world_y)

        # 2️⃣  Difficulty scaling – every 50 lanes speed rises ~10 %
        difficulty_factor = 1 + (player_world_y // (50 * LANE_HEIGHT)) * 0.1

        # 3️⃣  Move existing sprites
        self.all_obstacles.update(camera)

        # 4️⃣  Determine which lanes are currently visible (with a little buffer)
        first_lane = int(camera.offset_y // LANE_HEIGHT) - 2
        last_lane = int((camera.offset_y + camera.height) // LANE_HEIGHT) + 2
        # ``first_lane`` includes two off-screen buffer rows; base the safe area
        # on the actual bottom visible row instead.
        first_traffic_lane = (
            int(camera.offset_y // LANE_HEIGHT)
            + TRAFFIC_START_LANES_ABOVE_CAMERA
        )
        self._ensure_rows(last_lane + 1)

        # 5️⃣  Ensure each traffic lane has a spawn timer
        for lane in range(first_traffic_lane, last_lane + 1):
            if lane < 0:
                continue
            if self.rows[lane]["type"]["has_traffic"]:
                if lane not in self.spawn_timers:
                    self.spawn_timers[lane] = random.randint(30, 180)

        # 6️⃣  Decrement timers and spawn when they hit zero
        for lane in list(self.spawn_timers.keys()):
            if lane < first_traffic_lane:    # behind the traffic spawn area → clean up
                del self.spawn_timers[lane]
                continue

            self.spawn_timers[lane] -= 1
            if self.spawn_timers[lane] <= 0:
                row = self.rows[lane]
                # The row type determines its obstacle; trains only use tracks.
                if row["type"]["obstacle_type"] == "train":
                    train = Train(lane, row["type"]["direction"],
                                  speed_multiplier=difficulty_factor,
                                  biome=row["biome"])
                    train.update_screen_rect(camera)
                    self.all_obstacles.add(train)
                elif row["type"]["obstacle_type"] == "car":
                    car = Car(lane, row["type"]["direction"],
                              speed=row["vehicle_speed"],
                              biome=row["biome"])
                    car.update_screen_rect(camera)
                    self.all_obstacles.add(car)

                # reset the timer for this lane
                self.spawn_timers[lane] = random.randint(30, 180)

    # ------------------------------------------------------------------
    #  Drawing
    # ------------------------------------------------------------------
    def draw(self, surface, camera):
        """
        Draw full-width background rows (road / grass) and then every sprite.
        """
        first_lane = int(camera.offset_y // LANE_HEIGHT) - 2
        last_lane = int((camera.offset_y + camera.height) // LANE_HEIGHT) + 2
        self._ensure_rows(last_lane + 1)

        for lane in range(first_lane, last_lane + 1):
            if lane < 0:
                continue
            # bottom edge of the lane in world coordinates
            bottom_world_y = lane * LANE_HEIGHT
            # screen Y of the *top* of the lane rectangle
            top_screen_y = camera.world_to_screen(0, bottom_world_y)[1] - LANE_HEIGHT

            row = self.rows[lane]
            colour = row["biome"][row["type"]["colour_key"]]
            rect = pygame.Rect(0, top_screen_y, WIN_WIDTH, LANE_HEIGHT)
            pygame.draw.rect(surface, colour, rect)
            if self._is_road(lane):
                self._draw_road_tile(surface, lane, top_screen_y)
            if row["type"].get("obstacle_type") == "train":
                # Simple horizontal rails and sleepers make track rows distinct.
                rail_y = (top_screen_y + LANE_HEIGHT // 4,
                          top_screen_y + 3 * LANE_HEIGHT // 4)
                for y in rail_y:
                    pygame.draw.line(surface, (45, 45, 45), (0, y), (WIN_WIDTH, y), 5)
                for x in range(0, WIN_WIDTH, 32):
                    pygame.draw.line(surface, (115, 75, 45),
                                     (x, rail_y[0] - 8), (x, rail_y[1] + 8), 4)

        # Draw cars and trains on top of the background
        self.all_obstacles.draw(surface)
