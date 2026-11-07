
# crossy_road/game.py
import enum
import random

import pygame
from .constants import (
    WIN_WIDTH, WIN_HEIGHT, FPS,
    PLAYER_COLOR, PLAYER_RENDER_SIZE,
    LANE_HEIGHT, FOLLOW_DISTANCE,
    BG_COLOR,
    MAX_LIVES,
    VIEW_LANES_AHEAD,
    SENSING_RADIUS_TILES,
    SENSING_SIZE_TILES,
)
from .camera import Camera
from .entities import Player, Train
from .lane_manager import LaneManager
from .utils import draw_text


class Action(enum.Enum):
    """Discrete actions that can be fed to Game.step."""
    NOOP = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4


class Game:
    """
    Core Crossy-Road logic with a clean AI API.
    """

    def __init__(self, render: bool = False):
        self.render_mode = render

        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
            pygame.display.set_caption("Crossy Road - pygame (API demo)")
            self.clock = pygame.time.Clock()
        else:
            pygame.init()

        # Core objects
        self.camera = Camera(WIN_WIDTH, WIN_HEIGHT)
        start_x = WIN_WIDTH // 2
        start_y = 0
        self.player = Player(start_x, start_y)
        self.lanes = LaneManager()

        # Game-state tracking
        self.high_score = 0
        self.done = False
        self.reward = 0.0
        self.last_info = {}
        self.show_debug_grid = False

        # Automatic AI update speed (actions per second)
        self.ai_speed = 20
        self.ai_enabled = True
        self.ai_last_update = pygame.time.get_ticks()

        # Buttons
        self.debug_button_rect = pygame.Rect(
            WIN_WIDTH - 140, WIN_HEIGHT - 48, 130, 36
        )
        self.step_button_rect = pygame.Rect(
            WIN_WIDTH - 280, WIN_HEIGHT - 48, 130, 36
        )

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _world_to_lane(self, world_y):
        return int(world_y // LANE_HEIGHT)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)

        start_x = WIN_WIDTH // 2
        start_y = 0

        self.player.reset(start_x, start_y)
        self.lanes.reset()
        self.camera.offset_y = 0

        self.lanes.ensure_rows_through(
            int((self.camera.offset_y + self.camera.height) // LANE_HEIGHT) + 3
        )

        self.player.update_screen_rect(self.camera)

        self.done = False
        self.reward = 0.0

        self.last_info = {
            "seed": seed,
            "collision": False,
            "terminal_reason": None
        }

        return self.get_observation()

    def handle_click(self, position):
        """Handle in-game UI clicks."""

        if self.debug_button_rect.collidepoint(position):
            self.show_debug_grid = not self.show_debug_grid
            return True

        if self.step_button_rect.collidepoint(position):
            from .model import update
            update(self)
            return True

        return False

    def auto_step(self):
        """Run one AI action whenever the configured interval has elapsed."""
        if not self.ai_enabled or self.ai_speed <= 0:
            return

        now = pygame.time.get_ticks()
        interval_ms = 1000 / self.ai_speed

        if now - self.ai_last_update >= interval_ms:
            self.ai_last_update = now
            from .model import update
            update(self)

    # ------------------------------------------------------------------
    # Debug grid
    # ------------------------------------------------------------------

    def _draw_debug_grid(self):
        grid = pygame.Surface((WIN_WIDTH, WIN_HEIGHT), pygame.SRCALPHA)
        grid_colour = (255, 255, 255, 100)

        for x in range(0, WIN_WIDTH + 1, LANE_HEIGHT):
            pygame.draw.line(
                grid, grid_colour, (x, 0), (x, WIN_HEIGHT), 1
            )

        first_lane = int(self.camera.offset_y // LANE_HEIGHT) - 1
        last_lane = int(
            (self.camera.offset_y + WIN_HEIGHT) // LANE_HEIGHT
        ) + 1

        for lane in range(max(0, first_lane), last_lane + 1):
            _, y = self.camera.world_to_screen(
                0, lane * LANE_HEIGHT
            )
            pygame.draw.line(
                grid, grid_colour, (0, y), (WIN_WIDTH, y), 1
            )

        half_size = SENSING_SIZE_TILES * LANE_HEIGHT / 2

        player_center_x = self.player.world_x + PLAYER_RENDER_SIZE / 2
        player_center_y = self.player.world_y + PLAYER_RENDER_SIZE / 2

        sensing_left, sensing_bottom = self.camera.world_to_screen(
            player_center_x - half_size,
            player_center_y - half_size,
        )

        sensing_area = pygame.Rect(
            sensing_left,
            sensing_bottom - SENSING_SIZE_TILES * LANE_HEIGHT,
            SENSING_SIZE_TILES * LANE_HEIGHT,
            SENSING_SIZE_TILES * LANE_HEIGHT,
        )

        pygame.draw.rect(grid, (40, 220, 255, 35), sensing_area)
        pygame.draw.rect(grid, (40, 220, 255, 220), sensing_area, width=3)

        self.screen.blit(grid, (0, 0))

    # ------------------------------------------------------------------
    # Game step
    # ------------------------------------------------------------------

    def step(self, action: Action):
        """Advance the simulation one frame using action."""

        if self.done:
            return self.get_observation(), 0.0, True, self.last_info.copy()

        if not isinstance(action, Action):
            raise TypeError("action must be an Action enum member")

        # -------------------------------------------------
        # Process player input
        # -------------------------------------------------

        dx, dy = 0, 0

        if action == Action.LEFT:
            dx = -LANE_HEIGHT
        elif action == Action.RIGHT:
            dx = LANE_HEIGHT
        elif action == Action.UP:
            dy = LANE_HEIGHT
        elif action == Action.DOWN:
            dy = -LANE_HEIGHT

        # -------------------------------------------------
        # Attempt the move
        # -------------------------------------------------

        action_applied = False

        if dx != 0 or dy != 0:
            target_x = self.player.world_x + dx
            target_y = max(0, self.player.world_y + dy)

            if 0 <= target_x <= WIN_WIDTH - self.player.image.get_width():

                future_offset_y = self.camera.offset_y
                screen_x = target_x
                screen_y = WIN_HEIGHT - (target_y - future_offset_y)

                temp_rect = self.player.collision_rect_at(
                    screen_x,
                    screen_y - self.player.image.get_height()
                )

                blocked = any(
                    temp_rect.colliderect(ob.rect)
                    for ob in self.lanes.all_obstacles
                )

                if not blocked:
                    self.player.move(dx, dy)
                    action_applied = True
                    #print("[SOUND] hop")

        # -------------------------------------------------
        # Update world
        # -------------------------------------------------

        self.camera.update(
            self.player.world_y,
            FOLLOW_DISTANCE
        )

        self.lanes.update(
            self.camera,
            self.player.world_y
        )

        self.player.update(self.camera)

        # -------------------------------------------------
        # Collision handling
        # -------------------------------------------------

        collided = False

        if self.player.alive:
            if pygame.sprite.spritecollideany(
                self.player,
                self.lanes.all_obstacles
            ):
                collided = True

                #print("[SOUND] crash")

                # Preserve high score
                self.high_score = max(
                    self.high_score,
                    self.player.score
                )

                # Reset player to beginning
                self.player.reset(
                    WIN_WIDTH // 2,
                    0
                )

                # Reset camera
                self.camera.offset_y = 0

                # Update player screen position
                self.player.update_screen_rect(self.camera)

                # Game continues
                self.done = False

        # -------------------------------------------------
        # Scoring and reward
        # -------------------------------------------------

        old_score = self.player.score

        self.player.score = self.player.world_y // LANE_HEIGHT

        reward = self.player.score - old_score

        if collided:
            reward -= 2.0

        self.reward = reward

        self.last_info = {
            "action": action.name,
            "action_applied": action_applied,
            "collision": collided,
            "score_delta": self.player.score - old_score,
            "terminal_reason": None,
        }

        # Return observation.
        # Do NOT call model.update() here.
        # model.update() calls env.step(), so doing so would cause recursion.

        return (
            self.get_observation(),
            reward,
            self.done,
            self.last_info.copy()
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self):
        """Draw the current frame."""

        if not self.render_mode:
            raise RuntimeError(
                "Game was created with render=False; cannot call render()."
            )

        self.auto_step()
        self.screen.fill(BG_COLOR)

        # Background and traffic
        self.lanes.draw(self.screen, self.camera)

        if self.show_debug_grid:
            self._draw_debug_grid()

        # Player
        if self.player.alive:
            self.screen.blit(
                self.player.image,
                self.player.visual_rect
            )

        if self.show_debug_grid:
            pygame.draw.rect(
                self.screen,
                (255, 60, 190),
                self.player.rect,
                width=2
            )

        # HUD
        draw_text(
            self.screen,
            f"Score: {self.player.score}",
            30,
            (80, 20),
            centre=False
        )

        draw_text(
            self.screen,
            f"Lives: {self.player.lives}",
            30,
            (WIN_WIDTH - 260, 20),
            centre=False
        )

        draw_text(
            self.screen,
            f"High: {self.high_score}",
            30,
            (WIN_WIDTH - 280, 50),
            centre=False
        )

        # Debug button
        button_colour = (
            (45, 125, 75)
            if self.show_debug_grid
            else (65, 65, 65)
        )

        pygame.draw.rect(
            self.screen,
            button_colour,
            self.debug_button_rect,
            border_radius=6
        )

        pygame.draw.rect(
            self.screen,
            (220, 220, 220),
            self.debug_button_rect,
            width=1,
            border_radius=6
        )

        label = (
            "Grid: ON"
            if self.show_debug_grid
            else "Grid: OFF"
        )

        draw_text(
            self.screen,
            label,
            22,
            self.debug_button_rect.center
        )

        # Step button
        pygame.draw.rect(
            self.screen,
            (45, 100, 180),
            self.step_button_rect,
            border_radius=6
        )

        pygame.draw.rect(
            self.screen,
            (220, 220, 220),
            self.step_button_rect,
            width=1,
            border_radius=6
        )

        draw_text(
            self.screen,
            "Step",
            22,
            self.step_button_rect.center
        )

        pygame.display.flip()
        self.clock.tick(FPS)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def is_move_possible(self, direction: str):
        """Check whether a move is blocked."""

        dx, dy = 0, 0

        if direction == 'left':
            dx = -LANE_HEIGHT
        elif direction == 'right':
            dx = LANE_HEIGHT
        elif direction == 'up':
            dy = LANE_HEIGHT
        elif direction == 'down':
            dy = -LANE_HEIGHT
        else:
            raise ValueError(f"Invalid direction: {direction}")

        target_x = self.player.world_x + dx
        target_y = max(0, self.player.world_y + dy)

        if not (
            0 <= target_x <= WIN_WIDTH - self.player.image.get_width()
        ):
            return False, (target_x, target_y)

        future_offset_y = self.camera.offset_y
        screen_x = target_x
        screen_y = WIN_HEIGHT - (target_y - future_offset_y)

        temp_rect = self.player.collision_rect_at(
            screen_x,
            screen_y - self.player.image.get_height()
        )

        blocked = any(
            temp_rect.colliderect(ob.rect)
            for ob in self.lanes.all_obstacles
        )

        return (not blocked), (target_x, target_y)

    def front_is_open(self):
        """Check whether the tile directly above the player is free."""
        return self.is_move_possible('up')[0]

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def get_observation(self):
        """Return a dictionary describing the current game state."""

        player_info = {
            "world_x": self.player.world_x,
            "world_y": self.player.world_y,
            "lane": self._world_to_lane(self.player.world_y),
            "lives": self.player.lives,
            "alive": self.player.alive,
            "hitbox_screen": tuple(self.player.rect),
            "art_screen": tuple(self.player.visual_rect),
        }

        legal_actions = {
            "NOOP": True,
            "LEFT": self.is_move_possible("left")[0],
            "RIGHT": self.is_move_possible("right")[0],
            "UP": self.is_move_possible("up")[0],
            "DOWN": self.is_move_possible("down")[0],
        }

        map_rows = []

        for lane, row in sorted(self.lanes.rows.items()):
            row_type = row["type"]

            map_rows.append({
                "lane": lane,
                "type": row_type["name"],
                "has_traffic": row_type["has_traffic"],
                "obstacle_type": row_type.get("obstacle_type"),
                "direction": row_type.get("direction"),
                "biome": row["biome"].get("name", "unknown"),
                "vehicle_speed": row.get("vehicle_speed"),
            })

        obstacles = []
        obstacles_ahead = []

        for spr in self.lanes.all_obstacles:
            lane_diff = (
                spr.world_y - self.player.world_y
            ) // LANE_HEIGHT

            obstacle_info = {
                "type": "train" if isinstance(spr, Train) else "car",
                "world_x": spr.world_x,
                "world_y": spr.world_y,
                "lane": spr.lane_index,
                "speed": spr.speed,
                "direction": spr.direction,
                "width": spr.image.get_width(),
                "height": spr.image.get_height(),
                "hitbox_screen": tuple(spr.rect),
            }

            obstacles.append(obstacle_info)

            if 1 <= lane_diff <= VIEW_LANES_AHEAD:
                rel_x = spr.world_x - self.player.world_x

                obstacles_ahead.append({
                    "type": "train" if isinstance(spr, Train) else "car",
                    "lane_offset": int(lane_diff),
                    "relative_x": rel_x,
                    "direction": spr.direction,
                })

        observation = {
            "player": player_info,
            "score": self.player.score,
            "reward": self.reward,
            "done": self.done,
            "biome": self.lanes.current_biome.get("name", "unknown"),
            "camera": {
                "offset_y": self.camera.offset_y,
                "width": self.camera.width,
                "height": self.camera.height,
            },
            "available_actions": [
                action.name for action in Action
            ],
            "legal_actions": legal_actions,
            "front_open": self.front_is_open(),
            "map_rows": map_rows,
            "obstacles": obstacles,
            "obstacles_ahead": obstacles_ahead,
        }

        return observation
