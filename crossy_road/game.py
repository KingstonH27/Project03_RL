
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
    SENSING_GRID_COLUMNS,
    SENSING_GRID_ROWS,
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

    def __init__(self, render: bool = False, screen=None):
        self.width = screen.get_width() if screen is not None else WIN_WIDTH
        self.height = screen.get_height() if screen is not None else WIN_HEIGHT
        self.render_mode = render or screen is not None
        self._owns_display = render and screen is None

        if self.render_mode:
            pygame.init()
            if screen is None:
                self.screen = pygame.display.set_mode((self.width, self.height))
                pygame.display.set_caption("Crossy Road - pygame (API demo)")
                self.clock = pygame.time.Clock()
            else:
                self.screen = screen
                self.clock = None
        else:
            pygame.init()

        # Core objects
        self.camera = Camera(self.width, self.height)
        start_x = self.width // 2
        start_y = 0
        self.player = Player(start_x, start_y)
        self.lanes = LaneManager(width=self.width)

        # Game-state tracking
        self.high_score = 0
        self.done = False
        self.reward = 0.0
        self.last_info = {}
        self.show_debug_grid = False
        self.show_input_impacts = False
        self.sensitivity_scroll = 0

        # AI updates per rendered game frame; 0.25 means one update every four frames.
        self.ai_speed = 0.125/2
        self.ai_enabled = True
        self.ai_step_accumulator = 0.0

        # Buttons
        button_width = min(130, (self.width - 32) // 3)
        button_gap = 8
        button_left = (
            self.width - (button_width * 3 + button_gap * 2)
        ) // 2
        button_y = self.height - 48
        self.debug_button_rect = pygame.Rect(
            button_left + 2 * (button_width + button_gap),
            button_y,
            button_width,
            36,
        )
        self.step_button_rect = pygame.Rect(
            button_left + button_width + button_gap,
            button_y,
            button_width,
            36,
        )
        self.impact_button_rect = pygame.Rect(
            button_left,
            button_y,
            button_width,
            36,
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

        start_x = self.width // 2
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
        self.ai_step_accumulator = 0.0

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

        if self.impact_button_rect.collidepoint(position):
            self.show_input_impacts = not self.show_input_impacts
            self.sensitivity_scroll = 0
            return True

        if self.step_button_rect.collidepoint(position):
            from .model import update
            update(self)
            return True

        return False

    def scroll_sensitivity(self, amount):
        if self.show_input_impacts:
            self.sensitivity_scroll = max(0, self.sensitivity_scroll + amount)

    def auto_step(self):
        """Run AI actions at a rate relative to the rendered frame rate."""
        if not self.ai_enabled or self.ai_speed <= 0:
            return

        self.ai_step_accumulator += self.ai_speed
        update_count = int(self.ai_step_accumulator)
        self.ai_step_accumulator -= update_count

        for _ in range(update_count):
            from .model import update
            update(self)

    # ------------------------------------------------------------------
    # Debug grid
    # ------------------------------------------------------------------

    def _draw_debug_grid(self, action_scores):
        grid = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        grid_colour = (255, 255, 255, 100)

        for x in range(0, self.width + 1, LANE_HEIGHT):
            pygame.draw.line(
            grid, grid_colour, (x, 0), (x, self.height), 1
            )

        first_lane = int(self.camera.offset_y // LANE_HEIGHT) - 1
        last_lane = int(
            (self.camera.offset_y + self.height) // LANE_HEIGHT
        ) + 1

        for lane in range(max(0, first_lane), last_lane + 1):
            _, y = self.camera.world_to_screen(
                0, lane * LANE_HEIGHT
            )
            pygame.draw.line(
                grid, grid_colour, (0, y), (self.width, y), 1
            )

        half_width = SENSING_GRID_COLUMNS * LANE_HEIGHT / 2
        half_height = SENSING_GRID_ROWS * LANE_HEIGHT / 2

        player_center_x = self.player.world_x + PLAYER_RENDER_SIZE / 2
        player_center_y = self.player.world_y + PLAYER_RENDER_SIZE / 2

        sensing_left, sensing_bottom = self.camera.world_to_screen(
            player_center_x - half_width,
            player_center_y - half_height,
        )

        sensing_area = pygame.Rect(
            sensing_left,
            sensing_bottom - SENSING_GRID_ROWS * LANE_HEIGHT,
            SENSING_GRID_COLUMNS * LANE_HEIGHT,
            SENSING_GRID_ROWS * LANE_HEIGHT,
        )

        pygame.draw.rect(grid, (40, 220, 255, 35), sensing_area)
        pygame.draw.rect(grid, (40, 220, 255, 220), sensing_area, width=3)

        from . import API

        directions = (
            (Action.LEFT, -LANE_HEIGHT, 0),
            (Action.RIGHT, LANE_HEIGHT, 0),
            (Action.UP, 0, LANE_HEIGHT),
            (Action.DOWN, 0, -LANE_HEIGHT),
        )
        action_indices = {
            action: index
            for index, action in enumerate(API._DIRECTION_ACTIONS)
        }
        direction_scores = [
            float(action_scores[action_indices[action]])
            for action, _, _ in directions
        ]
        lowest_score = min(direction_scores)
        score_range = max(direction_scores) - lowest_score

        for action, dx, dy in directions:
            score = float(action_scores[action_indices[action]])
            badness = (
                0.5
                if score_range == 0
                else (max(direction_scores) - score) / score_range
            )
            alpha = int(35 + badness * 170)
            target_x = self.player.world_x + dx
            target_y = self.player.world_y + dy
            marker_x, marker_bottom = self.camera.world_to_screen(
                target_x,
                target_y
            )
            marker = pygame.Rect(
                marker_x,
                marker_bottom - LANE_HEIGHT,
                LANE_HEIGHT,
                LANE_HEIGHT
            )
            marker_surface = pygame.Surface(marker.size, pygame.SRCALPHA)
            marker_surface.fill((255, 0, 0, alpha))
            grid.blit(marker_surface, marker)

        self.screen.blit(grid, (0, 0))

    def _draw_input_sensitivity_panel(
        self,
        sensitivities,
        action_name,
        tile_types,
        vehicle_types,
    ):
        panel = pygame.Rect(8, 78, self.width - 16, self.height - 158)
        pygame.draw.rect(self.screen, (20, 20, 20), panel, border_radius=4)
        pygame.draw.rect(self.screen, (110, 110, 110), panel, width=1, border_radius=4)

        font = pygame.font.SysFont("monospace", 12)
        title = font.render(
            f"Input derivatives for {action_name}",
            True,
            (235, 235, 235)
        )
        self.screen.blit(title, (panel.x + 6, panel.y + 5))

        instructions = font.render(
            "Green raises Q; red lowers Q. Scroll to see all inputs.",
            True,
            (200, 200, 200)
        )
        self.screen.blit(instructions, (panel.x + 6, panel.y + 22))

        entries = [
            ("player_x", float(sensitivities[0])),
            ("player_y", float(sensitivities[1])),
        ]
        tile_input_start = 2
        vehicle_input_start = tile_input_start + (
            SENSING_GRID_ROWS * SENSING_GRID_COLUMNS * len(tile_types)
        )
        for row in range(SENSING_GRID_ROWS):
            for column in range(SENSING_GRID_COLUMNS):
                cell_index = row * SENSING_GRID_COLUMNS + column
                tile_start = tile_input_start + cell_index * len(tile_types)
                for tile_index, tile_type in enumerate(tile_types):
                    entries.append((
                        f"tile[{row},{column}]={tile_type}",
                        float(sensitivities[tile_start + tile_index])
                    ))

                vehicle_start = (
                    vehicle_input_start
                    + cell_index * len(vehicle_types)
                )
                for vehicle_index, vehicle_type in enumerate(vehicle_types):
                    entries.append((
                        f"vehicle[{row},{column}]={vehicle_type}",
                        float(sensitivities[vehicle_start + vehicle_index])
                    ))

        entries.sort(key=lambda entry: abs(entry[1]), reverse=True)
        row_height = 18
        list_top = panel.y + 50
        visible_rows = (panel.height - 58) // row_height
        max_scroll = max(0, len(entries) - visible_rows)
        max_impact = max(
            (abs(value) for _, value in entries),
            default=0.0
        )
        self.sensitivity_scroll = min(self.sensitivity_scroll, max_scroll)
        visible_entries = entries[
            self.sensitivity_scroll:self.sensitivity_scroll + visible_rows
        ]

        for row_index, (name, value) in enumerate(visible_entries):
            strength = abs(value) / max_impact if max_impact else 0.0
            if value > 0:
                colour = (25, 65 + int(130 * strength), 45)
            elif value < 0:
                colour = (70 + int(140 * strength), 35, 35)
            else:
                colour = (55, 55, 55)

            row_rect = pygame.Rect(
                panel.x + 5,
                list_top + row_index * row_height,
                panel.width - 10,
                row_height - 1
            )
            pygame.draw.rect(self.screen, colour, row_rect)
            text = font.render(f"{name}: {value:.4f}", True, (245, 245, 245))
            self.screen.blit(text, (row_rect.x + 4, row_rect.y + 1))

        range_text = font.render(
            f"{self.sensitivity_scroll + 1}-{self.sensitivity_scroll + len(visible_entries)} / {len(entries)}",
            True,
            (210, 210, 210)
        )
        self.screen.blit(
            range_text,
            (panel.right - range_text.get_width() - 8, panel.bottom - 20)
        )

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
        blocked_by_vehicle = False

        if dx != 0 or dy != 0:
            target_x = self.player.world_x + dx
            target_y = max(0, self.player.world_y + dy)

            if 0 <= target_x <= self.width - self.player.image.get_width():

                future_offset_y = self.camera.offset_y
                screen_x = target_x
                screen_y = self.height - (target_y - future_offset_y)

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
                else:
                    blocked_by_vehicle = True

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
            collided = blocked_by_vehicle or (
                pygame.sprite.spritecollideany(
                    self.player,
                    self.lanes.all_obstacles
                ) is not None
            )

            if collided:

                #print("[SOUND] crash")

                # Preserve high score
                self.high_score = max(
                    self.high_score,
                    self.player.score
                )

                # Reset player to beginning
                self.player.reset(
                    self.width // 2,
                    0
                )

                # Reset camera
                self.camera.offset_y = 0

                # Generate a fresh map for the next run
                self.lanes.reset()
                self.lanes.ensure_rows_through(
                    int((self.camera.offset_y + self.camera.height) // LANE_HEIGHT) + 3
                )

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

    def render(self, action_scores=None):
        """Draw the current frame."""

        if not self.render_mode:
            raise RuntimeError(
                "Game was created with render=False; cannot call render()."
            )

        self.auto_step()
        self.screen.fill(BG_COLOR)

        from . import model as ai_model

        current_data = None
        current_inputs = None
        if action_scores is None or self.show_input_impacts:
            current_data = ai_model.API.get_player_data(self)
            current_inputs = ai_model.getInputs(current_data)
        if action_scores is None:
            action_scores = ai_model.predict(current_inputs)
        selected_action_index = max(
            range(len(action_scores)),
            key=lambda index: action_scores[index]
        )
        input_sensitivities = None
        if self.show_input_impacts:
            input_sensitivities = ai_model.input_sensitivities(
                current_inputs,
                selected_action_index
            )

        # Background and traffic
        self.lanes.draw(self.screen, self.camera)

        if self.show_debug_grid:
            self._draw_debug_grid(action_scores)

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

        if self.show_input_impacts:
            self._draw_input_sensitivity_panel(
                input_sensitivities,
                ai_model.API._DIRECTION_ACTIONS[selected_action_index].name,
                ai_model.API.SENSING_TILE_TYPES,
                ai_model.API.SENSING_VEHICLE_TYPES,
            )

        # HUD
        if self.width < 420:
            draw_text(self.screen, f"Score: {self.player.score}", 18, (8, 6), centre=False)
            draw_text(self.screen, f"Lives: {self.player.lives}", 18, (8, 26), centre=False)
            draw_text(self.screen, f"High: {self.high_score}", 18, (8, 46), centre=False)
        else:
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
                (self.width - 260, 20),
                centre=False
            )
            draw_text(
                self.screen,
                f"High: {self.high_score}",
                30,
                (self.width - 280, 50),
                centre=False
            )

        outputs_x = self.width - 250
        outputs_y = 78
        outputs_font = pygame.font.SysFont(None, 16)
        outputs_panel = pygame.Rect(outputs_x, outputs_y, 242, 112)
        pygame.draw.rect(self.screen, (20, 20, 20), outputs_panel, border_radius=4)
        pygame.draw.rect(self.screen, (110, 110, 110), outputs_panel, width=1, border_radius=4)
        self.screen.blit(
            outputs_font.render("Action outputs", True, (235, 235, 235)),
            (outputs_x + 7, outputs_y + 5)
        )

        for index, action in enumerate(ai_model.API._DIRECTION_ACTIONS):
            row = outputs_font.render(
                f"{action.name}: {float(action_scores[index]):.4f}",
                True,
                (205, 205, 205)
            )
            self.screen.blit(row, (outputs_x + 7, outputs_y + 23 + index * 16))

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
            min(22, max(12, self.debug_button_rect.width // 5)),
            self.debug_button_rect.center
        )

        impact_colour = (
            (45, 125, 75)
            if self.show_input_impacts
            else (65, 65, 65)
        )
        pygame.draw.rect(
            self.screen,
            impact_colour,
            self.impact_button_rect,
            border_radius=6
        )
        pygame.draw.rect(
            self.screen,
            (220, 220, 220),
            self.impact_button_rect,
            width=1,
            border_radius=6
        )
        draw_text(
            self.screen,
            "Impact: ON" if self.show_input_impacts else "Impact: OFF",
            min(20, max(11, self.impact_button_rect.width // 5)),
            self.impact_button_rect.center
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
            min(22, max(12, self.step_button_rect.width // 5)),
            self.step_button_rect.center
        )

        if self._owns_display:
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
            0 <= target_x <= self.width - self.player.image.get_width()
        ):
            return False, (target_x, target_y)

        future_offset_y = self.camera.offset_y
        screen_x = target_x
        screen_y = self.height - (target_y - future_offset_y)

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
