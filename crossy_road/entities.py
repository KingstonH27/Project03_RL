# crossy_road/entities.py
import random
from pathlib import Path

import pygame
from .constants import (
    PLAYER_RENDER_SIZE, PLAYER_COLOR, PLAYER_ART_SMOOTHING, PLAYER_HITBOX_SCALE,
    MAX_LIVES,
    LANE_HEIGHT,
    CAR_MIN_WIDTH, CAR_MAX_WIDTH,
    CAR_MIN_SPEED, CAR_MAX_SPEED,
    CAR_SPEED,
    TRAIN_SPEED,
    WIN_WIDTH,
    CAR_HEIGHT,
)


CAR_ASSET_DIR = Path(__file__).resolve().parent / "assets" / "Cars"
PLAYER_ASSET_DIR = Path(__file__).resolve().parent / "assets" / "Player"


def _load_player_frames(filename):
    """Split a horizontal 32×32 player sprite sheet into native-size frames."""
    path = PLAYER_ASSET_DIR / filename
    if not path.is_file():
        return []
    try:
        sheet = pygame.image.load(str(path))
        frame_size = sheet.get_height()
        frames = [
            sheet.subsurface((x, 0, frame_size, frame_size)).copy()
            for x in range(0, sheet.get_width(), frame_size)
        ]
        # Pixel-art sprites use nearest-neighbour scaling, preserving hard edges.
        return [pygame.transform.scale(frame, (PLAYER_RENDER_SIZE, PLAYER_RENDER_SIZE))
                for frame in frames]
    except pygame.error:
        return []


def _load_random_car_image(direction):
    """Load one vehicle sprite from assets/Cars and orient it for its lane."""
    car_images = [
        path for path in CAR_ASSET_DIR.rglob("*.png")
        if path.parent != CAR_ASSET_DIR
    ] if CAR_ASSET_DIR.is_dir() else []
    if not car_images:
        return None

    try:
        image = pygame.image.load(str(random.choice(car_images)))
        # Source sprites face vertically; rotate them to travel across the row.
        return pygame.transform.rotate(image, -90 if direction == 1 else 90)
    except pygame.error:
        # Keep the game playable when an individual asset is unreadable.
        return None

# ----------------------------------------------------------------------
#  Player – lives, score, world position
# ----------------------------------------------------------------------
class Player(pygame.sprite.Sprite):
    """The avatar, stored in world coordinates and drawn via the Camera."""
    def __init__(self, start_x, start_y):
        super().__init__()
        self.animations = {
            "idle": _load_player_frames("Idle (32x32).png"),
            "run": _load_player_frames("Run (32x32).png"),
            "jump": _load_player_frames("Jump (32x32).png"),
            "hit": _load_player_frames("Hit (32x32).png"),
        }
        fallback = pygame.Surface((PLAYER_RENDER_SIZE, PLAYER_RENDER_SIZE), pygame.SRCALPHA)
        fallback.fill(PLAYER_COLOR)
        for name, frames in self.animations.items():
            if not frames:
                self.animations[name] = [fallback]
        self.animation_name = "idle"
        self.animation_frame = 0
        self.animation_ticks = 0
        self.animation_duration = 0
        self.facing = 1
        self.image = self.animations["idle"][0]
        self.world_x = start_x
        self.world_y = start_y
        self.lives = MAX_LIVES
        self.alive = True
        self.score = 0
        self.rect = self.image.get_rect()
        self.visual_rect = self.image.get_rect()
        self._visual_x = None
        self._visual_y = None

    def _refresh_image(self):
        frame = self.animations[self.animation_name][self.animation_frame]
        self.image = pygame.transform.flip(frame, self.facing < 0, False)

    def play_animation(self, name, duration=0):
        self.animation_name = name
        self.animation_frame = 0
        self.animation_ticks = 0
        self.animation_duration = duration
        self._refresh_image()

    def reset(self, start_x, start_y):
        self.world_x = start_x
        self.world_y = start_y
        self.lives = MAX_LIVES
        self.alive = True
        self.score = 0
        self.animation_name = "idle"
        self.animation_frame = 0
        self.animation_ticks = 0
        self.animation_duration = 0
        self.facing = 1
        self._refresh_image()
        self._visual_x = None
        self._visual_y = None

    def move(self, dx=0, dy=0):
        """Move one “tile”.  The caller checks bounds & collisions first."""
        if not self.alive:
            return
        self.world_x += dx
        self.world_y = max(0, self.world_y + dy)
        if dx:
            self.facing = 1 if dx > 0 else -1
            self.play_animation("run", duration=12)
        elif dy:
            self.play_animation("jump", duration=12)

    def update(self, camera):
        """Advance the current player animation and place it in screen space."""
        self.animation_ticks += 1
        if self.animation_ticks % 5 == 0:
            frames = self.animations[self.animation_name]
            self.animation_frame = (self.animation_frame + 1) % len(frames)
        if self.animation_duration:
            self.animation_duration -= 1
            if self.animation_duration == 0:
                self.play_animation("idle")
        self.update_screen_rect(camera)

    def collision_rect_at(self, screen_x, screen_y):
        """Return the scaled collision area at an artwork top-left position."""
        hitbox_width = max(1, round(self.image.get_width() * PLAYER_HITBOX_SCALE))
        hitbox_height = max(1, round(self.image.get_height() * PLAYER_HITBOX_SCALE))
        return pygame.Rect(
            round(screen_x + (self.image.get_width() - hitbox_width) / 2),
            round(screen_y + (self.image.get_height() - hitbox_height) / 2),
            hitbox_width,
            hitbox_height,
        )

    def update_screen_rect(self, camera):
        """Update collision position and smoothly ease the rendered sprite toward it."""
        sx, sy = camera.world_to_screen(self.world_x, self.world_y)
        art_top = sy - self.image.get_height()
        self.rect = self.collision_rect_at(sx, art_top)
        if self._visual_x is None:
            self._visual_x, self._visual_y = sx, art_top
        else:
            self._visual_x += (sx - self._visual_x) * PLAYER_ART_SMOOTHING
            self._visual_y += (art_top - self._visual_y) * PLAYER_ART_SMOOTHING
        self.visual_rect.topleft = (round(self._visual_x), round(self._visual_y))


# ----------------------------------------------------------------------
#  Car – normal traffic vehicle
# ----------------------------------------------------------------------
class Car(pygame.sprite.Sprite):
    def __init__(self, lane_index, direction, speed=CAR_SPEED, biome=None):
        super().__init__()
        self.lane_index = lane_index
        self.direction = direction                # +1 = left→right, -1 = right→left
        self.image = _load_random_car_image(direction)
        if self.image is None:
            self.width = random.randint(CAR_MIN_WIDTH, CAR_MAX_WIDTH)
            self.image = pygame.Surface((self.width, CAR_HEIGHT), pygame.SRCALPHA)
            self.image.fill((200, 0, 0))          # fallback when no asset is available
        else:
            self.width = self.image.get_width()

        # world Y = centre of the lane
        self.world_y = lane_index * LANE_HEIGHT + LANE_HEIGHT // 2

        # start just outside the screen on the appropriate side
        self.world_x = -self.width if direction == 1 else WIN_WIDTH + self.width

        # LaneManager assigns the road's speed, so cars on one road stay in sync.
        self.speed = speed

        self.rect = self.image.get_rect()

    def update_screen_rect(self, camera):
        sx, sy = camera.world_to_screen(self.world_x, self.world_y)
        self.rect.topleft = (sx, sy - self.image.get_height() // 2)

    def update(self, camera):
        """Move horizontally and delete when off‑screen."""
        self.world_x += self.speed * self.direction
        if (self.direction == 1 and self.world_x > WIN_WIDTH) or \
           (self.direction == -1 and self.world_x + self.width < 0):
            self.kill()
            return
        self.update_screen_rect(camera)


# ----------------------------------------------------------------------
#  Train – long, slower obstacle that appears on traffic lanes
# ----------------------------------------------------------------------
class Train(pygame.sprite.Sprite):
    def __init__(self, lane_index, direction, speed_multiplier=1.0, biome=None):
        super().__init__()
        self.lane_index = lane_index
        self.direction = direction
        # make the train noticeably longer than a car
        self.width = random.randint(CAR_MAX_WIDTH + 100, CAR_MAX_WIDTH + 300)

        self.image = pygame.Surface((self.width, CAR_HEIGHT), pygame.SRCALPHA)
        self.image.fill((100, 100, 200))         # placeholder colour

        self.world_y = lane_index * LANE_HEIGHT + LANE_HEIGHT // 2
        self.world_x = -self.width if direction == 1 else WIN_WIDTH + self.width

        # Trains share a fast fixed speed on every track.
        self.speed = TRAIN_SPEED

        self.rect = self.image.get_rect()

    def update_screen_rect(self, camera):
        sx, sy = camera.world_to_screen(self.world_x, self.world_y)
        self.rect.topleft = (sx, sy - CAR_HEIGHT // 2)

    def update(self, camera):
        self.world_x += self.speed * self.direction
        if (self.direction == 1 and self.world_x > WIN_WIDTH) or \
           (self.direction == -1 and self.world_x + self.width < 0):
            self.kill()
            return
        self.update_screen_rect(camera)
