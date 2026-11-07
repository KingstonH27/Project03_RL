# crossy_road/camera.py
from .constants import CAMERA_SMOOTHING


class Camera:
    """
    Camera that follows the player so that they stay FOLLOW_DISTANCE
      pixels from the bottom of the window.
    """
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.offset_x = 0          # (unused – could support side‑scroll)
        self.offset_y = 0          # vertical world offset (scroll)

    def update(self, player_world_y, follow_distance):
        """Scroll the view so the player stays `follow_distance` from the bottom."""
        target_offset_y = max(0, player_world_y - follow_distance)
        self.offset_y += (target_offset_y - self.offset_y) * CAMERA_SMOOTHING
        if abs(target_offset_y - self.offset_y) < 0.5:
            self.offset_y = target_offset_y

    def world_to_screen(self, world_x, world_y):
        """
        Convert a point expressed in *world* coordinates to screen coordinates.
        Returns integer pixel positions (screen_x, screen_y) where ``screen_y`` is
        the *bottom* of the object (the same convention used for obstacle rects).
        """
        screen_x = world_x - self.offset_x
        screen_y = self.height - (world_y - self.offset_y)
        return int(screen_x), int(screen_y)
