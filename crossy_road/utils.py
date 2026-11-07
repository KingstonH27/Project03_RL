# crossy_road/utils.py
import pygame


def draw_text(surface, text, size, pos, colour=(255, 255, 255), centre=True):
    """
    Render *text* on *surface*.
    If ``centre`` is True the text is centered on ``pos``; otherwise ``pos`` is the
    top‑left corner.
    """
    font = pygame.font.SysFont(None, size)
    rendered = font.render(text, True, colour)
    rect = rendered.get_rect()
    if centre:
        rect.center = pos
    else:
        rect.topleft = pos
    surface.blit(rendered, rect)
