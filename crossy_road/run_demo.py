#!/usr/bin/env python3
"""
run_demo.py – launch an interactive version of the Crossy‑Road demo.

The demo uses the same Game class that you will later import from the
package, so the logic you develop for an AI agent works exactly the same
as the manual play mode.
"""
import pygame
from crossy_road import Game, Action
from crossy_road.constants import WIN_WIDTH, WIN_HEIGHT


def main():
    # Create a game that *does* render a window
    game = Game(render=True)
    game.ai_enabled = False

    state = "menu"          # possible values: menu, playing, paused, game_over
    running = True

    while running:
        # -----------------------------------------------------------------
        #  Event handling (keyboard only – simple for the demo)
        # -----------------------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                game.handle_click(event.pos)
                continue

            if event.type == pygame.KEYDOWN:
                if state == "menu":
                    if event.key == pygame.K_SPACE:
                        state = "playing"
                        game.reset()
                        game.ai_enabled = True
                elif state == "playing":
                    if event.key == pygame.K_p:
                        state = "paused"
                        game.ai_enabled = False
                    elif event.key == pygame.K_LEFT:
                        game.step(Action.LEFT)
                    elif event.key == pygame.K_RIGHT:
                        game.step(Action.RIGHT)
                    elif event.key == pygame.K_UP:
                        game.step(Action.UP)
                    elif event.key == pygame.K_DOWN:
                        game.step(Action.DOWN)
                elif state == "paused":
                    if event.key == pygame.K_p:
                        state = "playing"
                        game.ai_enabled = True
                elif state == "game_over":
                    if event.key == pygame.K_r:
                        state = "playing"
                        game.reset()
                        game.ai_enabled = True

        # -----------------------------------------------------------------
        #  Frame update (when we are actually playing)
        # -----------------------------------------------------------------
        if state == "playing":
            # give the engine a NOOP step so that time‑based rewards are processed
            observation, reward, done, _ = game.step(Action.NOOP)
            if done:
                state = "game_over"
                game.ai_enabled = False

        # -----------------------------------------------------------------
        #  Render everything (the Game class draws the world + HUD)
        # -----------------------------------------------------------------
        game.render()

        # -----------------------------------------------------------------
        #  Simple overlays for menu / pause / game‑over screens
        # -----------------------------------------------------------------
        if state == "menu":
            overlay = pygame.Surface((WIN_WIDTH, WIN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            game.screen.blit(overlay, (0, 0))
            font = pygame.font.SysFont(None, 48)
            txt = font.render("Press SPACE to start", True, (255, 255, 255))
            rect = txt.get_rect(center=(WIN_WIDTH // 2, WIN_HEIGHT // 2))
            game.screen.blit(txt, rect)
            pygame.display.flip()
        elif state == "paused":
            overlay = pygame.Surface((WIN_WIDTH, WIN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            game.screen.blit(overlay, (0, 0))
            font = pygame.font.SysFont(None, 48)
            txt = font.render("PAUSED – press P to resume", True, (255, 255, 255))
            rect = txt.get_rect(center=(WIN_WIDTH // 2, WIN_HEIGHT // 2))
            game.screen.blit(txt, rect)
            pygame.display.flip()
        elif state == "game_over":
            overlay = pygame.Surface((WIN_WIDTH, WIN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            game.screen.blit(overlay, (0, 0))
            font = pygame.font.SysFont(None, 48)
            txt1 = font.render("GAME OVER", True, (255, 255, 255))
            txt2 = font.render("Press R to retry", True, (255, 255, 255))
            rect1 = txt1.get_rect(center=(WIN_WIDTH // 2, WIN_HEIGHT // 2 - 30))
            rect2 = txt2.get_rect(center=(WIN_WIDTH // 2, WIN_HEIGHT // 2 + 30))
            game.screen.blit(txt1, rect1)
            game.screen.blit(txt2, rect2)
            pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
