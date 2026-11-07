#!/usr/bin/env python3
"""
run_demo.py – launch an interactive version of the Crossy‑Road demo.

To run:
cd /Users/27hurt/PycharmProjects/Project03_RL
./.venv/bin/python -m crossy_road.run_demo

The demo uses the same Game class that you will later import from the
package, so the logic you develop for an AI agent works exactly the same
as the manual play mode.
"""
import pygame
import numpy as np
from crossy_road import Game, Action
from crossy_road import model as ai_model
from crossy_road.constants import (
    BG_COLOR,
    FPS,
    TRAINING_GAMES_X,
    TRAINING_GAMES_Y,
    WIN_WIDTH,
    WIN_HEIGHT,
)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption("Crossy Road - shared AI training")
    clock = pygame.time.Clock()

    def viewport_for(index):
        column = index % TRAINING_GAMES_X
        row = index // TRAINING_GAMES_X
        left = column * WIN_WIDTH // TRAINING_GAMES_X
        right = (column + 1) * WIN_WIDTH // TRAINING_GAMES_X
        top = row * WIN_HEIGHT // TRAINING_GAMES_Y
        bottom = (row + 1) * WIN_HEIGHT // TRAINING_GAMES_Y
        return pygame.Rect(left, top, right - left, bottom - top)

    game_count = TRAINING_GAMES_X * TRAINING_GAMES_Y
    games = [
        Game(
            render=True,
            screen=pygame.Surface(viewport_for(index).size)
        )
        for index in range(game_count)
    ]
    for game in games:
        game.ai_enabled = False

    state = "menu"
    running = True
    training_accumulator = 0.0

    def game_at(position):
        for index, game in enumerate(games):
            viewport = viewport_for(index)
            if viewport.collidepoint(position):
                local_position = (
                    position[0] - viewport.x,
                    position[1] - viewport.y,
                )
                return game, local_position
        return None, None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            if event.type == pygame.MOUSEWHEEL:
                game, _ = game_at(pygame.mouse.get_pos())
                if game is not None:
                    game.scroll_sensitivity(-event.y * 5)
                continue

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                game, local_position = game_at(event.pos)
                if game is not None:
                    game.handle_click(local_position)
                continue

            if event.type == pygame.KEYDOWN:
                if state == "menu":
                    if event.key == pygame.K_SPACE:
                        state = "playing"
                        for game in games:
                            game.reset()
                        training_accumulator = 0.0
                elif state == "playing":
                    if event.key == pygame.K_p:
                        state = "paused"
                        training_accumulator = 0.0
                    elif event.key == pygame.K_LEFT:
                        game, _ = game_at(pygame.mouse.get_pos())
                        if game is not None:
                            game.step(Action.LEFT)
                    elif event.key == pygame.K_RIGHT:
                        game, _ = game_at(pygame.mouse.get_pos())
                        if game is not None:
                            game.step(Action.RIGHT)
                    elif event.key == pygame.K_UP:
                        game, _ = game_at(pygame.mouse.get_pos())
                        if game is not None:
                            game.step(Action.UP)
                    elif event.key == pygame.K_DOWN:
                        game, _ = game_at(pygame.mouse.get_pos())
                        if game is not None:
                            game.step(Action.DOWN)
                elif state == "paused":
                    if event.key == pygame.K_p:
                        state = "playing"
                        training_accumulator = 0.0
                    elif event.key == pygame.K_r:
                        state = "playing"
                        for game in games:
                            game.reset()
                        training_accumulator = 0.0

        if state == "playing":
            active_games = [game for game in games if not game.done]
            for game in active_games:
                _, _, done, _ = game.step(Action.NOOP)
                if done:
                    game.ai_enabled = False

            if active_games:
                training_accumulator += games[0].ai_speed
                update_count = int(training_accumulator)
                training_accumulator -= update_count
                for _ in range(update_count):
                    ai_model.update_many(active_games)

        screen.fill(BG_COLOR)
        render_inputs = np.stack([
            ai_model.getInputs(ai_model.API.get_player_data(game))
            for game in games
        ]).astype(np.float32)
        render_scores = ai_model.predict(render_inputs)

        for index, game in enumerate(games):
            game.render(action_scores=render_scores[index])
            viewport = viewport_for(index)
            screen.blit(game.screen, viewport.topleft)
            pygame.draw.rect(screen, (175, 175, 175), viewport, width=1)

        if state == "menu":
            overlay = pygame.Surface((WIN_WIDTH, WIN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 170))
            screen.blit(overlay, (0, 0))
            font = pygame.font.SysFont(None, 42)
            txt = font.render("Press SPACE to start", True, (255, 255, 255))
            rect = txt.get_rect(center=(WIN_WIDTH // 2, WIN_HEIGHT // 2))
            screen.blit(txt, rect)
        elif state == "paused":
            overlay = pygame.Surface((WIN_WIDTH, WIN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            screen.blit(overlay, (0, 0))
            font = pygame.font.SysFont(None, 48)
            txt = font.render("PAUSED – press P to resume", True, (255, 255, 255))
            rect = txt.get_rect(center=(WIN_WIDTH // 2, WIN_HEIGHT // 2))
            screen.blit(txt, rect)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
