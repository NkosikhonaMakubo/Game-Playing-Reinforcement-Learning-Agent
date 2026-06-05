"""Pygame viewer for StackSmashCore (training demos)."""

from __future__ import annotations

import os

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import math

import pygame
from pymunk import Vec2d

from game_core import (
    HEIGHT,
    LAUNCH_X,
    LAUNCH_Y,
    LEVEL_MAX,
    PLATFORM_H,
    PLATFORM_W,
    PLATFORM_X,
    PLATFORM_Y,
    STACK_RIGHT,
    WIDTH,
    Phase,
    StackSmashCore,
    aim_angle_from_pitch,
)
from sprites import draw_tin_sprite, load_tin_sprites

# Re-export colors from game if missing — define locally
BG = (28, 32, 48)
STACK_PANEL = (38, 44, 62)
LAUNCH_PANEL = (48, 40, 58)
ACCENT = (255, 196, 72)
TEXT = (235, 238, 245)
MUTED = (150, 158, 175)
SUCCESS = (96, 220, 140)
DANGER = (255, 96, 96)

MAX_TINS = 17
BALL_RADIUS = 22


class StackSmashViewer:
    def __init__(self, title: str = "Stack & Smash — Agent") -> None:
        pygame.init()
        pygame.display.set_caption(title)
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("segoeui", 20)
        self.font_lg = pygame.font.SysFont("segoeui", 26, bold=True)
        self.tin_sprites, _ = load_tin_sprites(MAX_TINS)
        self._closed = False

    def handle_events(self) -> bool:
        """Return False if user closed the window."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._closed = True
                return False
        return not self._closed

    def tick(self, fps: int = 60) -> None:
        self.clock.tick(fps)

    def draw_ball(self, core: StackSmashCore) -> None:
        body = core.ball_body
        pos = (int(body.position.x), int(body.position.y))
        pygame.draw.circle(self.screen, (240, 240, 250), pos, BALL_RADIUS)
        pygame.draw.circle(self.screen, (90, 95, 120), pos, BALL_RADIUS, 3)

    def draw_aim_arrow(self, core: StackSmashCore) -> None:
        if core.phase != Phase.AIMING:
            return
        origin = core.ball_body.position
        angle = aim_angle_from_pitch(core.aim_pitch)
        length = 110
        end = origin + Vec2d(math.cos(angle), math.sin(angle)) * length
        o = (int(origin.x), int(origin.y))
        e = (int(end.x), int(end.y))
        pygame.draw.line(self.screen, ACCENT, o, e, 5)

    def draw_background(self) -> None:
        self.screen.fill(BG)
        pygame.draw.rect(self.screen, STACK_PANEL, (0, 0, STACK_RIGHT, HEIGHT))
        pygame.draw.rect(self.screen, LAUNCH_PANEL, (STACK_RIGHT, 0, WIDTH - STACK_RIGHT, HEIGHT))
        for y in range(0, HEIGHT, 14):
            pygame.draw.line(
                self.screen,
                (70, 78, 100),
                (STACK_RIGHT, y),
                (STACK_RIGHT, min(y + 7, HEIGHT)),
                2,
            )
        pygame.draw.rect(
            self.screen,
            (90, 98, 118),
            (PLATFORM_X, PLATFORM_Y, PLATFORM_W, PLATFORM_H),
            border_radius=4,
        )

    def draw(
        self,
        core: StackSmashCore,
        subtitle: str = "",
        extra: str = "",
    ) -> None:
        if not self.handle_events():
            return
        self.draw_background()
        for i, body in enumerate(core.tin_bodies):
            vid = i % len(self.tin_sprites)
            draw_tin_sprite(
                self.screen,
                self.tin_sprites[vid],
                (body.position.x, body.position.y),
                body.angle,
                scale=core.tin_scale,
            )
        if core.phase == Phase.STACKING and core.tins_remaining > 0:
            draw_tin_sprite(
                self.screen,
                self.tin_sprites[core.tins_remaining % len(self.tin_sprites)],
                (core.ghost_x, core.ghost_y),
                0.0,
                alpha=140,
                scale=core.tin_scale,
            )
        if core.phase in (Phase.AIMING, Phase.BALL_ACTIVE, Phase.RESOLVING, Phase.ROUND_END):
            self.draw_ball(core)
        if core.phase == Phase.AIMING:
            self.draw_aim_arrow(core)

        title = self.font_lg.render("Stack & Smash — Training", True, ACCENT)
        self.screen.blit(title, (16, 12))
        info = self.font.render(
            f"Level {core.level}/{LEVEL_MAX}  ·  {core.tin_count()} tins  ·  {core.phase.name}",
            True,
            TEXT,
        )
        self.screen.blit(info, (16, 44))
        if subtitle:
            sub = self.font.render(subtitle, True, MUTED)
            self.screen.blit(sub, (16, 68))
        if extra:
            color = SUCCESS if "win" in extra.lower() else DANGER if "lose" in extra.lower() else TEXT
            ex = self.font.render(extra, True, color)
            self.screen.blit(ex, ex.get_rect(center=(WIDTH // 2, HEIGHT // 2)))
        pygame.display.flip()

    def close(self) -> None:
        if not self._closed:
            pygame.quit()
            self._closed = True