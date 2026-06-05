"""Load tin sprites from assets/ and draw them on screen."""

from __future__ import annotations

import math
from pathlib import Path

import pygame

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
TIN_SPRITE_W = 48
TIN_SPRITE_H = 80


def build_ghost_sprite(base: pygame.Surface) -> pygame.Surface:
    ghost = base.copy()
    ghost.fill((255, 255, 255, 90), special_flags=pygame.BLEND_RGBA_ADD)
    ghost.set_alpha(170)
    return ghost


def load_tin_sprites(count: int = 6) -> tuple[list[pygame.Surface], pygame.Surface]:
    """Load tin_garbage_0.png .. tin_garbage_{count-1}.png from assets/."""
    sprites: list[pygame.Surface] = []
    for i in range(count):
        path = ASSETS_DIR / f"tin_garbage_{i}.png"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path.name}. Add PNGs under assets/ (need {count} files: "
                f"tin_garbage_0.png through tin_garbage_{count - 1}.png)."
            )
        sprites.append(pygame.image.load(path).convert_alpha())
    return sprites, build_ghost_sprite(sprites[0])


def draw_tin_sprite(
    target: pygame.Surface,
    sprite: pygame.Surface,
    body_position: tuple[float, float],
    angle_rad: float,
    alpha: int = 255,
    scale: float = 1.0,
) -> None:
    """Blit a tin sprite rotated to match the physics body."""
    if scale != 1.0:
        w = max(1, int(TIN_SPRITE_W * scale))
        h = max(1, int(TIN_SPRITE_H * scale))
        sprite = pygame.transform.smoothscale(sprite, (w, h))
    angle_deg = -math.degrees(angle_rad)
    rotated = pygame.transform.rotate(sprite, angle_deg)
    if alpha < 255:
        rotated = rotated.copy()
        rotated.set_alpha(alpha)
    rect = rotated.get_rect(center=(int(body_position[0]), int(body_position[1])))
    target.blit(rotated, rect)