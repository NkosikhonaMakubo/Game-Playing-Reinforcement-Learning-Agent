"""
Stack & Smash — a two-phase tin tower game.

Phase 1: Stack every tin vertically on the FIRST tin you drop (the base).
Phase 2: Smash the tower — win only if all level tins are in that column and all fall.

Controls
--------
Stacking:  move mouse (or arrows/WASD) to position the next tin, SPACE to drop
           ENTER when done placing to lock the stack and continue
Throwing:  UP/DOWN (or W/S) to aim, ENTER to throw
Any time:  R to restart
"""

from __future__ import annotations

import math
import os
import sys
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import pygame
import pymunk
from pymunk import Vec2d

from game_core import (
    AIM_LEFT,
    BALL_MASS,
    BALL_RADIUS,
    BASE_TIN_H,
    BASE_TIN_W,
    DEFAULT_PITCH,
    GHOST_MOVE_SPEED,
    GHOST_Y_MAX,
    GHOST_Y_MIN,
    HEIGHT,
    LAUNCH_X,
    LAUNCH_Y,
    LEVEL_MAX,
    LEVEL_MIN,
    MAX_PITCH,
    MAX_TINS,
    PLATFORM_H,
    PLATFORM_W,
    PLATFORM_X,
    PLATFORM_Y,
    STACK_RIGHT,
    SETTLE_SPEED,
    SETTLE_TIME,
    THROW_SPEED,
    WIDTH,
    Phase,
    add_platform,
    add_walls,
    aim_angle_from_pitch,
    clamp_ghost_pos,
    make_ball,
    make_space,
    make_tin,
    max_stack_height,
    stack_is_stable,
    tin_is_knocked,
    tin_scale_for_level,
    tins_for_level,
)
from sprites import TIN_SPRITE_H, TIN_SPRITE_W, draw_tin_sprite, load_tin_sprites

# --- display ---
FPS = 60
BG = (28, 32, 48)
STACK_PANEL = (38, 44, 62)
LAUNCH_PANEL = (48, 40, 58)
ACCENT = (255, 196, 72)
DANGER = (255, 96, 96)
SUCCESS = (96, 220, 140)
TEXT = (235, 238, 245)
MUTED = (150, 158, 175)

THROW_START_PROMPT = "Press Enter To Start Throwing"


def draw_ball(surface: pygame.Surface, body: pymunk.Body) -> None:
    pos = (int(body.position.x), int(body.position.y))
    pygame.draw.circle(surface, (240, 240, 250), pos, BALL_RADIUS)
    pygame.draw.circle(surface, (90, 95, 120), pos, BALL_RADIUS, 3)
    highlight = (pos[0] - 7, pos[1] - 7)
    pygame.draw.circle(surface, (255, 255, 255), highlight, 6)


def draw_aim_arrow(surface: pygame.Surface, origin: Vec2d, pitch: float) -> None:
    angle = aim_angle_from_pitch(pitch)
    length = 110
    end = origin + Vec2d(math.cos(angle), math.sin(angle)) * length
    o = (int(origin.x), int(origin.y))
    e = (int(end.x), int(end.y))
    pygame.draw.line(surface, ACCENT, o, e, 5)
    head = end - Vec2d(math.cos(angle), math.sin(angle)) * 22
    left = head + Vec2d(math.cos(angle + 2.6), math.sin(angle + 2.6)) * 16
    right = head + Vec2d(math.cos(angle - 2.6), math.sin(angle - 2.6)) * 16
    pygame.draw.polygon(
        surface,
        ACCENT,
        [e, (int(left.x), int(left.y)), (int(right.x), int(right.y))],
    )


class Game:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Stack & Smash — Tin Tower")
        self.window = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        self.screen = pygame.Surface((WIDTH, HEIGHT))
        self._window_scale = 1.0
        self._window_offset = (0, 0)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("segoeui", 22)
        self.font_lg = pygame.font.SysFont("segoeui", 30, bold=True)
        self.font_sm = pygame.font.SysFont("segoeui", 18)
        self.tin_sprites, self.tin_ghost_sprite = load_tin_sprites(MAX_TINS)
        self.tin_variant_ids: list[int] = []
        self.level = LEVEL_MIN
        self.total_score = 0
        self._score_at_level_start = 0
        self.start_round()

    def tin_count(self) -> int:
        return tins_for_level(self.level)

    def start_round(self) -> None:
        """Start (or restart) the current level."""
        self.space = make_space()
        add_platform(self.space)
        add_walls(self.space)
        self.tin_bodies = []
        self.tin_shapes = []
        self.tin_variant_ids = []
        self.ball_body, self.ball_shape = make_ball(self.space)
        self.phase = Phase.STACKING
        self.tins_remaining = self.tin_count()
        self.tin_scale = tin_scale_for_level(self.level)
        self.tin_w = BASE_TIN_W * self.tin_scale
        self.tin_h = BASE_TIN_H * self.tin_scale
        self.ghost_x = PLATFORM_X + PLATFORM_W / 2
        self.ghost_y = PLATFORM_Y - TIN_SPRITE_H * self.tin_scale * 1.1
        self.ghost_variant = 0
        self.aim_pitch = DEFAULT_PITCH
        self._ball_moment = pymunk.moment_for_circle(BALL_MASS, 0, BALL_RADIUS)
        self.stack_score = 0
        self.throw_score = 0
        self.message = ""
        self.message_color = TEXT
        self.top_prompt = ""
        self.settle_timer = 0.0
        self.throw_result = None
        self.lose_detail = ""
        self.stack_locked = False
        self._stack_settle_checked = False

    def _tower_tins_now(self) -> list:
        from game_core import tower_tins_at_lock

        tower, _, _, _ = tower_tins_at_lock(self.tin_bodies, self.tin_w, self.tin_h)
        return tower

    def _has_valid_vertical_tower(self) -> bool:
        from game_core import all_level_tins_on_tower

        required = self.tin_count()
        placed = len(self.tin_bodies)
        if self.tins_remaining > 0 or placed != required:
            return False
        if not stack_is_stable(self.tin_bodies):
            return False
        tower = self._tower_tins_now()
        return all_level_tins_on_tower(required, placed, len(tower))

    def _fail_invalid_tower(self, detail: str | None = None) -> None:
        """Invalid stack (e.g. tins beside each other) — immediate lose."""
        self.throw_result = "lose"
        self.lose_detail = detail or "All tins must form a vertical tower on the first tin"
        self.stack_score = 0
        self.throw_score = 0
        self.phase = Phase.ROUND_END
        self.top_prompt = ""
        self.message = ""
        self.message_color = DANGER
        self._stack_settle_checked = True

    def reset_game(self) -> None:
        """Full restart from level 1."""
        self.level = LEVEL_MIN
        self.total_score = 0
        self._score_at_level_start = 0
        self.start_round()

    def _window_to_game(self, pos: tuple[int, int]) -> tuple[float, float]:
        ox, oy = self._window_offset
        s = self._window_scale
        if s <= 0:
            return (float(pos[0]), float(pos[1]))
        return ((pos[0] - ox) / s, (pos[1] - oy) / s)

    def _on_window_resize(self, w: int, h: int) -> None:
        w = max(480, w)
        h = max(360, h)
        self.window = pygame.display.set_mode((w, h), pygame.RESIZABLE)

    def _present(self) -> None:
        """Scale fixed game canvas to the resizable window (maximize / drag edges)."""
        win_w, win_h = self.window.get_size()
        if win_w < 1 or win_h < 1:
            return
        scale = min(win_w / WIDTH, win_h / HEIGHT)
        dest_w = max(1, int(WIDTH * scale))
        dest_h = max(1, int(HEIGHT * scale))
        scaled = pygame.transform.smoothscale(self.screen, (dest_w, dest_h))
        self.window.fill(BG)
        ox = (win_w - dest_w) // 2
        oy = (win_h - dest_h) // 2
        self.window.blit(scaled, (ox, oy))
        self._window_scale = scale
        self._window_offset = (ox, oy)
        pygame.display.flip()

    def drop_ghost_tin(self) -> None:
        if self.tins_remaining <= 0:
            return
        body, shape = make_tin(self.space, self.ghost_x, self.ghost_y, self.tin_w, self.tin_h)
        self.tin_bodies.append(body)
        self.tin_shapes.append(shape)
        self.tin_variant_ids.append(self.ghost_variant)
        self.tins_remaining -= 1
        self.ghost_variant = (self.ghost_variant + 1) % len(self.tin_sprites)
        if self.tins_remaining == 0:
            self.phase = Phase.STACK_SETTLING
            self.settle_timer = 0.0
            self.top_prompt = ""
            self._stack_settle_checked = False

    def lock_stack(self) -> None:
        if not self._has_valid_vertical_tower():
            self._fail_invalid_tower()
            return
        from game_core import score_stack, tower_tins_at_lock

        tower, _cx, height, _spread = tower_tins_at_lock(
            self.tin_bodies, self.tin_w, self.tin_h
        )
        self.tower_tins = tower
        self.stack_score = score_stack(len(tower), height, self.tin_h)
        self.total_score += self.stack_score
        self.stack_locked = True
        self.phase = Phase.AIMING
        self.aim_pitch = DEFAULT_PITCH
        self.top_prompt = ""
        self.message_color = TEXT

    def launch_ball(self) -> None:
        # KINEMATIC → DYNAMIC clears mass/moment in pymunk; must restore before step()
        self.ball_body.body_type = pymunk.Body.DYNAMIC
        self.ball_body.mass = BALL_MASS
        self.ball_body.moment = self._ball_moment
        angle = aim_angle_from_pitch(self.aim_pitch)
        direction = Vec2d(math.cos(angle), math.sin(angle))
        self.ball_body.velocity = direction * THROW_SPEED
        self.phase = Phase.BALL_ACTIVE
        self.settle_timer = 0.0
        self.message = ""

    def resolve_throw(self) -> None:
        tower = getattr(self, "tower_tins", self.tin_bodies)
        knocked = sum(1 for b in tower if tin_is_knocked(b, self.tin_h))
        required = self.tin_count()
        placed = len(self.tin_bodies)
        n = len(tower)
        from game_core import all_level_tins_on_tower, level_complete, score_throw

        self.throw_score = score_throw(knocked)
        if level_complete(required, n, knocked, placed):
            self.throw_result = "win"
            if self.level >= LEVEL_MAX:
                self.message = "Level 12 cleared! You beat the game! Press ENTER."
            elif self.level + 1 <= LEVEL_MAX:
                next_tins = tins_for_level(self.level + 1)
                self.message = (
                    f"Level {self.level} complete! "
                    f"Press ENTER for level {self.level + 1} ({next_tins} tins)."
                )
            self.message_color = SUCCESS
        else:
            self.throw_result = "lose"
            if not all_level_tins_on_tower(required, placed, n):
                self.lose_detail = (
                    f"Only {n} of {required} tins were in the vertical tower. "
                    f"Stack on the first tin, then smash"
                )
            else:
                still_up = required - knocked
                self.lose_detail = (
                    f"{still_up} of {required} tins did not fall. "
                    f"Knock down the whole tower"
                )
            self.message_color = DANGER
        self.total_score += self.throw_score
        self.phase = Phase.ROUND_END

    def update_physics(self, dt: float) -> None:
        steps = 3
        for _ in range(steps):
            self.space.step(dt / steps)

    def update(self, dt: float) -> None:
        if self.phase in (Phase.STACKING, Phase.STACK_SETTLING, Phase.AIMING, Phase.BALL_ACTIVE, Phase.RESOLVING):
            self.update_physics(dt)

        if self.phase == Phase.STACKING and self.tins_remaining > 0:
            mx, my = self._window_to_game(pygame.mouse.get_pos())
            gx, gy = self.ghost_x, self.ghost_y
            if mx < STACK_RIGHT:
                gx = float(mx)
            gy = float(my)
            keys = pygame.key.get_pressed()
            step = GHOST_MOVE_SPEED * dt
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                gy -= step
            if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                gy += step
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                gx -= step
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                gx += step
            self.ghost_x, self.ghost_y = clamp_ghost_pos(gx, gy, self.tin_w)

        moving = False
        for body in self.tin_bodies:
            if body.velocity.length > SETTLE_SPEED or abs(body.angular_velocity) > 1.2:
                moving = True
                break
        if self.ball_body.body_type == pymunk.Body.DYNAMIC:
            if self.ball_body.velocity.length > SETTLE_SPEED:
                moving = True

        if self.phase == Phase.STACK_SETTLING and self.tins_remaining == 0:
            if not moving:
                self.settle_timer += dt
            else:
                self.settle_timer = 0.0
                self._stack_settle_checked = False
                self.top_prompt = ""
            settled = not moving and self.settle_timer >= SETTLE_TIME * 0.5
            timed_out = self.settle_timer >= 5.0
            if (settled or timed_out) and not self._stack_settle_checked:
                self._stack_settle_checked = True
                if self._has_valid_vertical_tower():
                    self.top_prompt = THROW_START_PROMPT
                else:
                    if not stack_is_stable(self.tin_bodies):
                        self._fail_invalid_tower(
                            "All tins must form a stable vertical tower on the first tin"
                        )
                    else:
                        self._fail_invalid_tower()

        if self.phase in (Phase.BALL_ACTIVE, Phase.RESOLVING) and not moving:
            self.settle_timer += dt
            if self.settle_timer >= SETTLE_TIME:
                self.phase = Phase.RESOLVING
                self.resolve_throw()
        elif self.phase == Phase.BALL_ACTIVE and moving:
            self.settle_timer = 0.0

    def draw_panel_bg(self) -> None:
        self.screen.fill(BG)
        pygame.draw.rect(self.screen, STACK_PANEL, (0, 0, STACK_RIGHT, HEIGHT))
        pygame.draw.rect(self.screen, LAUNCH_PANEL, (STACK_RIGHT, 0, WIDTH - STACK_RIGHT, HEIGHT))
        # Visual guide only (not a physics wall)
        for y in range(0, HEIGHT, 14):
            pygame.draw.line(
                self.screen,
                (70, 78, 100),
                (STACK_RIGHT, y),
                (STACK_RIGHT, min(y + 7, HEIGHT)),
                2,
            )
        # platform visual
        pygame.draw.rect(
            self.screen,
            (90, 98, 118),
            (PLATFORM_X, PLATFORM_Y, PLATFORM_W, PLATFORM_H),
            border_radius=4,
        )
        pygame.draw.rect(
            self.screen,
            (120, 128, 148),
            (PLATFORM_X, PLATFORM_Y - 4, PLATFORM_W, 6),
            border_radius=2,
        )

    def draw_hud(self) -> None:
        title = self.font_lg.render("Stack & Smash", True, ACCENT)
        self.screen.blit(title, (24, 18))
        level_txt = self.font.render(
            f"Level {self.level}/{LEVEL_MAX}  ·  {self.tin_count()} tins",
            True,
            ACCENT,
        )
        self.screen.blit(level_txt, (24, 56))
        score = self.font.render(
            f"Stack: {self.stack_score}   Throw: {self.throw_score}   Total: {self.total_score}",
            True,
            MUTED,
        )
        self.screen.blit(score, (24, 88))

        if self.phase == Phase.STACKING:
            help_lines = [
                "Move: mouse / WASD",
                "SPACE — drop",
                f"Left: {self.tins_remaining}/{self.tin_count()}",
            ]
        elif self.phase in (Phase.STACK_SETTLING,):
            if self.top_prompt:
                help_lines = ["ENTER — start throw", "R — new game"]
            else:
                help_lines = ["Stack settling…", "R — new game"]
        elif self.phase == Phase.AIMING:
            help_lines = [
                "UP/DOWN — aim",
                "ENTER — throw",
                f"Vertical tower on 1st tin · knock ALL {self.tin_count()} down",
            ]
        elif self.phase == Phase.ROUND_END:
            if self.throw_result == "win" and self.level < LEVEL_MAX:
                help_lines = ["ENTER — next level", "R — new game"]
            elif self.throw_result == "win":
                help_lines = ["ENTER — replay", "R — new game"]
            else:
                help_lines = ["ENTER — retry", "R — new game"]
        else:
            help_lines = ["R — new game"]

        y = HEIGHT - 120
        for line in help_lines:
            surf = self.font_sm.render(line, True, MUTED)
            self.screen.blit(surf, (24, y))
            y += 24

        if self.top_prompt:
            surf = self.font.render(self.top_prompt, True, TEXT)
            self.screen.blit(surf, surf.get_rect(center=(WIDTH // 2, 36)))

        if self.phase == Phase.ROUND_END and self.throw_result:
            if self.throw_result == "win" and self.level >= LEVEL_MAX:
                banner = "GAME COMPLETE!"
            elif self.throw_result == "win":
                banner = f"LEVEL {self.level} COMPLETE"
            else:
                banner = "YOU LOSE"
            color = SUCCESS if self.throw_result == "win" else DANGER
            surf = self.font_lg.render(banner, True, color)
            self.screen.blit(surf, surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 50)))
            if self.throw_result == "lose" and self.lose_detail:
                detail = self.font.render(self.lose_detail, True, color)
                self.screen.blit(detail, detail.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 10)))

    def draw_ghost_tin(self) -> None:
        draw_tin_sprite(
            self.screen,
            self.tin_sprites[self.ghost_variant],
            (self.ghost_x, self.ghost_y),
            0.0,
            alpha=140,
            scale=self.tin_scale,
        )

    def draw(self) -> None:
        self.draw_panel_bg()
        for body, vid in zip(self.tin_bodies, self.tin_variant_ids):
            draw_tin_sprite(
                self.screen,
                self.tin_sprites[vid],
                (body.position.x, body.position.y),
                body.angle,
                scale=self.tin_scale,
            )
        if self.phase == Phase.STACKING and self.tins_remaining > 0:
            self.draw_ghost_tin()
        if self.phase in (Phase.AIMING, Phase.BALL_ACTIVE, Phase.RESOLVING, Phase.ROUND_END):
            draw_ball(self.screen, self.ball_body)
        if self.phase == Phase.AIMING:
            draw_aim_arrow(self.screen, self.ball_body.position, self.aim_pitch)
        self.draw_hud()
        self._present()

    def advance_after_round(self) -> None:
        if self.throw_result == "win":
            if self.level < LEVEL_MAX:
                self.level += 1
            self._score_at_level_start = self.total_score
        else:
            self.total_score = self._score_at_level_start
        self.start_round()

    def handle_key(self, key: int) -> None:
        if key == pygame.K_r:
            self.reset_game()
            return

        if key == pygame.K_SPACE and self.phase == Phase.STACKING:
            self.drop_ghost_tin()

        if key == pygame.K_RETURN:
            if self.phase == Phase.ROUND_END:
                self.advance_after_round()
            elif self.phase == Phase.STACK_SETTLING and self.tins_remaining == 0:
                if self._has_valid_vertical_tower():
                    self.lock_stack()
                else:
                    self._fail_invalid_tower()
            elif self.phase == Phase.AIMING:
                self.launch_ball()

        if self.phase == Phase.AIMING:
            step = math.radians(3)
            if key in (pygame.K_UP, pygame.K_w):
                self.aim_pitch = max(-MAX_PITCH, self.aim_pitch - step)
            if key in (pygame.K_DOWN, pygame.K_s):
                self.aim_pitch = min(MAX_PITCH, self.aim_pitch + step)

    def run(self) -> None:
        while True:
            dt = self.clock.tick(FPS) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                if event.type == pygame.VIDEORESIZE:
                    w = getattr(event, "w", None) or getattr(event, "x", WIDTH)
                    h = getattr(event, "h", None) or getattr(event, "y", HEIGHT)
                    if hasattr(event, "size"):
                        w, h = event.size
                    self._on_window_resize(int(w), int(h))
                if event.type == pygame.KEYDOWN:
                    self.handle_key(event.key)
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.phase == Phase.STACKING:
                        self.drop_ghost_tin()
            self.update(dt)
            self.draw()


if __name__ == "__main__":
    Game().run()