"""Headless Stack & Smash simulation (physics, rules, levels). Used by main.py and RL env."""

from __future__ import annotations

import math
from enum import Enum, auto

import pymunk
from pymunk import Vec2d

# --- world layout ---
WIDTH, HEIGHT = 1200, 700
STACK_RIGHT = 760
PLATFORM_Y = HEIGHT - 90
PLATFORM_W = 420
PLATFORM_H = 18
PLATFORM_X = (STACK_RIGHT - PLATFORM_W) // 2 + 40

BASE_TIN_W, BASE_TIN_H = 40, 68
TIN_SCALE_MIN = 0.36
LEVEL_MIN = 1
LEVEL_MAX = 12
TINS_AT_LEVEL_1 = 6
GHOST_Y_MIN = 100
GHOST_Y_MAX = PLATFORM_Y - 24
GHOST_MOVE_SPEED = 320
STACK_HEIGHT_BUDGET = PLATFORM_Y - GHOST_Y_MIN - 36

TIN_MASS = 2.5
TIN_FRICTION = 0.85
TIN_ELASTICITY = 0.08

BALL_RADIUS = 22
BALL_MASS = 8.0
LAUNCH_X = WIDTH - 120
LAUNCH_Y = HEIGHT // 2 + 40
AIM_LEFT = math.pi
MAX_PITCH = math.radians(55)
DEFAULT_PITCH = -math.radians(12)
THROW_SPEED = 2200

PHYSICS_DT = 1.0 / 60.0
PHYSICS_SUBSTEPS = 3
SETTLE_SPEED = 12.0
SETTLE_TIME = 1.2
KNOCK_ANGLE = math.radians(28)

MAX_TINS = TINS_AT_LEVEL_1 + LEVEL_MAX - 1


def tins_for_level(level: int) -> int:
    level = max(LEVEL_MIN, min(LEVEL_MAX, level))
    return TINS_AT_LEVEL_1 + level - 1


def tin_scale_for_level(level: int) -> float:
    n = tins_for_level(level)
    stacked_height = BASE_TIN_H * n * 0.9
    scale = STACK_HEIGHT_BUDGET / stacked_height
    return max(TIN_SCALE_MIN, min(1.0, scale))


class Phase(Enum):
    STACKING = 0
    STACK_SETTLING = 1
    AIMING = 2
    BALL_ACTIVE = 3
    RESOLVING = 4
    ROUND_END = 5


def make_space() -> pymunk.Space:
    space = pymunk.Space()
    space.gravity = (0, 1400)
    return space


def add_platform(space: pymunk.Space) -> pymunk.Shape:
    body = space.static_body
    y = PLATFORM_Y + PLATFORM_H / 2
    shape = pymunk.Segment(
        body,
        (PLATFORM_X, y),
        (PLATFORM_X + PLATFORM_W, y),
        PLATFORM_H / 2,
    )
    shape.friction = 1.0
    shape.elasticity = 0.05
    space.add(shape)
    return shape


def add_walls(space: pymunk.Space) -> None:
    body = space.static_body
    floor_y = HEIGHT + 40
    left_wall = pymunk.Segment(body, (20, 0), (20, floor_y), 8)
    right_wall = pymunk.Segment(body, (WIDTH - 20, 0), (WIDTH - 20, floor_y), 8)
    ground_y = HEIGHT - 24
    ground = pymunk.Segment(body, (20, ground_y), (WIDTH - 20, ground_y), 10)
    for seg in (left_wall, right_wall, ground):
        seg.friction = 0.7
        seg.elasticity = 0.1
        space.add(seg)


def make_tin(
    space: pymunk.Space,
    x: float,
    y: float,
    tin_w: float,
    tin_h: float,
) -> tuple[pymunk.Body, pymunk.Poly]:
    area_scale = (tin_w * tin_h) / (BASE_TIN_W * BASE_TIN_H)
    mass = TIN_MASS * area_scale
    moment = pymunk.moment_for_box(mass, (tin_w, tin_h))
    body = pymunk.Body(mass, moment)
    body.position = x, y
    radius = max(1.0, 3 * tin_w / BASE_TIN_W)
    shape = pymunk.Poly.create_box(body, (tin_w, tin_h), radius=radius)
    shape.friction = TIN_FRICTION
    shape.elasticity = TIN_ELASTICITY
    space.add(body, shape)
    return body, shape


def make_ball(space: pymunk.Space) -> tuple[pymunk.Body, pymunk.Circle]:
    moment = pymunk.moment_for_circle(BALL_MASS, 0, BALL_RADIUS)
    body = pymunk.Body(BALL_MASS, moment, body_type=pymunk.Body.KINEMATIC)
    body.position = LAUNCH_X, LAUNCH_Y
    shape = pymunk.Circle(body, BALL_RADIUS)
    shape.friction = 0.4
    shape.elasticity = 0.65
    space.add(body, shape)
    return body, shape


def aim_angle_from_pitch(pitch: float) -> float:
    return AIM_LEFT - pitch


# Scoring (human + RL): longer tower → more stack pts; each tin that falls → throw pts
POINTS_PER_TIN_HEIGHT = 8
POINTS_PER_TIN_STACKED = 4
POINTS_PER_TIN_FALLEN = 10

# Horizontal alignment with the first (base) tin
TOWER_X_BAND = 0.55


def tin_on_platform(body: pymunk.Body, tin_w: float, tin_h: float) -> bool:
    """On the platform deck and not already fallen."""
    if tin_is_knocked(body, tin_h):
        return False
    cx, cy = body.position
    if cx < PLATFORM_X - tin_w * 0.25 or cx > PLATFORM_X + PLATFORM_W + tin_w * 0.25:
        return False
    if cy > PLATFORM_Y + tin_h * 0.45:
        return False
    return True


def vertical_tower_from_first_tin(
    tin_bodies: list[pymunk.Body], tin_w: float, tin_h: float
) -> tuple[list[pymunk.Body], float, float, float]:
    """
    Tower = vertical column above the first placed tin (base).
    Tins beside the base or fallen off are not in the tower.

    Returns (tower_tins, base_x, height_px, horizontal_spread_px).
    """
    if not tin_bodies:
        return [], PLATFORM_X + PLATFORM_W / 2, 0.0, 0.0

    base = tin_bodies[0]
    base_x = base.position.x
    base_y = base.position.y
    x_band = tin_w * TOWER_X_BAND
    min_lift = tin_h * 0.22

    tower: list[pymunk.Body] = []
    if tin_on_platform(base, tin_w, tin_h):
        tower.append(base)

    for body in tin_bodies[1:]:
        if not tin_on_platform(body, tin_w, tin_h):
            continue
        if abs(body.position.x - base_x) > x_band:
            continue
        if body.position.y < base_y - min_lift:
            tower.append(body)

    if not tower:
        return [], base_x, 0.0, 0.0

    height = max_stack_height(tower)
    spread = (
        max(b.position.x for b in tower) - min(b.position.x for b in tower)
        if len(tower) > 1
        else 0.0
    )
    return tower, base_x, height, spread


def tower_tins_at_lock(
    tin_bodies: list[pymunk.Body], tin_w: float, tin_h: float
) -> tuple[list[pymunk.Body], float, float, float]:
    return vertical_tower_from_first_tin(tin_bodies, tin_w, tin_h)


def score_stack(tower_count: int, tower_height_px: float, tin_h: float) -> int:
    """Longer / taller tower earns more (stack phase)."""
    height_units = tower_height_px / max(tin_h, 1.0)
    return int(height_units * POINTS_PER_TIN_HEIGHT + tower_count * POINTS_PER_TIN_STACKED)


def score_throw(knocked_count: int) -> int:
    """More tins knocked down = more points (throw phase)."""
    return knocked_count * POINTS_PER_TIN_FALLEN


def all_level_tins_on_tower(
    level_tin_count: int, placed_count: int, tower_count: int
) -> bool:
    """Every level tin was placed and is in the vertical tower on the first tin."""
    return placed_count == level_tin_count and tower_count == level_tin_count


def level_complete(
    level_tin_count: int,
    tower_count: int,
    knocked_count: int,
    placed_count: int,
) -> bool:
    """Win the level: all level tins on the tower, and all of them fell."""
    return all_level_tins_on_tower(level_tin_count, placed_count, tower_count) and (
        knocked_count == level_tin_count
    )


def throw_is_win(tower_count: int, knocked_count: int) -> bool:
    """Alias when tower already equals level size."""
    return tower_count > 0 and knocked_count == tower_count


def tin_is_on_tower(body: pymunk.Body, tin_w: float, tin_h: float) -> bool:
    return tin_on_platform(body, tin_w, tin_h)


def tin_is_knocked(body: pymunk.Body, tin_h: float) -> bool:
    angle = abs(body.angle % math.pi)
    if angle > math.pi / 2:
        angle = math.pi - angle
    if angle > KNOCK_ANGLE:
        return True
    cx, cy = body.position
    if cy > PLATFORM_Y + tin_h * 0.6:
        return True
    if cx < PLATFORM_X - 30 or cx > PLATFORM_X + PLATFORM_W + 30:
        return True
    return False


def stack_is_stable(tins: list[pymunk.Body]) -> bool:
    if not tins:
        return False
    for body in tins:
        if body.velocity.length > SETTLE_SPEED or abs(body.angular_velocity) > 1.5:
            return False
        cx, cy = body.position
        if cy > PLATFORM_Y + 8:
            return False
        if cx < PLATFORM_X - 10 or cx > PLATFORM_X + PLATFORM_W + 10:
            return False
    return True


def max_stack_height(tins: list[pymunk.Body]) -> float:
    if not tins:
        return 0.0
    top = min(body.position.y for body in tins)
    return PLATFORM_Y - top


def clamp_ghost_pos(x: float, y: float, tin_w: float) -> tuple[float, float]:
    x = max(PLATFORM_X + tin_w / 2, min(PLATFORM_X + PLATFORM_W - tin_w / 2, x))
    y = max(GHOST_Y_MIN, min(GHOST_Y_MAX, y))
    return x, y


def norm_ghost_x(x: float, tin_w: float) -> float:
    lo = PLATFORM_X + tin_w / 2
    hi = PLATFORM_X + PLATFORM_W - tin_w / 2
    return (x - lo) / max(hi - lo, 1.0)


def denorm_ghost_x(t: float, tin_w: float) -> float:
    lo = PLATFORM_X + tin_w / 2
    hi = PLATFORM_X + PLATFORM_W - tin_w / 2
    return lo + max(0.0, min(1.0, t)) * (hi - lo)


def norm_ghost_y(y: float) -> float:
    return (y - GHOST_Y_MIN) / max(GHOST_Y_MAX - GHOST_Y_MIN, 1.0)


def denorm_ghost_y(t: float) -> float:
    return GHOST_Y_MIN + max(0.0, min(1.0, t)) * (GHOST_Y_MAX - GHOST_Y_MIN)


def norm_pitch(pitch: float) -> float:
    return (pitch + MAX_PITCH) / (2.0 * MAX_PITCH)


def denorm_pitch(t: float) -> float:
    return -MAX_PITCH + max(0.0, min(1.0, t)) * (2.0 * MAX_PITCH)


class StackSmashCore:
    """Headless game state machine + pymunk world."""

    def __init__(self, level: int = LEVEL_MIN) -> None:
        self.level = max(LEVEL_MIN, min(LEVEL_MAX, level))
        self.throw_result: str | None = None
        self.knocked_count = 0
        self.still_up_count = 0
        self.tower_tins: list[pymunk.Body] = []
        self.tower_tin_count = 0
        self.off_tower_tin_count = 0
        self.tower_center_x = PLATFORM_X + PLATFORM_W / 2
        self.tower_height = 0.0
        self.tower_spread_x = 0.0
        self.stack_score = 0
        self.throw_score = 0
        self.settle_timer = 0.0
        self.stack_locked_unstable = False
        self._ball_moment = pymunk.moment_for_circle(BALL_MASS, 0, BALL_RADIUS)
        self.reset()

    def tin_count(self) -> int:
        return tins_for_level(self.level)

    def reset(self, level: int | None = None) -> None:
        if level is not None:
            self.level = max(LEVEL_MIN, min(LEVEL_MAX, level))
        self.space = make_space()
        add_platform(self.space)
        add_walls(self.space)
        self.tin_bodies: list[pymunk.Body] = []
        self.tin_shapes: list[pymunk.Poly] = []
        self.ball_body, self.ball_shape = make_ball(self.space)
        self.phase = Phase.STACKING
        self.tins_remaining = self.tin_count()
        self.level_tin_count = self.tin_count()
        self.tin_scale = tin_scale_for_level(self.level)
        self.tin_w = BASE_TIN_W * self.tin_scale
        self.tin_h = BASE_TIN_H * self.tin_scale
        self.ghost_x = PLATFORM_X + PLATFORM_W / 2
        self.ghost_y = PLATFORM_Y - BASE_TIN_H * self.tin_scale * 1.1
        self.aim_pitch = DEFAULT_PITCH
        self.throw_result = None
        self.knocked_count = 0
        self.still_up_count = 0
        self.tower_tins = []
        self.tower_tin_count = 0
        self.off_tower_tin_count = 0
        self.tower_center_x = PLATFORM_X + PLATFORM_W / 2
        self.tower_height = 0.0
        self.tower_spread_x = 0.0
        self.stack_score = 0
        self.throw_score = 0
        self.settle_timer = 0.0
        self.stack_locked_unstable = False

    def stack_target_x(self) -> float:
        """Drop on the first tin — the base of the vertical tower."""
        if self.tin_bodies:
            return self.tin_bodies[0].position.x
        return PLATFORM_X + PLATFORM_W / 2

    def current_stack_height(self) -> float:
        tower, _, height, _ = vertical_tower_from_first_tin(
            self.tin_bodies, self.tin_w, self.tin_h
        )
        return height

    def refresh_tower_tins(self) -> None:
        """Snapshot vertical tower above first tin (for throw scoring)."""
        tower, cx, height, spread = vertical_tower_from_first_tin(
            self.tin_bodies, self.tin_w, self.tin_h
        )
        self.tower_tins = tower
        self.tower_tin_count = len(tower)
        self.level_tin_count = self.tin_count()
        self.tower_center_x = cx
        self.tower_height = height
        self.tower_spread_x = spread
        self.off_tower_tin_count = len(self.tin_bodies) - self.tower_tin_count

    def step_physics(self, dt: float = PHYSICS_DT) -> None:
        for _ in range(PHYSICS_SUBSTEPS):
            self.space.step(dt / PHYSICS_SUBSTEPS)

    def bodies_moving(self) -> bool:
        for body in self.tin_bodies:
            if body.velocity.length > SETTLE_SPEED or abs(body.angular_velocity) > 1.2:
                return True
        if self.ball_body.body_type == pymunk.Body.DYNAMIC:
            if self.ball_body.velocity.length > SETTLE_SPEED:
                return True
        return False

    def set_ghost_from_norm(self, x_norm: float, y_norm: float) -> None:
        x = denorm_ghost_x(x_norm, self.tin_w)
        y = denorm_ghost_y(y_norm)
        self.ghost_x, self.ghost_y = clamp_ghost_pos(x, y, self.tin_w)

    def drop_tin(self) -> bool:
        if self.phase != Phase.STACKING or self.tins_remaining <= 0:
            return False
        body, shape = make_tin(self.space, self.ghost_x, self.ghost_y, self.tin_w, self.tin_h)
        self.tin_bodies.append(body)
        self.tin_shapes.append(shape)
        self.tins_remaining -= 1
        if self.tins_remaining == 0:
            self.phase = Phase.STACK_SETTLING
            self.settle_timer = 0.0
        return True

    def try_lock_stack(self, *, force: bool = False) -> bool:
        if self.phase not in (Phase.STACK_SETTLING, Phase.STACKING) or self.tins_remaining > 0:
            return False
        stable = stack_is_stable(self.tin_bodies)
        if not stable and not force:
            return False
        self.refresh_tower_tins()
        self.stack_score = score_stack(
            self.tower_tin_count, self.tower_height, self.tin_h
        )
        self.phase = Phase.AIMING
        self.aim_pitch = DEFAULT_PITCH
        self.stack_locked_unstable = not stable
        return True

    def set_aim_from_norm(self, pitch_norm: float) -> None:
        """Map Gym action in [-1, 1] to pitch (DEFAULT_PITCH near 0)."""
        t = (float(pitch_norm) + 1.0) * 0.5
        self.aim_pitch = denorm_pitch(t)

    def launch_ball(self) -> bool:
        if self.phase != Phase.AIMING:
            return False
        self.ball_body.body_type = pymunk.Body.DYNAMIC
        self.ball_body.mass = BALL_MASS
        self.ball_body.moment = self._ball_moment
        angle = aim_angle_from_pitch(self.aim_pitch)
        direction = Vec2d(math.cos(angle), math.sin(angle))
        self.ball_body.velocity = direction * THROW_SPEED
        self.phase = Phase.BALL_ACTIVE
        self.settle_timer = 0.0
        return True

    def resolve_throw(self) -> tuple[float, bool]:
        """Level complete only if all level tins were on the tower and all fell."""
        tower = self.tower_tins
        knocked = sum(1 for b in tower if tin_is_knocked(b, self.tin_h))
        n_tower = len(tower)
        level_n = self.tin_count()
        placed = len(self.tin_bodies)
        self.knocked_count = knocked
        self.still_up_count = n_tower - knocked
        self.off_tower_tin_count = placed - n_tower
        self.throw_score = score_throw(knocked)
        won = level_complete(level_n, n_tower, knocked, placed)
        if won:
            self.throw_result = "win"
            self.phase = Phase.ROUND_END
            return float(self.throw_score), True
        self.throw_result = "lose"
        self.phase = Phase.ROUND_END
        return float(self.throw_score), False

    def advance_settling(self, dt: float = PHYSICS_DT) -> None:
        if self.phase == Phase.STACK_SETTLING:
            if not self.bodies_moving():
                self.settle_timer += dt
                if self.settle_timer >= SETTLE_TIME * 0.35 and stack_is_stable(self.tin_bodies):
                    self.try_lock_stack()
                elif self.settle_timer >= SETTLE_TIME * 1.25:
                    # RL must auto-advance; unstable stacks still get a throw attempt
                    self.try_lock_stack(force=True)
            else:
                self.settle_timer = 0.0
        elif self.phase == Phase.BALL_ACTIVE:
            if not self.bodies_moving():
                self.settle_timer += dt
                if self.settle_timer >= SETTLE_TIME * 0.65:
                    self.phase = Phase.RESOLVING
            else:
                self.settle_timer = 0.0
        elif self.phase == Phase.RESOLVING:
            self.resolve_throw()

    def observation_vector(self) -> list[float]:
        """Fixed-size float observation for RL."""
        target_x = self.stack_target_x()
        align_err = abs(self.ghost_x - target_x) / max(self.tin_w * 2.5, 1.0)
        stack_h = self.current_stack_height()
        obs: list[float] = [
            self.phase.value / 5.0,
            self.level / LEVEL_MAX,
            self.tins_remaining / MAX_TINS,
            norm_ghost_x(self.ghost_x, self.tin_w),
            norm_ghost_y(self.ghost_y),
            norm_pitch(self.aim_pitch),
            1.0 if self.ball_body.body_type == pymunk.Body.DYNAMIC else 0.0,
            max(0.0, 1.0 - min(1.0, align_err)),
            min(1.0, stack_h / max(STACK_HEIGHT_BUDGET, 1.0)),
            min(1.0, self.tower_height / max(STACK_HEIGHT_BUDGET, 1.0)),
        ]
        tower_set = set(self.tower_tins)
        for i in range(MAX_TINS):
            if i < len(self.tin_bodies):
                b = self.tin_bodies[i]
                obs.extend(
                    [
                        norm_ghost_x(b.position.x, self.tin_w),
                        norm_ghost_y(b.position.y),
                        math.sin(b.angle),
                        math.cos(b.angle),
                        1.0 if tin_is_knocked(b, self.tin_h) else 0.0,
                        1.0 if b in tower_set else 0.0,
                    ]
                )
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        return obs

    @property
    def observation_size(self) -> int:
        return 10 + MAX_TINS * 6