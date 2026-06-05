"""
Gymnasium environment for Stack & Smash.

Rules: stack ALL tins vertically on the first tin → throw → knock ALL down.
  Fail if any tin is not in that column; win only if the full tower falls.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from game_core import (
    DEFAULT_PITCH,
    GHOST_Y_MAX,
    GHOST_Y_MIN,
    MAX_PITCH,
    PHYSICS_DT,
    PLATFORM_Y,
    Phase,
    StackSmashCore,
    all_level_tins_on_tower,
    clamp_ghost_pos,
    level_complete,
    score_stack,
    score_throw,
)


class StackSmashEnv(gym.Env):
    """Gymnasium wrapper around StackSmashCore."""

    metadata = {"render_modes": [], "render_fps": 60}

    def __init__(
        self,
        level: int = 1,
        max_stack_steps: int = 200,
        max_aim_steps: int = 50,
        max_settle_steps: int = 300,
        max_ball_steps: int = 400,
        render_mode: str | None = None,
        fast_mode: bool = False,
    ) -> None:
        super().__init__()
        self.level = level
        self.fast_mode = fast_mode
        self.max_stack_steps = 200 if fast_mode else max_stack_steps
        self.max_aim_steps = 40 if fast_mode else max_aim_steps
        self.max_settle_steps = 180 if fast_mode else max_settle_steps
        self.max_ball_steps = 220 if fast_mode else max_ball_steps
        self._settle_substeps = 10 if fast_mode else 6
        self._ball_substeps = 12 if fast_mode else 8
        self.render_mode = render_mode

        self.core = StackSmashCore(level=level)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.core.observation_size,),
            dtype=np.float32,
        )

        self._stack_steps = 0
        self._aim_steps = 0
        self._settle_steps = 0
        self._ball_steps = 0
        self._steps_since_drop = 0

    def _stack_drop_position(self) -> tuple[float, float]:
        """Ghost over the tower base, above the current stack (human raises ghost to stack up)."""
        tx = self.core.stack_target_x()
        stack_h = self.core.current_stack_height()
        if stack_h > 1.0:
            drop_y = PLATFORM_Y - stack_h - self.core.tin_h * 0.55
        else:
            drop_y = PLATFORM_Y - self.core.tin_h * 1.1
        drop_y = max(GHOST_Y_MIN, min(GHOST_Y_MAX, drop_y))
        x, y = clamp_ghost_pos(tx, drop_y, self.core.tin_w)
        return x, y

    def _snap_ghost_to_stack_target(self) -> None:
        self.core.ghost_x, self.core.ghost_y = self._stack_drop_position()

    def _obs(self) -> np.ndarray:
        return np.asarray(self.core.observation_vector(), dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        level = self.level
        if options and "level" in options:
            level = int(options["level"])
        self.core.reset(level=level)
        self._stack_steps = 0
        self._aim_steps = 0
        self._settle_steps = 0
        self._ball_steps = 0
        self._steps_since_drop = 0
        return self._obs(), {"phase": self.core.phase.name, "level": self.core.level}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {"phase": self.core.phase.name}

        phase = self.core.phase

        if phase == Phase.STACKING:
            self._stack_steps += 1
            self._snap_ghost_to_stack_target()
            dropped = False
            if float(action[2]) > 0.15:
                if self.core.drop_tin():
                    target_x = self.core.stack_target_x()
                    align = max(
                        0.0,
                        1.0
                        - abs(self.core.ghost_x - target_x)
                        / max(self.core.tin_w * 2.0, 1.0),
                    )
                    reward += 0.5 + align * 1.2
                    reward += 0.25 * len(self.core.tin_bodies)
                    dropped = True
                else:
                    reward -= 0.1
            else:
                reward -= 0.01
            self._steps_since_drop = 0 if dropped else self._steps_since_drop + 1
            auto_after = 18 if self.fast_mode else 28
            if self._steps_since_drop >= auto_after and self.core.tins_remaining > 0:
                self._snap_ghost_to_stack_target()
                if self.core.drop_tin():
                    reward += 0.15
                    info["auto_drop"] = True
                    self._steps_since_drop = 0
            self.core.step_physics(PHYSICS_DT)
            if self._stack_steps >= self.max_stack_steps and self.core.tins_remaining > 0:
                while self.core.tins_remaining > 0:
                    self._snap_ghost_to_stack_target()
                    self.core.drop_tin()
                info["force_finish_stack"] = True
            if self._stack_steps >= self.max_stack_steps and self.core.tins_remaining > 0:
                truncated = True
                reward -= 12.0 - 1.5 * self.core.tins_remaining
                info["trunc_reason"] = "stack_timeout"

        elif phase == Phase.STACK_SETTLING:
            self._settle_steps += 1
            prev_phase = self.core.phase
            for _ in range(self._settle_substeps):
                self.core.step_physics(PHYSICS_DT)
                self.core.advance_settling(PHYSICS_DT)
            if self.core.phase == Phase.AIMING and prev_phase == Phase.STACK_SETTLING:
                level_n = self.core.tin_count()
                placed = len(self.core.tin_bodies)
                tower_n = self.core.tower_tin_count
                if not all_level_tins_on_tower(level_n, placed, tower_n):
                    truncated = True
                    reward -= 6.0
                    reward += tower_n * 0.5
                    info["trunc_reason"] = "incomplete_tower"
                    info["win"] = False
                    info["tower_tins"] = tower_n
                    info["level_tins"] = level_n
                else:
                    pts = score_stack(tower_n, self.core.tower_height, self.core.tin_h)
                    reward += float(pts) * 0.1
                    info["stack_points"] = pts
                    info["tower_tins"] = tower_n
                    info["tower_height"] = round(self.core.tower_height, 1)
            if self._settle_steps >= self.max_settle_steps:
                truncated = True
                info["trunc_reason"] = "settle_timeout"

        elif phase == Phase.AIMING:
            self._aim_steps += 1
            # Fine-tune around the human default aim (action in [-1, 1] → ±~25°)
            delta = float(action[0]) * MAX_PITCH * 0.45
            self.core.aim_pitch = max(-MAX_PITCH, min(MAX_PITCH, DEFAULT_PITCH + delta))
            reward += 0.03 * (1.0 - min(1.0, abs(float(action[0]))))
            if float(action[1]) > 0.15 and self.core.launch_ball():
                reward += 0.2
            if self._aim_steps >= self.max_aim_steps and self.core.phase == Phase.AIMING:
                self.core.aim_pitch = DEFAULT_PITCH
                self.core.launch_ball()
                info["auto_throw"] = True

        elif phase in (Phase.BALL_ACTIVE, Phase.RESOLVING):
            self._ball_steps += 1
            for _ in range(self._ball_substeps):
                self.core.step_physics(PHYSICS_DT)
                self.core.advance_settling(PHYSICS_DT)
            if self._ball_steps >= self.max_ball_steps and self.core.phase != Phase.ROUND_END:
                self.core.phase = Phase.RESOLVING
                self.core.resolve_throw()
            if self.core.phase == Phase.ROUND_END:
                level_n = self.core.tin_count()
                placed = len(self.core.tin_bodies)
                knocked = self.core.knocked_count
                tower_n = self.core.tower_tin_count
                throw_pts = score_throw(knocked)
                reward += float(throw_pts) * 0.35
                reward += float(knocked) * 1.5
                if knocked == 0:
                    reward -= 8.0
                info["throw_points"] = throw_pts
                info["knocked"] = knocked
                info["tower_total"] = tower_n
                info["level_tins"] = level_n
                info["still_up"] = level_n - knocked
                terminated = True
                if level_complete(level_n, tower_n, knocked, placed):
                    reward += 25.0
                    info["win"] = True
                else:
                    reward -= 2.0 * max(0, level_n - knocked)
                    info["win"] = False

        elif phase == Phase.ROUND_END:
            terminated = True
            info["win"] = self.core.throw_result == "win"

        info["phase"] = self.core.phase.name
        info["tins_remaining"] = self.core.tins_remaining
        return self._obs(), reward, terminated, truncated, info

    def close(self) -> None:
        pass


try:
    gym.register(
        id="StackSmash-v0",
        entry_point="env:StackSmashEnv",
    )
except gym.error.Error:
    pass