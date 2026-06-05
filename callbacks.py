"""Stable-Baselines3 callbacks: optional live demo + learning curve."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from game_core import PHYSICS_DT, Phase
from env import StackSmashEnv


class TrainingVizCallback(BaseCallback):
    """
    Training metrics + optional visualization.

    Default (fast): only writes logs/training_curve.png — no pygame, no live graph.
    --watch: periodic or continuous pygame demos.
    --plot: live matplotlib window (slower).
    """

    def __init__(
        self,
        level: int = 1,
        render_every: int = 8192,
        plot_path: Path | None = None,
        verbose: int = 1,
        colab_mode: bool = False,
        live_demo: bool = False,
        live_plot: bool = False,
        continuous_demo: bool = False,
        result_pause_sec: float = 1.5,
        demo_steps_per_train_step: int = 4,
    ) -> None:
        super().__init__(verbose)
        self.level = level
        self.colab_mode = colab_mode
        self.live_demo = live_demo and not colab_mode
        self.live_plot = live_plot or colab_mode
        self.continuous_demo = continuous_demo and self.live_demo
        self.result_pause_sec = result_pause_sec
        self.demo_steps_per_train_step = max(1, demo_steps_per_train_step)
        self.render_every = max(512, render_every)
        self.plot_path = plot_path or Path("logs/training_curve.png")
        self.plot_path.parent.mkdir(parents=True, exist_ok=True)

        self.timesteps: list[int] = []
        self.mean_rewards: list[float] = []
        self.win_rates: list[float] = []

        self._viewer: Any = None
        self._demo_env: StackSmashEnv | None = None
        self._fig: Any = None
        self._ax_reward: Any = None
        self._ax_win: Any = None
        self._last_render_step = -self.render_every

        self._episode_count = 0
        self._demo_obs: np.ndarray | None = None
        self._demo_done = True
        self._demo_info: dict[str, Any] = {}
        self._demo_total_r = 0.0
        self._demo_pause_frames = 0

    def _init_callback(self) -> None:
        if self.colab_mode:
            matplotlib.use("Agg")
            self.live_plot = True
        elif not self.live_plot:
            matplotlib.use("Agg")
        else:
            import matplotlib.pyplot as plt

            plt.ion()

        import matplotlib.pyplot as plt

        self._fig, (self._ax_reward, self._ax_win) = plt.subplots(2, 1, figsize=(8, 6))
        if self.live_plot and not self.colab_mode:
            try:
                self._fig.canvas.manager.set_window_title("Stack & Smash — Training progress")
            except Exception:
                pass

        if self.live_demo:
            self._demo_env = StackSmashEnv(level=self.level)
            self._demo_done = True
            if self.verbose:
                if self.continuous_demo:
                    print("[viz] Live demo: continuous episodes (training will be slower).")
                else:
                    print(f"[viz] Live demo: one episode every {self.render_every:,} steps.")

    def _update_plot(self) -> None:
        if not self.timesteps:
            return
        import matplotlib.pyplot as plt

        self._ax_reward.clear()
        self._ax_win.clear()
        self._ax_reward.plot(self.timesteps, self.mean_rewards, color="#ffc848", linewidth=2)
        self._ax_reward.set_ylabel("Mean episode reward")
        self._ax_reward.set_title("Learning curve")
        self._ax_reward.grid(True, alpha=0.3)

        self._ax_win.plot(self.timesteps, self.win_rates, color="#60dc8c", linewidth=2)
        self._ax_win.set_xlabel("Training timesteps")
        self._ax_win.set_ylabel("Win rate (eval, completed throws)")
        self._ax_win.set_ylim(-0.05, 1.05)
        self._ax_win.grid(True, alpha=0.3)

        self._fig.tight_layout()
        self._fig.savefig(self.plot_path, dpi=120)
        if self.colab_mode:
            try:
                from IPython.display import clear_output, display

                clear_output(wait=True)
                display(self._fig)
            except ImportError:
                pass
        elif self.live_plot:
            plt.pause(0.001)

    def _record_rollout_stats(self) -> None:
        if not hasattr(self.model, "ep_info_buffer") or len(self.model.ep_info_buffer) == 0:
            return
        rewards = [float(ep.get("r", 0)) for ep in self.model.ep_info_buffer]
        if not rewards:
            return
        self.timesteps.append(int(self.num_timesteps))
        self.mean_rewards.append(float(np.mean(rewards)))
        if len(self.win_rates) < len(self.mean_rewards):
            self.win_rates.append(self.win_rates[-1] if self.win_rates else 0.0)
        self._update_plot()

    def record_eval(self, mean_reward: float, win_rate: float) -> None:
        if self.timesteps and self.timesteps[-1] == int(self.num_timesteps):
            self.mean_rewards[-1] = mean_reward
            self.win_rates[-1] = win_rate
        else:
            self.timesteps.append(int(self.num_timesteps))
            self.mean_rewards.append(mean_reward)
            self.win_rates.append(win_rate)
        self._update_plot()

    def _ensure_viewer(self) -> bool:
        if not self.live_demo or self._demo_env is None:
            return False
        if self._viewer is None:
            from viewer import StackSmashViewer

            if self.verbose:
                print("[viz] Opening game window…")
            self._viewer = StackSmashViewer()
        return True

    def _demo_subtitle(self) -> str:
        return f"Episode {self._episode_count}  ·  training step {int(self.num_timesteps):,}"

    def _demo_extra(self, done: bool) -> str:
        if self._demo_pause_frames > 0:
            if self._demo_info.get("win"):
                return "WIN — next episode…"
            return (
                f"Retrying — {self._demo_info.get('still_up', '?')} of "
                f"{self._demo_info.get('tower_total', self._demo_info.get('total', '?'))} "
                f"tower tins did not fall"
            )
        if done:
            if self._demo_info.get("win"):
                return "WIN — all tins fell!"
            if self._demo_info.get("win") is False:
                return (
                    f"LOSE — {self._demo_info.get('still_up', '?')} of "
                    f"{self._demo_info.get('tower_total', '?')} tower tins did not fall"
                )
        return f"Phase: {self._demo_info.get('phase', '?')}"

    def _start_demo_episode(self) -> None:
        if self._demo_env is None:
            return
        self._episode_count += 1
        self._demo_obs, _ = self._demo_env.reset()
        self._demo_done = False
        self._demo_info = {"phase": self._demo_env.core.phase.name}
        self._demo_total_r = 0.0

    def _draw_demo_frame(self, env: StackSmashEnv, done: bool) -> bool:
        if self._viewer is None:
            return False
        if not self._viewer.handle_events():
            return False
        self._viewer.draw(
            env.core,
            subtitle=self._demo_subtitle(),
            extra=self._demo_extra(done),
        )
        self._viewer.tick(60)
        return True

    def _step_demo_physics(self, env: StackSmashEnv, done: bool) -> None:
        if env.core.phase not in (
            Phase.STACK_SETTLING,
            Phase.BALL_ACTIVE,
            Phase.RESOLVING,
        ):
            return
        for _ in range(3):
            if self._viewer is None or not self._viewer.handle_events():
                return
            env.core.step_physics(PHYSICS_DT)
            env.core.advance_settling(PHYSICS_DT)
            self._viewer.draw(
                env.core,
                subtitle=self._demo_subtitle(),
                extra=self._demo_extra(done),
            )
            self._viewer.tick(60)
            if env.core.phase not in (
                Phase.STACK_SETTLING,
                Phase.BALL_ACTIVE,
                Phase.RESOLVING,
            ):
                break

    def _tick_continuous_demo(self) -> None:
        if not self.continuous_demo or self._demo_env is None or not self._ensure_viewer():
            return
        env = self._demo_env

        if self._demo_pause_frames > 0:
            self._draw_demo_frame(env, done=True)
            self._demo_pause_frames -= 1
            if self._demo_pause_frames == 0:
                self._demo_done = True
            return

        budget = self.demo_steps_per_train_step
        while budget > 0:
            if self._demo_done:
                self._start_demo_episode()
            assert self._demo_obs is not None
            action, _ = self.model.predict(self._demo_obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            self._demo_obs = obs
            self._demo_info = info
            self._demo_total_r += float(reward)
            done = terminated or truncated
            self._demo_done = done
            if not self._draw_demo_frame(env, done):
                return
            self._step_demo_physics(env, done)
            budget -= 1
            if done:
                self._demo_pause_frames = max(20, int(self.result_pause_sec * 40))
                if self.verbose:
                    print(
                        f"[viz] Episode {self._episode_count}: reward={self._demo_total_r:.1f} "
                        f"win={info.get('win')}"
                    )
                break

    def _play_one_episode(self) -> None:
        if self._demo_env is None or not self._ensure_viewer():
            return
        env = self._demo_env
        self._episode_count += 1
        ep = self._episode_count
        obs, _ = env.reset()
        done = False
        info: dict[str, Any] = {}
        steps = 0
        while not done and steps < 2500:
            if self._viewer is None or not self._viewer.handle_events():
                return
            action, _ = self.model.predict(obs, deterministic=False)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            self._draw_demo_frame(env, done)
            self._step_demo_physics(env, done)
        if self._viewer is not None and done:
            pause = max(20, int(self.result_pause_sec * 40))
            for _ in range(pause):
                if not self._viewer.handle_events():
                    break
                self._demo_info = info
                self._draw_demo_frame(env, True)
            if self.verbose:
                print(f"[viz] Demo episode {ep}: win={info.get('win')}")
        self._last_render_step = int(self.num_timesteps)

    def _on_rollout_end(self) -> None:
        self._record_rollout_stats()
        if self.live_demo and not self.continuous_demo:
            if self.num_timesteps - self._last_render_step >= self.render_every:
                self._play_one_episode()

    def _on_step(self) -> bool:
        if self.live_demo and self.continuous_demo:
            self._tick_continuous_demo()
        return True

    def _on_training_end(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._demo_env is not None:
            self._demo_env.close()
        if self._fig is not None and self.live_plot and not self.colab_mode:
            import matplotlib.pyplot as plt

            plt.ioff()
        print(f"Training plot saved to {self.plot_path.resolve()}")


class EvalWinRateCallback(BaseCallback):
    """Periodic headless eval; updates the training graph with win rate."""

    def __init__(
        self,
        viz: TrainingVizCallback,
        level: int = 1,
        eval_every: int = 10_000,
        n_episodes: int = 2,
        fast_mode: bool = True,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.viz = viz
        self.level = level
        self.eval_every = max(1000, eval_every)
        self.n_episodes = n_episodes
        self.fast_mode = fast_mode
        self._last_eval = -self.eval_every
        self._last_throw_rate = 0.0
        self._last_mean_knocked = 0.0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval < self.eval_every:
            return True
        self._last_eval = int(self.num_timesteps)
        env = StackSmashEnv(level=self.level, fast_mode=self.fast_mode)
        wins = 0
        throws = 0
        knocked_list: list[int] = []
        rewards: list[float] = []
        try:
            for _ in range(self.n_episodes):
                obs, _ = env.reset()
                done = False
                total_r = 0.0
                info: dict[str, Any] = {}
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_r += reward
                    done = terminated or truncated
                rewards.append(total_r)
                if info.get("win"):
                    wins += 1
                if env.core.throw_result is not None:
                    throws += 1
                    knocked_list.append(int(info.get("knocked", env.core.knocked_count)))
        finally:
            env.close()

        mean_r = float(np.mean(rewards)) if rewards else 0.0
        win_rate = wins / max(len(rewards), 1)
        throw_rate = throws / max(len(rewards), 1)
        mean_knocked = float(np.mean(knocked_list)) if knocked_list else 0.0
        self._last_throw_rate = throw_rate
        self._last_mean_knocked = mean_knocked
        self.viz.record_eval(mean_r, win_rate)
        if self.verbose:
            print(
                f"[eval] step {self.num_timesteps:,}: "
                f"reward={mean_r:.1f} win={win_rate:.0%} "
                f"throws={throw_rate:.0%} avg_tower_knocked={mean_knocked:.1f}"
            )
        return True