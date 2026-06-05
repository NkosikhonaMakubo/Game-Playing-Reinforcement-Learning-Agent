"""
Train a PPO agent on Stack & Smash.

Train the agent (game rules: vertical tower on 1st tin, knock all down to win):

  pip install -r requirements-rl.txt
  del models\\ppo_stack_smash.zip   # if retraining from scratch
  python train.py --timesteps 500000 --level 1

Optional:
  python train.py --timesteps 500000 --watch    # demo every 8192 steps
  python watch_agent.py --model models\\ppo_stack_smash.zip
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from callbacks import EvalWinRateCallback, TrainingVizCallback
from env import StackSmashEnv


def progress_bar_enabled() -> bool:
    try:
        import rich  # noqa: F401
        import tqdm  # noqa: F401

        return True
    except ImportError:
        return False


def tensorboard_log_dir(log_dir: Path) -> str | None:
    try:
        import tensorboard  # noqa: F401

        return str(log_dir / "tensorboard")
    except ImportError:
        return None


def make_single_env(level: int, log_dir: Path, rank: int, fast_mode: bool):
    def _init():
        env = StackSmashEnv(level=level, fast_mode=fast_mode)
        return Monitor(env, filename=str(log_dir / f"monitor_{rank}.csv"))

    return _init


def make_vec_envs(level: int, log_dir: Path, n_envs: int, fast_mode: bool, parallel: bool):
    fns = [make_single_env(level, log_dir, i, fast_mode) for i in range(n_envs)]
    if n_envs == 1:
        return DummyVecEnv(fns)
    if parallel:
        return SubprocVecEnv(fns, start_method="spawn")
    return DummyVecEnv(fns)


def train(
    timesteps: int,
    level: int,
    n_envs: int,
    save_path: Path,
    log_dir: Path,
    render_every: int,
    eval_every: int,
    n_steps: int,
    colab: bool = False,
    live_demo: bool = False,
    live_plot: bool = False,
    continuous_demo: bool = False,
    parallel_envs: bool = True,
    fast_mode: bool = True,
    eval_episodes: int = 2,
) -> None:
    save_path.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    tb_log = tensorboard_log_dir(log_dir)

    vec = make_vec_envs(level, log_dir, n_envs, fast_mode, parallel_envs)

    viz_cb = TrainingVizCallback(
        level=level,
        render_every=render_every,
        plot_path=log_dir / "training_curve.png",
        verbose=1,
        colab_mode=colab,
        live_demo=live_demo and not colab,
        live_plot=live_plot and not colab,
        continuous_demo=continuous_demo and live_demo and not colab,
    )
    eval_cb = EvalWinRateCallback(
        viz=viz_cb,
        level=level,
        eval_every=eval_every,
        n_episodes=eval_episodes,
        fast_mode=fast_mode,
        verbose=1,
    )
    callbacks = CallbackList([viz_cb, eval_cb])

    model = PPO(
        "MlpPolicy",
        vec,
        verbose=1,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=64,
        gamma=0.99,
        ent_coef=0.08,
        tensorboard_log=tb_log,
    )

    mode = "fast (headless)" if not live_demo and not live_plot else "with live view"
    print(f"Training — {mode}")
    print(f"  Parallel envs : {n_envs} ({'subprocess' if parallel_envs and n_envs > 1 else 'in-process'})")
    print(f"  Fast physics  : {'on' if fast_mode else 'off'}")
    print(f"  PPO n_steps   : {n_steps}")
    if live_demo:
        if continuous_demo:
            print("  Game window   : continuous episodes while training")
        else:
            print(f"  Game window   : demo episode every {render_every:,} training steps")
    else:
        print("  Game window   : off (add --watch to see the agent play)")
    if live_plot:
        print("  Live graph    : on")
    else:
        print(f"  Learning curve: saved to {(log_dir / 'training_curve.png').resolve()} (every {eval_every:,} steps)")
    if tb_log:
        print(f"  TensorBoard   : {Path(tb_log).resolve()}")
    use_progress = progress_bar_enabled()
    if not use_progress:
        print("  Progress bar  : off (pip install tqdm rich)")
    print()

    out = save_path / "ppo_stack_smash"
    try:
        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            progress_bar=use_progress,
            tb_log_name="PPO",
        )
        model.save(str(out))
        print(f"Saved model to {out}.zip")
        if not live_demo:
            print(f"Watch it play: python watch_agent.py --model {out}.zip --level {level}")
    except KeyboardInterrupt:
        print("\nTraining interrupted — saving checkpoint...")
        model.save(str(out))
        print(f"Saved model to {out}.zip")
    finally:
        vec.close()


def evaluate(model_path: Path, level: int, episodes: int, render: bool) -> None:
    from stable_baselines3 import PPO

    from viewer import StackSmashViewer

    model = PPO.load(str(model_path))
    env = StackSmashEnv(level=level)
    viewer = StackSmashViewer(title="Stack & Smash — Evaluation") if render else None
    wins = 0
    try:
        for ep in range(episodes):
            obs, _ = env.reset()
            done = False
            total_r = 0.0
            while not done:
                if viewer and not viewer.handle_events():
                    break
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_r += reward
                done = terminated or truncated
                if viewer:
                    extra = ""
                    if done and info.get("win"):
                        extra = "WIN"
                    elif done:
                        extra = (
                            f"LOSE — {info.get('still_up', '?')} of "
                            f"{info.get('tower_total', '?')} tower tins up"
                        )
                    viewer.draw(
                        env.core,
                        subtitle=f"Episode {ep + 1}/{episodes}",
                        extra=extra,
                    )
                    viewer.tick(60)
            if info.get("win"):
                wins += 1
            print(
                f"Episode {ep + 1}: reward={total_r:.1f} win={info.get('win')} "
                f"still_up={info.get('still_up', '-')}"
            )
    finally:
        env.close()
        if viewer:
            viewer.close()
    print(f"Win rate: {wins}/{episodes} ({100 * wins / max(episodes, 1):.0f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train/eval Stack & Smash RL agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python train.py --timesteps 200000\n"
            "  python train.py --timesteps 200000 --watch\n"
            "  python train.py --watch --plot --watch-continuous\n"
        ),
    )
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--n-envs", type=int, default=8, help="Parallel training envs (default 8)")
    parser.add_argument("--n-steps", type=int, default=1024, help="PPO rollout length per env")
    parser.add_argument("--save-dir", type=Path, default=Path("models"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument(
        "--render-every",
        type=int,
        default=8192,
        help="With --watch: show a demo episode every N training steps",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=10_000,
        help="Update saved learning-curve PNG every N steps",
    )
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--model", type=Path, default=Path("models/ppo_stack_smash.zip"))
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--no-render", action="store_true", help="Headless evaluation")
    parser.add_argument("--colab", action="store_true", help="Colab: inline graph only")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Show pygame window during training (periodic demo episodes)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Live matplotlib graph during training (slower)",
    )
    parser.add_argument(
        "--watch-continuous",
        action="store_true",
        help="With --watch: keep playing episodes non-stop (much slower)",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Use in-process vec env instead of subprocess (slower, easier to debug)",
    )
    parser.add_argument(
        "--no-fast",
        action="store_true",
        help="Disable fast physics / shorter episode limits",
    )
    args = parser.parse_args()

    if args.eval:
        evaluate(args.model, args.level, args.episodes, render=not args.no_render)
    else:
        train(
            args.timesteps,
            args.level,
            args.n_envs,
            args.save_dir,
            args.log_dir,
            args.render_every,
            args.eval_every,
            args.n_steps,
            colab=args.colab,
            live_demo=args.watch or args.colab,
            live_plot=args.plot or args.colab,
            continuous_demo=args.watch_continuous,
            parallel_envs=not args.no_parallel,
            fast_mode=not args.no_fast,
            eval_episodes=5,
        )


if __name__ == "__main__":
    main()