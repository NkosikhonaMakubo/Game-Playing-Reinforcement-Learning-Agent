"""
Watch a trained agent play Stack & Smash (pygame window only).

Usage:
  python watch_agent.py
  python watch_agent.py --model models/ppo_stack_smash.zip --level 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from env import StackSmashEnv
from viewer import StackSmashViewer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/ppo_stack_smash.zip"))
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=0, help="0 = loop forever")
    args = parser.parse_args()

    model = PPO.load(str(args.model))
    env = StackSmashEnv(level=args.level)
    viewer = StackSmashViewer(title="Stack & Smash — Watch Agent")

    print("Close the pygame window to quit. Episodes loop until you close it.")
    try:
        ep = 0
        while True:
            ep += 1
            if args.episodes > 0 and ep > args.episodes:
                break
            obs, _ = env.reset()
            done = False
            total_r = 0.0
            while not done:
                if not viewer.handle_events():
                    return
                action, _ = model.predict(obs, deterministic=False)
                obs, reward, terminated, truncated, info = env.step(action)
                total_r += reward
                done = terminated or truncated
                extra = ""
                if done:
                    tt = info.get("tower_total", "?")
                    extra = (
                        "WIN!"
                        if info.get("win")
                        else f"LOSE ({info.get('still_up', '?')} of {tt} tower tins up)"
                    )
                viewer.draw(
                    env.core,
                    subtitle=f"Episode {ep}" + (f"/{args.episodes}" if args.episodes > 0 else ""),
                    extra=extra,
                )
                viewer.tick(60)
            print(f"Episode {ep}: reward={total_r:.1f} win={info.get('win')}")
            if not info.get("win"):
                for _ in range(120):
                    if not viewer.handle_events():
                        return
                    viewer.draw(
                        env.core,
                        subtitle=f"Episode {ep} — LOSE",
                        extra=f"Retrying… {info.get('still_up', '?')} tower tins did not fall",
                    )
                    viewer.tick(60)
    finally:
        env.close()
        viewer.close()


if __name__ == "__main__":
    main()