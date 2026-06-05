<p align="center">
  <img src="docs/images/hero-banner.jpg" alt="Stack & Smash" width="100%" />
</p>

<h1 align="center">Stack & Smash</h1>

<p align="center">
  <strong>Stack tins. Build the tower. Throw the ball. Knock them all down.</strong>
</p>

<p align="center">
  <a href="#play">Play</a> ·
  <a href="#ai-agent">AI Agent</a> ·
  <a href="#how-it-works">Rules</a> ·
  <a href="#setup">Setup</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Pygame-2.5+-FF6F00?style=for-the-badge" alt="Pygame" />
  <img src="https://img.shields.io/badge/Pymunk-Physics-4A90D9?style=for-the-badge" alt="Pymunk" />
  <img src="https://img.shields.io/badge/RL-PPO-7B5CFC?style=for-the-badge" alt="PPO" />
</p>

---

## At a glance

| | |
|:---:|:---|
| 🎮 | **Physics stacking game** - Pygame + Pymunk |
| 🤖 | **Local RL agent** - trained with PPO Reinforcement Learning Algorithm (Stable-Baselines3) |
| 🎯 | **Same rules** for human and AI - one shared `game_core.py` |
| 🏆 | **Level 1 max score** = **129** (perfect tower + full knockdown) |

<p align="center">
  <img src="docs/images/Screenshot-human.png" alt="Stack phase and smash phase" width="720" />
</p>

<p align="center"><em>Stack every tin on the first can · Throw · Crush the tower to win</em></p>

---

## How it works

<a id="how-it-works"></a>

```mermaid
flowchart LR
    A[Drop tins] --> B[Settle]
    B --> C{Valid vertical tower?}
    C -->|No| D[Lose]
    C -->|Yes| E[Aim & throw]
    E --> F{All tins down?}
    F -->|Yes| G[Win]
    F -->|No| D
```

**Tower rule** - Only a **vertical column on your first tin** counts. Side placements or cans off the platform do not count.

**Win** - Every level tin on that tower, then **every** one falls after the throw.

---

## Play

<a id="play"></a>

```powershell
pip install -r requirements.txt
python main.py
```

| | Control |
|:---:|:---|
| Move tin | Mouse · Arrow keys · WASD |
| Drop | **Space** · Left click |
| Lock stack | **Enter** (after all tins placed) |
| Aim | **Up / Down** or **W / S** |
| Throw | **Enter** |
| Restart | **R** |

**Scoring** - `Stack points + Throw points` on the UI. Taller tower = more stack pts · Each knocked tin = 10 pts.

---

## AI agent

<a id="ai-agent"></a>

The agent learns with **reinforcement learning** (not defined rules). It practices in a Gymnasium environment, then you watch the saved policy play.

```mermaid
flowchart TB
    subgraph you["Human player"]
        M[main.py]
    end
    subgraph core["Shared brain"]
        G[game_core.py]
    end
    subgraph rl["Machine learning"]
        E[env.py] --> T[train.py]
        T --> Z[(ppo_stack_smash.zip)]
        Z --> W[watch_agent.py]
    end
    M --> G
    E --> G
    W --> G
```

| Command | What it does |
|---------|----------------|
| `python train.py --timesteps 500000 --level 1` | Train agent using level 1 500000 times (~12 min on average) |
| `python watch_agent.py` | **Watch only** - saved model plays the game, no training |
| `python train.py --watch` | Train **and** watch live demos (slower training) |

> **Note:** `watch_agent` prints `reward=~77` - that is the **RL training score**, not the human UI total (**129** max on level 1).

<p align="center">
  <img src="logs/training_curve.png" alt="Training learning curve" width="640" />
</p>

<p align="center"><em>Learning curve saved during training · <code>logs/training_curve.png</code></em></p>

---

## Setup

<a id="setup"></a>

```powershell
git clone <your-repo-url>
cd Game-Playing-Reinforcement-Learning-Agent

pip install -r requirements.txt      # play
pip install -r requirements-rl.txt   # train + watch agent
```

**Requires** `assets/tin_garbage_0.png` … `tin_garbage_16.png` (included in repo).

---

## Project layout

```
tin-tower-game/
├── main.py              # Human game
├── game_core.py         # Rules + physics (shared)
├── env.py               # RL environment
├── train.py             # PPO training
├── watch_agent.py       # Demo trained agent
├── sprites.py           # Loads tin artwork
├── assets/              # Tin PNGs
├── models/              # Trained agent (.zip)
└── logs/                # Training curve
```

---

## Levels

| Level | Tins |
|:-----:|:----:|
| 1 | 6 |
| 2 | 7 |
| … | … |
| 12 | 17 |

Use `--level N` with `train.py` and `watch_agent.py` for a specific level.

---

## Screenshots

<p align="center">
  <img src="docs/images/Screenshot-agent.png" alt="Agent playing game" width="640" />
  <img src="docs/images/Screenshot-human.png" alt="Human playing game" width="640" />
  <img src="docs/images/Screenshot-win.png" alt="Winning the game" width="640" />
</p>

| File | Suggested content |
|------|-------------------|
| `docs/images/screenshot-human.png` | Human playing and valid tower |
| `docs/images/screenshot-agent.png` | `watch_agent.py` mid-throw |
| `docs/images/screenshot-win.png` | Win / score UI |


---

## Deeper docs

| Resource | Description |
|----------|-------------|
| `Stack_Smash_Presentation_Study_Guide.pdf` | Full talk track & Q&A (if present) |
| `game_core.py` | Scoring formulas & tower logic |

---

<p align="center">
  <sub>Built with Pygame · Pymunk · Gymnasium · Stable-Baselines3</sub>
