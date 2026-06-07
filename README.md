# Agario RL browser experiment

This repository is a deterministic Agar.io-style reinforcement-learning lab.
Python owns the simulator, reward terms, opponent policies, PPO trainer, and
checkpoint loading. The user-facing game is now a local browser app that reads
live Python frames over WebSockets and renders a smooth canvas arena.

The project is not a wrapper around the public Agar.io website. It trains agents
inside the local `AgarioWorld` simulator so runs are reproducible, inspectable,
and testable.

## Main workflows

### 1. Run the browser game

```powershell
python scripts/run_game.py
```

Useful modes:

```powershell
python scripts/run_game.py --mode showcase
python scripts/run_game.py --mode play
python scripts/run_game.py --mode training-view
```

Open `http://127.0.0.1:5173/` if the browser does not open automatically.

What the modes mean:

- `showcase`: bot-vs-bot full arena with no checkpoint or log writes.
- `play`: human controls `agent_0` with mouse steering and Space split.
- `training-view`: same live arena, with training and human-readiness telemetry
  emphasized in the UI.

### 2. Train agents

Baseline training:

```powershell
python scripts/train.py --updates 20 --device auto
```

Recommended human-ready training path:

```powershell
python scripts/train_human_ready.py --updates 80 --device auto
```

Scenario-curriculum training:

```powershell
python scripts/train.py --updates 20 --scenario-preset agario_curriculum --continuing-respawn --device auto
```

Training writes CSV metrics and checkpoints. The browser game reads those
metrics for visibility but does not write checkpoints unless a training command
is running.

### 3. Evaluate agents

```powershell
python scripts/eval.py --checkpoint checkpoints/latest.pt --episodes 5 --deterministic --device auto
```

Human-readiness evaluation:

```powershell
python scripts/eval_human_readiness.py --checkpoint checkpoints/human_ready_v1/latest.pt --episodes 5
```

## Install

Python:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .[dev]
```

Node.js is required for the Vite frontend. `scripts/run_game.py` runs
`npm install` inside `web/` on first launch unless `--skip-npm-install` is set.

Optional Intel Arc / XPU wheels:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
```

## What is happening on screen

- Colored cells are agents. In `play` mode, `You` is the human-controlled cell.
- Pellets are small food dots. Agents grow by eating pellets and smaller cells.
- Viruses are green spiked hazards. Large cells must route around them.
- Ejected mass appears as small blue pieces when the simulator enables ejects.
- Edge arrows show off-screen threats and targets so users understand danger
  even when the map is bigger than the viewport.
- The minimap shows the full arena, all agents, viruses, and the current camera
  window.
- Training panels show scenario, update count, active checkpoint, reward-derived
  split safety, useful splits, unsafe splits, FPS, and current population.

## Project layout

- `agario_rl/env/`: world rules, collisions, observations, rewards, respawn, and
  the multi-agent env wrapper.
- `agario_rl/rl/`: shared PPO trainer, networks, buffers, and async utilities.
- `agario_rl/opponents.py`: scripted and checkpoint-backed opponent policies
  used for human-ready mixed-opponent training.
- `agario_rl/web/`: FastAPI app, WebSocket session runtime, and JSON frame
  serialization for the browser.
- `web/`: Vite + TypeScript canvas frontend.
- `scripts/run_game.py`: starts the API and browser dev server together.
- `scripts/train.py`: baseline and scenario-curriculum PPO training.
- `scripts/train_human_ready.py`: mixed-opponent training path for agents that
  should behave better around humans.
- `scripts/eval.py` and `scripts/eval_human_readiness.py`: checkpoint
  evaluation.
- `docs/`: quickstart, architecture, controls/tuning, learning notes, and
  experiment reports.
- `docs/project_story_and_readiness.md`: what was finished, why the browser
  reset matters, and how to verify readiness on different devices.

## Human-ready training goal

The reward and UI focus is survival-quality behavior, not just splitting often.
The recommended training path mixes checkpoint opponents with scripted foragers,
evaders, hunters, and objective-driven bots. The reward terms and visible
counters emphasize:

- safe survival near larger threats
- useful split attacks only when the target is catchable
- penalties for unsafe splits and deaths after bad splits
- target pressure without chasing into corners or viruses
- continuing respawn quality so agents recover into playable matches

## Verification

Fast checks:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_browser_runtime.py tests\test_simulator_upgrade.py tests\test_scenario_training_smoke.py
```

Full Python suite:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Frontend build:

```powershell
cd web
npm install
npm run build
```

Browser smoke target:

```powershell
python scripts/run_game.py --mode play
```

Then inspect `http://127.0.0.1:5173/?mode=play`.

For a complete project narrative and readiness checklist, read
`docs/project_story_and_readiness.md`.
