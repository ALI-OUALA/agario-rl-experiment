# Agent handoff guide

This file is the first stop for coding agents working on the Agario RL browser
experiment. The current project uses a Python simulator plus a browser canvas
frontend. There is no public native renderer, native play path, or old
reference UI path.

## Start here

Read these files first:

1. `config/default.yaml`
2. `agario_rl/env/world.py`
3. `agario_rl/web/runtime.py`
4. `agario_rl/web/frames.py`
5. `web/src/main.ts`
6. `scripts/run_game.py`
7. `scripts/train.py`

Then read the docs page that matches the task:

- `docs/quickstart.md` for user commands
- `docs/runtime_architecture.md` for Python server, WebSocket frames, and
  browser rendering
- `docs/controls_and_tuning.md` for browser controls and config tuning
- `docs/how_it_learns.md` for PPO, rewards, opponents, and human readiness
- `README.md` for the project narrative

## Project shape

Training calls `AgarioMultiAgentEnv.reset()` and
`AgarioMultiAgentEnv.step(actions)`, which wrap `AgarioWorld`. The browser game
uses `BrowserGameSession`, which advances the same env and serializes each frame
for the TypeScript canvas renderer.

Main directories:

- `agario_rl/env/`: simulator rules, observations, rewards, and env wrapper
- `agario_rl/rl/`: PPO trainer, networks, buffers, and async utilities
- `agario_rl/web/`: FastAPI app, WebSocket session runtime, browser frame schema
- `web/`: Vite + TypeScript canvas frontend
- `scripts/`: run game, train, evaluate, benchmark, and report helpers
- `tests/`: simulator, training, browser runtime, CLI, and logging tests
- `docs/`: user docs and experiment notes

## Stable contracts

Preserve these unless the user explicitly asks for a breaking change:

- `AgarioMultiAgentEnv.reset()` returns per-agent observations.
- `AgarioMultiAgentEnv.step(actions)` accepts the existing action contract.
- Continuous actions are `[steer_x, steer_y, split]`.
- Classic mode remains the default for `scripts/train.py` unless scenario flags
  are passed.
- Browser frames are JSON payloads built in `agario_rl/web/frames.py`.
- The browser renderer consumes frames and sends input; it does not own game
  rules.
- Checkpoints and CSV logs are written by training scripts, not by
  `scripts/run_game.py`.

## Runtime map

Public game command:

```powershell
python scripts/run_game.py
```

Modes:

```powershell
python scripts/run_game.py --mode showcase
python scripts/run_game.py --mode play
python scripts/run_game.py --mode training-view
```

Training:

```powershell
python scripts/train.py --updates 20 --device auto
python scripts/train.py --updates 20 --scenario-preset agario_curriculum --continuing-respawn --device auto
python scripts/train_human_ready.py --updates 80 --device auto
```

Evaluation:

```powershell
python scripts/eval.py --checkpoint checkpoints/latest.pt --episodes 5 --deterministic --device auto
python scripts/eval_human_readiness.py --checkpoint checkpoints/human_ready_v1/latest.pt --episodes 5
```

## Simulator upgrade map

The richer simulator is controlled through these config groups:

- `viruses`: virus count, mass, feeding, and split behavior
- `mass_decay`: passive mass loss and minimum mass floor
- `observation_features`: additive threat, target, virus, and eject features
- `reward_terms`: threat escape, target pressure, corner penalty, survival
  quality, split safety, virus split, and respawn components
- `scenario_curriculum`: stage labels and scenario preset behavior
- `simulation.continuing_respawn`: respawn eliminated agents instead of ending
  each round
- `simulation.respawn_mass`: starting mass for continuing-mode respawns
- `checkpoint.latest_path`: default checkpoint path for train/eval

`apply_scenario_preset(config, "agario_curriculum")` enables richer training.
`apply_scenario_preset(config, "full_arena")` configures the browser-scale arena.

## Protected artifacts

Do not modify these during normal code or docs tasks:

- `checkpoints/*.pt`
- `checkpoints/human_ready_v1/*.pt`
- `logs/*.csv`
- `docs/assets/*.png`

Only update those files when the user explicitly asks for fresh training,
evaluation, charts, or report generation.

## Safe checks

Use the project venv when it exists:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Targeted browser/runtime checks:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_browser_runtime.py tests\test_simulator_upgrade.py tests\test_scenario_training_smoke.py
```

Frontend:

```powershell
cd web
npm install
npm run build
```

Docs whitespace:

```powershell
git diff --check -- README.md CONTRIBUTING.md AGENTS.md docs
```

## Change rules

- Add or update tests for every world-rule, observation, reward, CLI, WebSocket
  frame, or browser input behavior change.
- Keep PPO code out of the browser and browser drawing out of PPO.
- Add new browser-visible state in `agario_rl/web/frames.py` and draw it in
  `web/src/main.ts`.
- Keep public commands documented in both `README.md` and `docs/quickstart.md`.
- Keep config fields documented in `docs/controls_and_tuning.md`.
- Preserve checkpoints, logs, and docs assets unless the task explicitly asks
  for a new experiment run.

## Known gotchas

- The project path contains spaces. Quote paths in shell commands.
- On this Windows setup, `rg` can fail with access denied. Switch to
  PowerShell-native `Get-ChildItem` and `Select-String`.
- Full training writes checkpoints and CSV metrics. Use short smoke tests or
  temporary paths for verification.
- The workload is often rollout-bound, so `--device xpu` is not automatically
  faster than CPU.
- Public Agar.io website training is a separate browser-automation and
  computer-vision project. This repo trains on the internal simulator.
