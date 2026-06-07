# Project story and readiness

This project started as a small Agar.io-style reinforcement-learning simulator.
The important reset is that it is now a complete local browser experiment:
Python runs the world and learning code, while the browser renders a live,
playable arena from WebSocket frames.

## What was finished

- The old native renderer and supervisor UI path were removed from the public
  workflow.
- `scripts/run_game.py` now starts both the FastAPI simulation server and Vite
  browser frontend.
- `agario_rl/web/` provides the runtime, WebSocket API, and browser frame
  schema.
- `web/` provides a TypeScript canvas renderer with camera smoothing, minimap,
  leaderboard, training telemetry, split-safety counters, and human controls.
- PPO training and evaluation still run through the Python simulator, not the
  browser.
- Documentation now explains the user commands, architecture, controls,
  reward story, experiment state, and agent handoff rules.

## Current public experience

Run:

```powershell
python scripts/run_game.py
```

The browser opens a live arena. `showcase` runs bot-vs-bot, `play` gives the
human control of `agent_0`, and `training-view` emphasizes learning telemetry.
All three modes use the same simulator rules.

The visible story is:

1. Agents eat pellets to build mass.
2. Larger agents pressure smaller agents.
3. Threat arrows and target arrows explain off-screen danger.
4. Viruses and split counters show why reckless growth can fail.
5. Training panels connect what the user sees to PPO metrics and checkpoint
   state.

## Training story

Baseline training remains:

```powershell
python scripts/train.py --updates 20 --device auto
```

The richer human-facing path is:

```powershell
python scripts/train_human_ready.py --updates 80 --device auto
```

That path trains one learner against a mixed opponent pool so behavior is less
narrow than pure self-play. The goal is not only high mass; it is survivable
movement, useful splits, lower unsafe splits, and readable behavior around a
human player.

## Any-device readiness

The browser runtime is intentionally modest:

- the frontend is a small Vite/TypeScript canvas app
- physics is stepped by Python at a capped local frame rate
- browser mode uses a reduced pellet density compared with heavy training
- checkpoint policies are optional; scripted opponents run when checkpoint
  loading is unavailable
- `--device auto` lets training choose CPU, CUDA, or XPU when installed

For weaker laptops, use:

```powershell
python scripts/run_game.py --mode showcase --skip-npm-install
python scripts/train.py --updates 5 --device cpu
```

For stronger machines, use scenario curriculum:

```powershell
python scripts/train.py --updates 20 --scenario-preset agario_curriculum --continuing-respawn --device auto
```

## Protected artifacts

Routine code and docs work should not regenerate or delete:

- `checkpoints/*.pt`
- `checkpoints/human_ready_v1/*.pt`
- `logs/*.csv`
- `docs/assets/*.png`

Fresh training or report generation may update them only when requested.

## Verification checklist

Before treating the project as ready:

```powershell
.\.venv\Scripts\python.exe -m pytest
cd web
npm install
npm run build
```

Live training smoke without touching protected logs or checkpoints:

```powershell
$smoke = Join-Path $env:TEMP "agario_rl_smoke"
python scripts/train.py --updates 1 --device cpu --checkpoint "$smoke/latest.pt" --checkpoint-dir "$smoke" --metrics-csv "$smoke/train_metrics.csv"
```

Optional live smoke:

```powershell
python scripts/run_game.py --mode play
```

Then open `http://127.0.0.1:5173/?mode=play` and confirm the arena, minimap,
leaderboard, HUD, WebSocket status, mouse steering, keyboard steering, split,
and reset work.
