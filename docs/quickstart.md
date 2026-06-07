# Quickstart

This page shows the three normal workflows: run the game, train agents, and
evaluate agents.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .[dev]
```

Install Node.js if `npm` is not available. The first `scripts/run_game.py` run
installs frontend packages inside `web/` automatically.

## Run the browser game

```powershell
python scripts/run_game.py
```

Open `http://127.0.0.1:5173/`.

Modes:

```powershell
python scripts/run_game.py --mode showcase
python scripts/run_game.py --mode play
python scripts/run_game.py --mode training-view
```

Useful flags:

- `--api-port 8765`: FastAPI and WebSocket port.
- `--web-port 5173`: Vite frontend port.
- `--checkpoint checkpoints/human_ready_v1/latest.pt`: checkpoint loaded for
  checkpoint-backed opponents.
- `--no-open`: start servers without opening a browser.
- `--skip-npm-install`: do not run `npm install` automatically.

Controls in `play` mode:

- Move mouse to steer.
- Press Space to split.
- Press R or the Reset button to reset the arena.

## Train agents

Baseline:

```powershell
python scripts/train.py --updates 20 --device auto
```

Human-ready mixed-opponent path:

```powershell
python scripts/train_human_ready.py --updates 80 --device auto
```

Scenario curriculum:

```powershell
python scripts/train.py --updates 20 --scenario-preset agario_curriculum --continuing-respawn --device auto
```

Resume:

```powershell
python scripts/train.py --resume --updates 500 --checkpoint checkpoints/latest.pt --device auto
```

Training writes checkpoints and `logs/train_metrics.csv`. The browser game can
display that latest metrics row but does not save new checkpoints.

Low-resource smoke:

```powershell
$smoke = Join-Path $env:TEMP "agario_rl_smoke"
python scripts/train.py --updates 1 --device cpu --checkpoint "$smoke/latest.pt" --checkpoint-dir "$smoke" --metrics-csv "$smoke/train_metrics.csv"
```

This proves the live training loop works without touching protected project
checkpoints.

## Evaluate agents

```powershell
python scripts/eval.py --checkpoint checkpoints/latest.pt --episodes 5 --deterministic --device auto
```

Human-readiness evaluation:

```powershell
python scripts/eval_human_readiness.py --checkpoint checkpoints/human_ready_v1/latest.pt --episodes 5
```

## Verify changes

Python:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_browser_runtime.py tests\test_simulator_upgrade.py tests\test_scenario_training_smoke.py
```

Frontend:

```powershell
cd web
npm install
npm run build
```

Browser smoke:

```powershell
python scripts/run_game.py --mode play
```

Then open `http://127.0.0.1:5173/?mode=play` and confirm that the arena,
pellets, agents, minimap, leaderboard, training state, and keyboard/mouse input
are visible.

For the full story and readiness checklist, read
`docs/project_story_and_readiness.md`.
