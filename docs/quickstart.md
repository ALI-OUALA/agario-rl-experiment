# Quickstart

This page gets you from a fresh checkout to a running experiment as quickly as
possible. It covers install, fresh training, resumed training, supervision,
play mode, and evaluation, then points you to the deeper docs.

## Install the project

Start in the project root and create a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

The interactive runtime and play mode depend on the `raylib` Python package.
The project imports it through `pyray`.

## Run the main workflows

Use these commands to cover the most common experiment flows.

### Train quickly

Run a short headless training pass when you want a new sequence that starts at
update `1`.

```bash
python scripts/train.py --updates 10
```

This command writes:

- `logs/train_metrics.csv`
- `checkpoints/checkpoint_*.pt`
- `checkpoints/latest.pt`

### Resume a saved checkpoint

Run resume mode when you want to continue a checkpoint to a target milestone
without resetting the update counter.

```bash
python scripts/train.py --resume --updates 500 --checkpoint checkpoints/latest.pt
```

### Open the observer cockpit

Run the public interactive runtime when you want live controls and charts.

```bash
python scripts/supervise.py --load-checkpoint
```

The side panel shows session cards, training cards, agent cards, and live
charts. The control surface lets you pause, step, change speed, toggle
training, save or load checkpoints, and change camera focus without leaving
the main window.

### Play against the current agents

Run the dedicated play mode when you want to test the trained bots directly.

```bash
python scripts/play.py --checkpoint checkpoints/latest.pt
```

Play mode uses these controls:

- move the mouse to steer
- press `Space` to split
- press `E` to eject mass
- press `Enter` to restart after death or at the end of a round

### Evaluate a checkpoint

Run a short deterministic evaluation against the latest checkpoint.

```bash
python scripts/eval.py --episodes 5 --deterministic
```

### Benchmark step or render cost

Run the benchmark script when you want a quick performance check.

```bash
python scripts/benchmark_perf.py --mode step --steps 400
python scripts/benchmark_perf.py --mode render --frames 180 --overlay full
```

## Learn the basic cockpit controls

The observer cockpit exposes the same actions through buttons and keyboard
shortcuts. These are the most important shortcuts to learn first:

- `Space`: pause or resume
- `N`: step one physics tick
- `Shift+N`: step one decision tick
- `-` and `+`: slow down or speed up
- `T`: toggle Train More
- `R`: reset the episode and session wins
- `1`, `2`, `3`: focus the camera on an agent
- `W`, `A`, `S`, `D` or arrow keys: pan the camera manually
- middle mouse drag: pan the viewport
- `0`: return to follow mode
- `Tab`: switch compact and full cockpit layouts
- `F11`: toggle fullscreen
- `F1`: toggle the built-in help overlay

## Next steps

Use these pages after your first run:

1. Read [README.md](../README.md) for the full project overview.
2. Read [experiment-results.md](./experiment-results.md) for the current
   milestone report.
3. Read [controls_and_tuning.md](./controls_and_tuning.md) for the full
   cockpit reference.
4. Read [runtime_architecture.md](./runtime_architecture.md) to understand the
   snapshot-driven runtime design.
