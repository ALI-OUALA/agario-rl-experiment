# Quickstart

This page is optimized for copy-paste use on Windows. It gives you the
shortest path to install the project, run the main workflows, and test the
current checkpoints.

## Copy-paste setup

Run these commands from the project root.

### Create and activate the venv

```powershell
python -m venv .venv
```

```powershell
.\.venv\Scripts\Activate.ps1
```

### Install the project

```powershell
python -m pip install -e .[dev]
```

### Optional: install Intel Arc XPU wheels

Run this only if you want to test Intel Arc acceleration.

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
```

### Optional: verify Intel Arc detection

```powershell
python -c "import torch; print(torch.xpu.is_available())"
```

## Most common commands

If you only want the main commands in one place, use this section.

### Train from scratch

```powershell
python scripts/train.py --updates 10 --device auto
```

### Train with the scenario curriculum

```powershell
python scripts/train.py --updates 10 --scenario-preset agario_curriculum --continuing-respawn --device auto
```

### Resume the main checkpoint to a target update

```powershell
python scripts/train.py --resume --updates 500 --checkpoint checkpoints/latest.pt --device auto
```

### Run a no-save showcase demo

```powershell
python scripts/showcase.py
```

### Play against the current agents

```powershell
python scripts/play.py --checkpoint checkpoints/latest.pt
```

### Evaluate the current checkpoint

```powershell
python scripts/eval.py --checkpoint checkpoints/latest.pt --episodes 5 --deterministic --device auto
```

### Train the mixed-opponent human-ready run

```powershell
python scripts/train_human_ready.py --updates 80 --device auto
```

### Resume the mixed-opponent human-ready run

```powershell
python scripts/train_human_ready.py --resume --updates 160 --checkpoint checkpoints/human_ready_v1/latest.pt --checkpoint-dir checkpoints/human_ready_v1 --metrics-csv logs/human_ready_v1_train_metrics.csv --opponent-checkpoint checkpoints/checkpoint_00500.pt --device cpu --inference-device cpu
```

### Evaluate human-readiness

```powershell
python scripts/eval_human_readiness.py --checkpoint checkpoints/human_ready_v1/latest.pt --episodes 20 --device auto
```

## Workflow guide

Use this section when you want a little context with the commands.

### Train quickly

Run a short headless training pass when you want a new sequence that starts at
update `1`.

```powershell
python scripts/train.py --updates 10 --device auto
```

This command writes:

- `logs/train_metrics.csv`
- `checkpoints/checkpoint_*.pt`
- `checkpoints/latest.pt`

### Train with the large full-arena preset

```powershell
python scripts/train.py --updates 10 --scenario-preset full_arena --device auto
```

The `full_arena` preset uses a larger Agar.io-style map, more agents, more
pellets, viruses, mass decay, ejected mass, continuing respawn, and the richer
observation and reward terms. It is the default presentation target for
showcase and play modes.

### Train with the scenario curriculum

Run the scenario preset when you want the richer Agar.io-style environment.

```powershell
python scripts/train.py --updates 10 --scenario-preset agario_curriculum --continuing-respawn --device auto
```

This path keeps the existing `env.reset()` and `env.step(actions)` contract,
but it activates viruses, ejected mass, mass decay, continuing respawn, richer
observation features, and extra reward terms. It writes the same files as
classic training:

- `logs/train_metrics.csv`
- `checkpoints/checkpoint_*.pt`
- `checkpoints/latest.pt`

Use `--scenario-preset classic` or omit the flag when you want the baseline
environment.

### Resume a saved checkpoint

Run resume mode when you want to continue a checkpoint to a target milestone
without resetting the update counter.

```powershell
python scripts/train.py --resume --updates 500 --checkpoint checkpoints/latest.pt --device auto
```

### Run a no-save showcase

Run this when you want a visitor-friendly example without writing checkpoints
or metric logs.

```powershell
python scripts/showcase.py
```

The showcase defaults to the large full-arena preset and the
`checkpoints/human_ready_v1/latest.pt` checkpoint. If the checkpoint is missing
or incompatible, it still runs with the scripted opponent pool.

For CI or a quick local smoke test:

```powershell
python scripts/showcase.py --headless --seconds 5 --checkpoint checkpoints/missing_showcase.pt
```

### Play against the current agents

Run the dedicated play mode when you want to test the trained bots directly.

```powershell
python scripts/play.py --checkpoint checkpoints/latest.pt
```

Play mode uses these controls:

- move the mouse to steer
- press `Space` to split
- press `Enter` to restart after death or at the end of a round

The player no longer gets a human-only eject action in default play mode. That
older setup was unfair because the bots had never been trained to answer that
extra option.

### Evaluate a checkpoint

Run a short deterministic evaluation against the latest checkpoint.

```powershell
python scripts/eval.py --checkpoint checkpoints/latest.pt --episodes 5 --deterministic --device auto
```

### Train against the mixed opponent pool

Run the human-readiness training loop when you want a fresh learner to face
something stronger than mirror self-play.

```powershell
python scripts/train_human_ready.py --updates 80 --device auto
```

If you want to continue the current `human_ready_v1` checkpoint instead of
starting over, use:

```powershell
python scripts/train_human_ready.py --resume --updates 160 --checkpoint checkpoints/human_ready_v1/latest.pt --checkpoint-dir checkpoints/human_ready_v1 --metrics-csv logs/human_ready_v1_train_metrics.csv --opponent-checkpoint checkpoints/checkpoint_00500.pt --device cpu --inference-device cpu
```

### Evaluate human-readiness proxies

Run the proxy evaluator when you want metrics that reflect the kinds of
mistakes a human punishes, such as corner camping and bad threat response.

```powershell
python scripts/eval_human_readiness.py --checkpoint checkpoints/human_ready_v1/latest.pt --episodes 20 --device auto
```

### Benchmark CPU versus Intel Arc

Run the training benchmark before deciding whether `xpu` helps on your
machine.

```powershell
python scripts/benchmark_perf.py --mode train --updates 2 --device cpu
```

```powershell
python scripts/benchmark_perf.py --mode train --updates 2 --device xpu
```

The benchmark prints rollout time and PPO update time separately. That matters
because this project is still heavily rollout-bound.

### Benchmark step or render cost

Run the benchmark script when you want a quick performance check.

```powershell
python scripts/benchmark_perf.py --mode step --steps 400
```

```powershell
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
