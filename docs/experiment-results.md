# Experiment results

This repo is currently an interactive local experiment, not a finalized
benchmark paper. The important result of the current reset is architectural:
the old native UI was removed and replaced by a browser game that can be tested
with normal browser automation while keeping the Python simulator authoritative.

## Current state

- Browser game runtime: `python scripts/run_game.py`
- Modes: `showcase`, `play`, `training-view`
- Python simulation: `AgarioWorld` through `AgarioMultiAgentEnv`
- Frontend: Vite + TypeScript canvas app under `web/`
- Training: PPO scripts unchanged in principle, with human-ready mixed-opponent
  training still available
- Artifacts: existing checkpoints, logs, and docs images are preserved unless a
  fresh experiment run is requested

## Visible human-readiness metrics

The browser frame includes:

- split attempts
- useful splits
- unsafe splits
- split safety
- leader final mass
- active policy source
- current checkpoint
- latest update count from the training CSV

These are not a substitute for evaluation, but they make training behavior
visible enough for a normal user to understand what the agents are doing.

## Recommended evaluation

Use deterministic evaluation for basic checkpoint quality:

```powershell
python scripts/eval.py --checkpoint checkpoints/latest.pt --episodes 5 --deterministic --device auto
```

Use human-readiness evaluation for mixed-opponent behavior:

```powershell
python scripts/eval_human_readiness.py --checkpoint checkpoints/human_ready_v1/latest.pt --episodes 5
```

When publishing a result, report the exact checkpoint path, update count,
scenario preset, device, and command output.
