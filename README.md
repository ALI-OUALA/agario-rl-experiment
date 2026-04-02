# Agario RL experiment

This repository is a reinforcement learning lab built around a deterministic
Agar.io-style world. You can train a shared PPO policy, observe it in the
Raylib cockpit, evaluate checkpoints, and now play directly against the
trained agents in a dedicated human mode.

As of April 2, 2026, the main checkpoint has been continued from update `366`
to update `500`. The repository also includes a write-up of the experiment,
local chart assets for publication, a blog draft, and a LinkedIn draft.

## Project overview

The project keeps the experiment contract intentionally stable. The trained
agents still use the same observation space, action space, reward shaping, and
shared-policy PPO setup that the earlier milestones used.

- Deterministic 2D world simulation with pellets, split, merge, and cell
  eating.
- Shared-parameter PPO with GAE, entropy regularization, and peer imitation.
- Async training support for the observer cockpit.
- Raylib observer cockpit for live control, telemetry, and checkpointing.
- Dedicated human-play mode with `1` player and `2` trained agents.
- Publication-ready Markdown files and local chart assets under `docs/`.

## Repository layout

The codebase is small enough to inspect end to end. These folders and scripts
matter most when you run or extend the experiment.

- `agario_rl/env/`: deterministic world simulation and environment wrapper.
- `agario_rl/rl/`: policy network, PPO trainer, imitation buffer, and async
  worker.
- `agario_rl/rendering/`: Raylib observer cockpit renderer and immutable frame
  models.
- `agario_rl/supervisor/`: cockpit controller, semantic commands, and runtime
  stats.
- `agario_rl/play/`: human input adapter and headless-ready human-vs-bots
  session wrapper.
- `scripts/train.py`: headless training entrypoint with resume-to-target
  milestone support.
- `scripts/supervise.py`: interactive observer cockpit entrypoint.
- `scripts/play.py`: human-playable mode against the trained agents.
- `scripts/eval.py`: deterministic checkpoint evaluation.
- `scripts/generate_report_assets.py`: chart generation for documentation.
- `docs/`: experiment report, blog draft, LinkedIn draft, and reference docs.

## Install

Start in the project root, create a virtual environment, and install the
project in editable mode.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

The interactive runtime and play mode depend on the `raylib` Python package,
which is imported through `pyray`.

## Run the main workflows

The project now supports five common flows: fresh training, resumed training,
supervision, play mode, and evaluation.

### Train from scratch

Use a fresh run when you want a new training series that starts at update `1`.

```bash
python scripts/train.py --updates 20
```

This command writes:

- canonical metrics to `logs/train_metrics.csv`
- numbered checkpoints to `checkpoints/checkpoint_*.pt`
- the latest training state to `checkpoints/latest.pt`

### Resume to a target milestone

Use resume mode when you want to continue a saved checkpoint without resetting
the update counter.

```bash
python scripts/train.py --resume --updates 500 --checkpoint checkpoints/latest.pt
```

If the checkpoint already stores `update_count=366`, this command runs updates
`367` through `500`, updates `checkpoints/latest.pt`, and writes
`checkpoints/checkpoint_00500.pt`.

### Open the observer cockpit

Use the observer cockpit when you want to inspect behavior, step the
simulation, or continue training interactively.

```bash
python scripts/supervise.py --load-checkpoint
```

The cockpit uses the same canonical metrics file as headless training, so new
interactive updates append cleanly instead of starting a second numbering
series.

### Play against the trained agents

Use play mode when you want a human-readable feel check without changing the
trained-agent interface.

```bash
python scripts/play.py --checkpoint checkpoints/latest.pt
```

The current play controls are:

- move the mouse to steer
- press `Space` to split
- press `E` to eject mass
- press `Enter` to restart after death or at the end of a round

### Evaluate the current checkpoint

Use evaluation when you want a deterministic score pass against the saved
policy.

```bash
python scripts/eval.py --episodes 5 --deterministic
```

### Regenerate publication assets

Use the asset generator when you want to rebuild the local charts used by the
experiment report and the blog draft.

```bash
python scripts/generate_report_assets.py --baseline-eval -2.211 --final-eval -6.284
```

## Latest experiment snapshot

The current repository includes two reportable milestones from the same
checkpoint lineage.

| Milestone | Source | Quick read |
| --- | --- | --- |
| Update `366` | `checkpoints/latest.pt` before continuation | High value loss, publishable baseline snapshot, deterministic 5-episode average return `-2.211` |
| Update `500` | continued run on April 2, 2026 | Lower final losses, new numbered checkpoint, deterministic 5-episode average return `-6.284` |

Read the full interpretation in [docs/experiment-results.md](docs/experiment-results.md).

## Documentation map

The documentation is now split by job rather than by audience guesswork.

- [docs/quickstart.md](docs/quickstart.md): shortest path to install, train,
  supervise, play, and evaluate.
- [docs/experiment-results.md](docs/experiment-results.md): milestone-based
  experiment report with the `100`, `300`, `366`, and `500` observations.
- [docs/blog-case-study.md](docs/blog-case-study.md): hybrid technical and
  build-story blog draft with local chart assets.
- [docs/linkedin-post.md](docs/linkedin-post.md): short LinkedIn draft based on
  the same milestones.
- [docs/controls_and_tuning.md](docs/controls_and_tuning.md): observer cockpit
  controls and config tuning.
- [docs/runtime_architecture.md](docs/runtime_architecture.md): snapshot-driven
  runtime design.
- [docs/how_it_learns.md](docs/how_it_learns.md): concise PPO and imitation
  overview.

## Notes about the current metrics history

The canonical CSV is intentionally honest about what exists and what does not.
The repository has logged updates `1..200` and `367..500`. Updates `201..366`
were produced during an older interactive phase that advanced the checkpoint
without writing every update to the CSV.

The repo does not fabricate that missing segment. Instead, the experiment
report preserves the update `366` checkpoint metrics and evaluation summary as
their own milestone.

## Next steps

If you want to keep pushing the experiment, these are the cleanest next moves.

1. Spend time in `scripts/play.py` and `scripts/supervise.py` to verify whether
   update `500` feels better than the quick deterministic evaluation suggests.
2. Add richer evaluation metrics, such as per-agent win rate, survival time,
   and average final mass, before making stronger claims publicly.
3. If you tune the reward or curriculum again, preserve the current
   observation/action contract so future blog updates stay comparable.
