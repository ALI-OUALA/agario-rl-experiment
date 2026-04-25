# Agent handoff guide

This file is the first stop for coding agents working on the Agario RL
experiment. It explains how to inspect the project, which contracts are stable,
which files are generated artifacts, and which commands prove a change.

## Start here

Read these files first when you work on the simulator or cockpit upgrade:

1. `config/default.yaml`
2. `agario_rl/env/world.py`
3. `agario_rl/rendering/models.py`
4. `scripts/train.py`
5. `scripts/showcase.py`

Then read the docs page that matches the task:

- `docs/quickstart.md` for user commands
- `docs/controls_and_tuning.md` for cockpit controls and config tuning
- `docs/runtime_architecture.md` for the fixed-step runtime and render
  snapshot model
- `docs/how_it_learns.md` for the PPO, imitation, scenario, observation, and
  reward overview
- `README.md` for the complete project narrative and current experiment state

## Project shape

This repo is a deterministic Python RL lab, not a browser wrapper around the
public Agar.io site. Training calls `AgarioMultiAgentEnv.reset()` and
`AgarioMultiAgentEnv.step(actions)`, which wrap `AgarioWorld`.

The main directories are:

- `agario_rl/env/`: world rules, entities, observations, rewards, and the env
  wrapper
- `agario_rl/rl/`: shared PPO trainer, async worker, replay buffers, and
  networks
- `agario_rl/rendering/`: immutable frame models and the Raylib cockpit
- `agario_rl/supervisor/`: shared render controller state and runtime stats
- `agario_rl/play/`: human input and play-session code
- `scripts/`: training, showcase, play, evaluation, benchmarks, and report
  helpers
- `tests/`: pytest coverage for simulator, training, rendering, CLI, and logs
- `docs/`: user docs, architecture notes, reports, and article drafts

## Stable contracts

Preserve these contracts unless the user explicitly asks for a breaking
change:

- `AgarioMultiAgentEnv.reset()` returns per-agent observations.
- `AgarioMultiAgentEnv.step(actions)` accepts the existing action contract.
- Continuous actions are `[steer_x, steer_y, split]`.
- Classic mode remains the default when no scenario flags are passed.
- New simulator behavior is additive through config or
  `--scenario-preset agario_curriculum`.
- Rendering consumes immutable `RenderFrame` payloads, not trainer internals.
- Raylib emits semantic `SupervisorCommand` values, not business logic.

## Simulator upgrade map

The upgraded simulator is controlled through these config groups:

- `viruses`: virus count, mass, feeding, and split behavior
- `mass_decay`: passive cell mass loss and minimum mass floor
- `observation_features`: additive threat, target, virus, split-ready, and
  eject-ready observation features
- `reward_terms`: threat escape, target pressure, corner penalty, survival
  quality, split safety, virus split, and respawn components
- `scenario_curriculum`: stage labels and scenario preset behavior
- `simulation.continuing_respawn`: respawn eliminated agents instead of ending
  a round on each elimination
- `simulation.respawn_mass`: starting mass for continuing-mode respawns

`apply_scenario_preset(config, "agario_curriculum")` enables the richer
training path. It turns on viruses, mass decay, observation extensions, reward
terms, and staged scenario labels. `classic` keeps the baseline behavior.

## Cockpit and CLI map

Use these commands for the main workflows:

```powershell
python scripts/train.py --updates 20 --device auto
```

```powershell
python scripts/train.py --updates 20 --scenario-preset agario_curriculum --continuing-respawn --device auto
```

```powershell
python scripts/showcase.py
```

Public CLI flags added by the simulator/cockpit upgrade are:

- `--scenario-preset`: `classic`, `agario_curriculum`, or `full_arena`
- `--continuing-respawn`: keep eliminated agents in the arena

The public `scripts/supervise.py` entrypoint was removed. Use
`scripts/showcase.py` for the visible no-save arena and `scripts/train.py` or
`scripts/train_human_ready.py` for training.

## Protected artifacts

Do not modify these during normal code or docs tasks:

- `checkpoints/*.pt`
- `checkpoints/human_ready_v1/*.pt`
- `logs/*.csv`
- `docs/assets/*.png`

Only update those files when the user explicitly asks for a fresh training,
evaluation, chart, or report-generation pass. If they are already dirty, treat
them as user or experiment output and leave them alone.

`README.md`, `CONTRIBUTING.md`, and files under `docs/` are safe to update when
the task asks for documentation. Keep them consistent with the current code.

## Safe checks

Use the project venv when it exists:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Use these targeted checks for simulator and cockpit changes:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_render_cli.py tests\test_simulator_upgrade.py tests\test_scenario_training_smoke.py
```

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_render_frame_snapshot.py tests\test_render_ui_flags.py tests\test_world_rules.py
```

Use this docs patch check before finishing docs work:

```powershell
git diff --check -- README.md CONTRIBUTING.md AGENTS.md docs
```

## Change rules

Follow these rules when editing the project:

- Add or update tests for every world-rule, observation, reward, CLI, render
  frame, or cockpit behavior change.
- Keep renderer code out of PPO logic and PPO code out of Raylib drawing.
- Add new render data to `agario_rl/rendering/models.py` and build it in
  `agario_rl/rendering/view_model.py`.
- Keep CLI flags documented in both `README.md` and `docs/quickstart.md`.
- Keep config fields documented in `docs/controls_and_tuning.md`.
- Run the targeted tests before the full test suite when the change touches
  simulator or cockpit behavior.
- Report exact verification commands and outcomes in the final response.

## Known gotchas

Keep these environment details in mind:

- The project path contains spaces. Use quoted paths in shell commands.
- On Windows PowerShell, use `Get-Content -LiteralPath` for paths that contain
  special characters.
- Full training writes checkpoints and CSV metrics. Use short smoke tests or
  temporary directories when you only need verification.
- The workload is often rollout-bound, so `--device xpu` is not automatically
  faster than CPU.
- Training on the public Agar.io website is a separate browser-automation and
  computer-vision project. This repo trains on the internal simulator.
