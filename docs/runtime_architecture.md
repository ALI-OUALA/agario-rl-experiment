# Runtime architecture

This page explains how the interactive runtime is wired after the observer
cockpit upgrade. The key change is that rendering no longer consumes
renderer-specific input events or reaches directly into trainer internals.

## Design goals

The runtime architecture is built around four constraints:

- the simulation timestep stays authoritative
- rendering never mutates world or trainer state mid-frame
- operator input is semantic, not backend-specific
- the cockpit can expose rich telemetry without coupling itself to PPO code

## Main components

The interactive runtime is built from four pieces.

### World and environment

`AgarioWorld` owns the deterministic simulation. `AgarioMultiAgentEnv` wraps
that world and exposes reset, step, render, and renderer command polling.

### Trainer and async worker

`SharedPPOTrainer` owns rollout collection, PPO updates, imitation replay, and
checkpoint save and load. `AsyncTrainerCoordinator` optionally runs PPO updates
off the main thread so the cockpit stays responsive.

### Supervisor controller

`SupervisorController` is the runtime state machine. It consumes semantic
`SupervisorCommand` values such as `toggle_pause`, `step_decision`,
`toggle_auto_train`, or `load_checkpoint`. It updates operator state, exposes
transient events for the main loop, and publishes short status messages for the
cockpit banner.

### Render snapshot layer

`build_render_frame()` converts world state, trainer metrics, controller state,
and telemetry history into an immutable `RenderFrame`. The renderer reads that
payload and nothing else.

## Runtime flow

Each interactive frame follows the same sequence.

1. Poll async trainer updates and merge the latest metrics.
2. Build one `RenderFrame` from world state, metrics, controller state, and
   rolling session history.
3. Draw the Raylib cockpit from that frame.
4. Poll semantic commands from Raylib.
5. Let the controller apply those commands.
6. Apply environment and trainer side effects such as map changes or
   checkpoint actions.
7. Run physics steps, policy decisions, and optional PPO updates.
8. Record win counts and telemetry history for the next frame.

## Why the snapshot model matters

The snapshot model keeps the runtime easier to evolve because:

- the renderer does not depend on PPO implementation details
- the controller does not depend on Raylib key codes
- tests can validate frame construction separately from window creation
- local camera and fullscreen behavior stay isolated in the renderer

## Semantic command model

The renderer emits typed `SupervisorCommand` values. Current command categories
include:

- lifecycle: `quit`, `reset_episode`
- pacing: `toggle_pause`, `step_physics`, `step_decision`, `speed_delta`
- training: `toggle_auto_train`, `toggle_curriculum`
- environment: `map_scale`, `focus_agent`
- checkpointing: `save_checkpoint`, `load_checkpoint`
- presentation: `toggle_overlay_mode`, `toggle_grid`, `toggle_help`

This keeps keyboard and mouse actions behaviorally equivalent.

## Telemetry history

`RuntimeSessionStats` tracks both session win counters and short rolling metric
history used by the charts. The current cockpit charts track:

- render FPS
- frame time
- mean reward
- total loss
- total wins
- update count

The chart history is lightweight and exists only to support operator
visibility.

## Renderer boundary

The public runtime now uses a fixed Raylib renderer. The environment still
builds rendering through a small factory boundary so the simulation and the
cockpit stay decoupled, but there is no public backend selection path anymore.

## Next steps

Use these related pages when you need more context:

1. Read `docs/quickstart.md` for first-run instructions.
2. Read `docs/controls_and_tuning.md` for the cockpit reference.
3. Read `README.md` for the broader project and learning overview.
