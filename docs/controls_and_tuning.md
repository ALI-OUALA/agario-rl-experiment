# Cockpit reference and tuning

This page is the operator reference for the Raylib observer cockpit. It
explains what each panel shows, which controls are available, and which config
groups have the highest leverage when you tune experiments.

## Cockpit layout

The runtime defaults to an Agar.io-style full-window arena. Compact telemetry
is drawn over the arena so training remains visible without shrinking the
world. Press `Tab` to open the full cockpit drawer when you need detailed
controls and charts.

### Arena viewport

The world viewport shows the active simulation with camera smoothing and render
interpolation. The simulation timestep stays authoritative, and rendering only
interpolates between stored cell positions.

The arena also includes:

- a leaderboard, score chip, minimap, and compact training metrics
- a status chip for checkpoint and runtime messages
- an optional world grid
- viruses, ejected mass, agent labels, and intent arrows when the upgraded
  scenario or high render detail is active

### Full telemetry drawer

The full drawer opens with `Tab` and groups the operator-facing information
into stable blocks.

- **Session cards** show state, speed, FPS, frame time, physics throughput,
  queue depth, and update count.
- **Training cards** show policy, value, entropy, imitation loss, and sync age.
- **Scenario strip** shows the active scenario name, stage, and preset.
- **Control surface** exposes the runtime actions as clickable buttons.
- **Agent observer cards** show alive state, mass, return, eliminations, wins,
  and focus state.
- **Reward breakdown** shows the strongest recent reward components for each
  agent.
- **Live telemetry** shows rolling mini-charts for FPS, reward, loss, wins,
  and updates.

## Control surface

The control surface is designed so every important action is available on
screen and through the keyboard.

### Buttons

The cockpit renders these main actions:

- **Pause** or **Resume**
- **Step tick**
- **Step decision**
- **Slower**
- **Faster**
- **Train More**
- **Curriculum**
- **Map -**
- **Map +**
- **Reset**
- **Save**
- **Load**
- **Grid**
- **Cockpit**
- **Fullscreen**
- **Help**

### Keyboard shortcuts

Use these shortcuts for the same actions:

- `Space`: pause or resume
- `N`: step one physics tick
- `Shift+N`: step one decision tick
- `-` and `+`: decrease or increase simulation speed
- `T`: toggle Train More
- `C`: toggle curriculum
- `R`: reset the episode and session wins
- `M`: increase map size
- `Shift+M`: decrease map size
- `1`, `2`, `3`: focus the camera on an agent
- `W`, `A`, `S`, `D` or arrow keys: switch to free camera and pan
- middle mouse drag: pan the world viewport
- `0`: return to agent-follow mode
- `Tab`: switch compact and full cockpit layouts
- `G`: toggle grid
- `F11`: toggle fullscreen
- `P`: save checkpoint
- `L`: load checkpoint
- `F1`: toggle the help overlay

## Status messages

The controller publishes short status messages in the viewport banner. You see
messages when you:

- change speed
- toggle training or curriculum
- change map size
- save or load a checkpoint
- focus the camera
- switch between follow camera and free camera
- switch cockpit density
- toggle fullscreen
- show or hide help

Missing checkpoints are surfaced as warnings instead of failing silently.

## High-impact tuning groups

Edit `config/default.yaml` when you want to change runtime behavior. These
groups usually matter most.

### Simulation

Use `simulation.*` to control pacing and camera feel.

- `physics_hz`: authoritative world tick rate
- `decision_hz`: policy decision frequency
- `max_substeps_per_frame`: frame backlog cap
- `camera_smoothness`: camera tracking response
- `zoom_smoothness`: zoom response
- `continuing_respawn`: respawn eliminated agents instead of ending the round
- `respawn_mass`: starting mass for continuing-mode respawns

### Physics

Use `physics.*` to change the environment itself.

- movement speed and drag
- split boost and cooldowns
- merge timing
- eating threshold and assimilation efficiency
- cell count cap
- ejected mass amount, cooldown, and speed

### Viruses and mass decay

Use these groups when you want the richer Agar.io-style simulator.

- `viruses.*`: virus count, mass, feeding threshold, split pieces, and spawn
  behavior
- `mass_decay.*`: passive mass decay rate and minimum mass floor

The classic preset leaves viruses and mass decay disabled by default.

### Observation and reward extensions

Use these groups when you want training signals that are more aligned with
visible play.

- `observation_features.*`: additive threat, target, virus, split-ready, and
  eject-ready features
- `reward_terms.*`: threat escape, target pressure, corner penalty, survival
  quality, split safety, virus split, and respawn reward components

These fields are additive. When `observation_features.enabled` is false, the
classic observation vector remains unchanged.

### Scenario presets

Use `scenario_curriculum.*` to label and stage the upgraded environment.
`agario_curriculum` moves through:

- `pellet_growth`
- `evasion`
- `hunting`
- `virus_control`
- `mixed_arena`
- `full_arena`

Early stages keep the world simpler. Later stages unlock virus behavior and
the full arena. `full_arena` skips the staged ramp and starts with the large
2000-style map, more agent slots, more pellets, viruses, eject support, and
continuing respawn.

### Rewards and PPO

Use `rewards.*` and `rl.*` when behavior quality matters more than visual
presentation.

- reward shaping determines aggression and survival bias
- split shaping discourages repeated split attempts, penalizes splits near
  larger threats or dangerous viruses, and rewards splits only when a close
  weak target makes the attack useful
- PPO clip, learning rate, and epoch count affect update stability
- `rl.split_logit_bias` and `rl.unready_split_logit_penalty` make the policy
  less eager to split before reward feedback arrives
- imitation settings affect how strongly the best agent influences the others

### Async training

Use `async_training.*` when the cockpit must stay responsive during training.

- rollout queue size affects backlog tolerance
- minimum transitions per job affects update cadence
- max pending weight updates affects how much stale work you keep around

### Rendering

Use `render.*` when you want to change the public runtime presentation.

- `window_width` and `window_height`: viewport size
- `side_panel_width`: full telemetry drawer width
- `start_fullscreen`: fullscreen startup toggle
- `window_resizable`: enable live resize behavior
- `overlay_mode_default`: compact or full startup layout
- `grid_enabled_default`: grid state at startup
- `fps`: target render FPS
- `show_agent_labels`: agent label visibility
- `show_score_chip`: bottom-left viewport chip visibility

## Scenario and showcase commands

Use these commands when you want the upgraded simulator to train or run as a
visible no-save showcase.

```powershell
python scripts/train.py --updates 20 --scenario-preset agario_curriculum --continuing-respawn --device auto
```

```powershell
python scripts/showcase.py
```

The supported scenario presets are `classic`, `agario_curriculum`, and
`full_arena`. The removed `scripts/supervise.py` command is no longer part of
the public workflow; use `scripts/showcase.py` for presentation and
`scripts/train.py` for training.

## Human-play fairness

The default human-play mode no longer gives the player a human-only eject
action. That earlier version was misleading because the player had a tactical
option the RL agents had never been trained to recognize or punish.

The fairest default comparison is now:

- same continuous movement interface for both sides
- same split mechanic for both sides
- no player-only eject shortcut

## Next steps

Use these follow-up pages when you need more depth:

1. Read `docs/runtime_architecture.md` for the snapshot and command model.
2. Read `docs/how_it_learns.md` for the PPO and imitation overview.
3. Read `README.md` for the complete project reference.
