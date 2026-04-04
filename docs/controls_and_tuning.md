# Cockpit reference and tuning

This page is the operator reference for the Raylib observer cockpit. It
explains what each panel shows, which controls are available, and which config
groups have the highest leverage when you tune experiments.

## Cockpit layout

The runtime uses a single-window layout with a world viewport on the left and a
control and telemetry panel on the right.

### World viewport

The world viewport shows the active simulation with camera smoothing and render
interpolation. The simulation timestep stays authoritative, and rendering only
interpolates between stored cell positions.

The viewport also includes:

- a status banner for checkpoint and runtime messages
- a focus chip showing the active camera mode and target
- an optional world grid

### Side panel

The side panel groups the operator-facing information into stable blocks.

- **Session cards** show state, speed, FPS, frame time, physics throughput,
  queue depth, and update count.
- **Training cards** show policy, value, entropy, imitation loss, and sync age.
- **Control surface** exposes the runtime actions as clickable buttons.
- **Agent observer cards** show alive state, mass, return, eliminations, wins,
  and focus state.
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

### Physics

Use `physics.*` to change the environment itself.

- movement speed and drag
- split boost and cooldowns
- merge timing
- eating threshold and assimilation efficiency
- cell count cap

### Rewards and PPO

Use `rewards.*` and `rl.*` when behavior quality matters more than visual
presentation.

- reward shaping determines aggression and survival bias
- PPO clip, learning rate, and epoch count affect update stability
- imitation settings affect how strongly the best agent influences the others

### Async training

Use `async_training.*` when the cockpit must stay responsive during training.

- rollout queue size affects backlog tolerance
- minimum transitions per job affects update cadence
- max pending weight updates affects how much stale work you keep around

### Rendering

Use `render.*` when you want to change the public runtime presentation.

- `window_width` and `window_height`: viewport size
- `side_panel_width`: full cockpit side panel width
- `start_fullscreen`: fullscreen startup toggle
- `window_resizable`: enable live resize behavior
- `overlay_mode_default`: compact or full startup layout
- `grid_enabled_default`: grid state at startup
- `fps`: target render FPS

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
