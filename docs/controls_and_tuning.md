# Controls and tuning

## Browser controls

`play` mode controls:

- Mouse position steers `agent_0`.
- Space sends one split action.
- R resets the arena.
- The Reset button sends the same reset message.
- The mode selector reconnects the WebSocket with `showcase`, `play`, or
  `training-view`.

The browser sends normalized continuous actions: `[steer_x, steer_y, split]`.
Python consumes the split flag once so holding Space does not spam every frame.

## Game modes

- `showcase`: no human player; all agents are policy-controlled.
- `play`: `agent_0` is human-controlled and the rest are mixed opponents.
- `training-view`: no-save view that emphasizes training and human-readiness
  telemetry.

All modes use the same simulator rules.

## Important config groups

`simulation`

- `physics_hz`: physics stepping rate.
- `decision_hz`: action decision rate.
- `continuing_respawn`: respawn eliminated agents in long matches.
- `respawn_mass`: mass used for respawned agents.

`viruses`

- `enabled`: enable virus entities.
- `initial_count` and `max_count`: arena density.
- `min_split_mass`: mass threshold where viruses become dangerous.
- `feed_to_split`: fed mass required before a virus creates another virus.

`mass_decay`

- `enabled`: passive mass loss.
- `per_second`: decay rate.
- `min_mass`: lower bound for decay.

`observation_features`

- `include_threats`: add larger-opponent features.
- `include_viruses`: add virus features.
- `include_eject_state`: add eject readiness.

`reward_terms`

- `threat_escape_scale`: reward moving away from dangerous larger cells.
- `target_pressure_scale`: reward pressuring smaller targets.
- `corner_penalty`: discourage hiding in corners.
- `survival_quality_scale`: reward staying alive with useful mass.
- `split_attempt_penalty`: discourage random split spam.
- `unsafe_split_penalty`: penalize bad split attempts.
- `useful_split_bonus`: reward safe, useful split attacks.
- `virus_split_bonus`: reward useful virus interactions.
- `respawn_penalty`: cost for getting eliminated in continuing mode.

`checkpoint`

- `latest_path`: default checkpoint path for train/eval commands.

## Scenario presets

`classic` keeps the smaller baseline simulator behavior.

`agario_curriculum` enables viruses, mass decay, observation extensions, reward
terms, and staged curriculum labels for training.

`full_arena` expands the map and population for the browser game, enables
continuing respawn, and makes the UI feel closer to a large Agar.io match.

## Human-readiness tuning

For agents that should behave well with humans, prefer:

```powershell
python scripts/train_human_ready.py --updates 80 --device auto
```

That path uses a mixed opponent pool, which is more useful for human-facing
behavior than self-play against one narrow policy. Watch these UI counters:

- useful splits rising while unsafe splits stay low
- split safety staying positive
- threat escapes improving
- final mass increasing without constant corner hiding
- deaths after split not increasing

## Device tuning

For weaker devices, keep the browser in `showcase` or `play` mode and train
short CPU smoke runs first:

```powershell
python scripts/train.py --updates 5 --device cpu
```

For stronger machines, use `--device auto` so PyTorch can choose CUDA or XPU
when the correct wheels are installed. If rollout collection dominates runtime,
CPU can still be competitive because the simulator loop is often the bottleneck.
