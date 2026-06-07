# How it learns

The agent learns from the internal simulator, not from screen pixels. Every
training step uses structured observations from `AgarioWorld`, actions from the
PPO policy, and reward terms computed from simulator state.

## Action space

The continuous action is:

```text
[steer_x, steer_y, split]
```

`steer_x` and `steer_y` are normalized direction values. `split` is interpreted
as a binary intent.

## Observations

The observation vector includes the controlled agent, nearby pellets, nearby
opponents, and optional richer features:

- larger threats
- smaller targets
- viruses
- split/eject readiness
- scenario state

The browser renderer does not change observations. It only visualizes the same
world state.

## PPO loop

`scripts/train.py` creates `AgarioMultiAgentEnv` and `SharedPPOTrainer`.

Each update:

1. collect rollout transitions from the simulator
2. compute PPO losses and imitation loss
3. update the shared policy
4. append metrics to `logs/train_metrics.csv`
5. save checkpoints on the configured interval

## Human-ready training

`scripts/train_human_ready.py` is the recommended path when the goal is agents
that behave reasonably around humans. It trains with a mixed opponent pool:

- checkpoint anchor
- pellet forager
- threat-aware evader
- opportunistic hunter
- objective-driven Agar bot

This prevents the policy from overfitting to one opponent style and makes
survival, chasing, fleeing, and split timing more visible.

## Reward priorities

The current reward shaping is designed to avoid split spam:

- useful split bonus only helps when the split is tactically useful
- unsafe split penalty punishes splitting into danger
- threat escape rewards moving away from larger enemies
- target pressure rewards chasing smaller targets without reckless splits
- corner penalty discourages passive hiding
- survival quality rewards alive, growing behavior
- respawn penalty makes deaths matter in continuing mode

The browser UI exposes these ideas through split counters, human-readiness
metrics, leaderboard mass, policy source, and the latest training update.
