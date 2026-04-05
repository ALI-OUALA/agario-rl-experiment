# Experiment results

This page captures the current public state of the Agar.io reinforcement
learning experiment. It preserves the update `366` checkpoint as a reportable
milestone, records the continued run to update `500`, documents the first
mixed-opponent retrain, and explains the logging and evaluation gaps honestly
instead of inventing missing data.

## Summary

The experiment now has three relevant checkpoints:

- update `366`, the preserved pre-resume snapshot
- update `500`, the continued self-play line
- `human_ready_v1` update `80`, a fresh retrain against stronger opponents

![Training summary chart with the interactive logging gap highlighted](assets/training-summary.png)

## Milestone observations

The most useful observations are still qualitative. They explain how the
policy looked to a human observer before the quantitative artifacts were
strong enough to stand on their own.

### Around update 100

At roughly update `100`, one agent would often eat another and then the two
remaining agents drifted toward the corners. That behavior is consistent with
an early short-term reward bias: agents were learning that immediate survival
and opportunistic elimination worked, but they were not yet sustaining a more
natural predator-prey loop across the whole map.

### Around update 300

At roughly update `300`, the behavior looked more natural. Agents were more
likely to chase smaller opponents, disengage from larger opponents, and trade
space in a way that resembled recognizable Agar.io instincts instead of simply
collapsing into corners.

### Update 366 baseline

Update `366` is the last stable checkpoint before the continuation run. The
repository did not have canonical CSV logging for every interactive update in
that phase, so this milestone is preserved as a checkpoint snapshot plus a
deterministic evaluation summary.

| Metric | Value at update `366` |
| --- | ---: |
| Policy loss | `0.0080` |
| Value loss | `63.5176` |
| Entropy | `2.7028` |
| Imitation loss | `0.8676` |
| Total loss | `31.9133` |
| Batch size | `1038` |
| Deterministic evaluation average return | `-2.211` |

The deterministic 5-episode baseline at update `366` produced these winners:

- Episode 1: `agent_1`
- Episode 2: `agent_0`
- Episode 3: `agent_2`
- Episode 4: `agent_2`
- Episode 5: `agent_2`

### Update 500 continuation

Update `500` came from continuing the update `366` checkpoint with the new
resume-aware training path. The final checkpoint was written to
`checkpoints/checkpoint_00500.pt`, and the canonical CSV now includes updates
`367..500`.

| Metric | Value at update `500` |
| --- | ---: |
| Policy loss | `-0.0160` |
| Value loss | `0.2942` |
| Entropy | `2.8653` |
| Imitation loss | `0.9404` |
| Total loss | `0.2905` |
| Batch size | `2049` |
| Deterministic evaluation average return | `-6.284` |

The deterministic 5-episode evaluation at update `500` produced these winners:

- Episode 1: `agent_1`
- Episode 2: `agent_2`
- Episode 3: `agent_1`
- Episode 4: `agent_2`
- Episode 5: `agent_0`

![Evaluation comparison between the update 366 and update 500 checkpoints](assets/eval-comparison.png)

## How to read the current results

The continuation run improved the final reported losses dramatically, but the
quick deterministic evaluation did not improve with it. In other words, the
optimizer looks more settled by update `500`, while the five-episode
evaluation slice looks worse than the `366` baseline.

That does not prove the policy regressed globally. It does mean the current
public claim must stay conservative: the agents look more structured than they
did early in training, but the evaluation story is not yet strong enough to
claim a clear performance jump from `366` to `500`.

## Logging coverage

The current CSV is canonical, but it is not continuous. The repository logs
updates `1..200`, then a gap, then updates `367..500`. The missing interval
comes from an older interactive-training phase that advanced the checkpoint
without recording every update to `logs/train_metrics.csv`.

The repo does not fabricate rows for that missing interval. Instead, it keeps
the `366` checkpoint as a named milestone and shows the logging gap directly.

![Logged update coverage showing the interactive-only gap between 200 and 367](assets/update-coverage.png)

## Human-play mode

The repository now includes `scripts/play.py`, which places one human player
into the same three-slot world as the trained agents. This mode exists to help
you answer a question that the quick deterministic evaluation cannot fully
answer on its own: does the policy feel smarter to play against, or does it
only look more numerically stable?

Play mode keeps the trained-agent contract intact:

- the RL agents still use the same observation schema
- the RL agents still use the same continuous action interface
- the RL agents still use the same reward shaping and shared policy weights
- the player no longer gets a human-only eject action in default play mode

That last point matters. The previous version of play mode gave the human an
extra mass-eject action that the RL agents had never been trained to answer.
That meant the human comparison was unfair before the bots even made a
decision.

## Human-readiness retrain

The first retrain against stronger opponents was designed to attack the exact
gap that showed up in play mode: the agents looked passable in self-play, but
humans could still exploit them easily.

### What changed

The `human_ready_v1` run started from scratch and trained one learner agent at
a time against a rotating pool of opponents:

- the frozen `checkpoint_00500.pt` policy
- a pellet-foraging scripted opponent
- a threat-aware evasive scripted opponent
- an opportunistic hunter scripted opponent

This makes the learner see behaviors that mirror self-play alone was not
producing consistently.

### What the new metrics mean

The new human-readiness evaluation uses proxy metrics that map to things a
human actually notices in a match:

- `win_rate`: whether the learner can actually finish mixed-opponent episodes
  on top
- `mean_survival_steps`: how long it stays alive
- `mean_final_mass`: how much total control it ends with
- `corner_time_fraction`: how often it hides in corners instead of contesting
  space
- `threat_avoidance_rate`: how often it increases distance from larger nearby
  threats
- `small_target_pressure_rate`: how often it closes distance on smaller nearby
  targets

These are not perfect human evaluations, but they are much closer to
human-vs-agent failure modes than raw PPO loss alone.

### Baseline vs retrain

The current `500` checkpoint and the first mixed-opponent retrain look like
this on the 20-episode human-readiness benchmark:

| Checkpoint | Win rate | Mean survival | Mean final mass | Corner time | Threat avoidance | Small-target pressure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Self-play update `500` | `0.00` | `448.0` | `96.879` | `0.595` | `0.205` | `0.200` |
| `human_ready_v1` update `80` | `0.00` | `502.65` | `14.0` | `0.641` | `0.248` | `1.000` |

On the older deterministic 5-episode self-play evaluation, the
`human_ready_v1` checkpoint also came in worse than the self-play `500`
checkpoint, with an average return of `-14.752`.

This is an honest mixed result:

- the retrain survives longer
- it responds to nearby threats slightly better
- it pressures smaller targets much more consistently
- it still does not win the benchmark
- it collapses into low final mass and too much corner time

So the first retrain improved some human-relevant instincts, but it is not yet
strong enough to call the human-vs-agent problem solved.

## Speed and hardware findings

The repo now supports explicit torch device selection, including Intel XPU.
That means the trainer can target:

- `cpu`
- `cuda` when available
- `xpu` on supported Intel Arc hardware

On the author machine used for this pass, the Intel Arc A370M was detected
successfully with `torch 2.11.0+xpu`. However, the benchmark result was still
CPU-faster for this workload:

| Device setup | Rollout mean | PPO update mean |
| --- | ---: | ---: |
| CPU train + CPU inference | `4.15s` | `1.03s` |
| XPU train + CPU inference | `5.09s` | `7.54s` |

This matters because the project is not purely matrix-multiply bound. A large
share of the total wall time still comes from environment stepping, rollout
collection, Python control flow, and frequent small-batch inference. In other
words, Intel Arc support works here, but it is not yet a net speed win for
this specific repo.

The current speed work still improved the engineering state:

- the trainer now supports explicit `--device` and `--inference-device`
  choices
- training logs now record rollout time and PPO update time separately
- the benchmark script now exposes a `train` mode so you can test CPU vs XPU
  directly before committing to a long run

## Recommended next steps

The experiment is in a good public state, but the next research step is clear.

1. Spend time in play mode and the observer cockpit to compare update `500`
   behavior against the earlier `300` and `366` impressions.
2. Add richer evaluation metrics, such as win rate, final mass, and survival
   time, before treating `500` as a stronger checkpoint than `366`.
3. If you keep training past `500`, preserve the observation/action contract so
   future write-ups remain comparable.
