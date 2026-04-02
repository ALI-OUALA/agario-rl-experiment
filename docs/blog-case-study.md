# From weird corner camping to playable bots: a PPO Agar.io case study

This draft is written as a hybrid case study. It explains what the project is,
how the reinforcement learning loop works, what the agents actually did at key
milestones, and why the final story is more interesting than a simple
"training went up, therefore it worked" claim.

## The project in one paragraph

I built a deterministic Agar.io-style world with three agents that all share a
single PPO policy. The environment includes pellets, split, merge, cell
eating, and lightweight reward shaping around mass gain, eliminations,
survival, and death. Around that core loop, I added a Raylib observer cockpit
for training control and a new human-playable mode so I can test the agents
directly instead of relying on charts alone.

## Why this experiment was worth documenting

The interesting part of this project is not just that PPO can move circles
around a map. The interesting part is how the behavior changed as the policy
began to trade off aggression, survival, and spatial control.

At first, the agents found brittle local wins. Later, they started to show the
kind of predator-prey behavior you expect in Agar.io: chase what is smaller,
run from what is bigger, and keep moving through live space rather than
freezing into the map corners.

## The learning setup

The environment stays intentionally compact so behavior is easier to reason
about.

- The agents share one PPO policy.
- The observation vector encodes self-state, nearby pellets, nearby opponents,
  and a small amount of global context.
- The action interface stays fixed as continuous steering plus a split flag.
- The reward function emphasizes mass gain, eliminations, death, and late
  survival.
- Peer imitation adds behavior-cloning pressure from the best trajectory at
  episode end.

That design lets the experiment stay small enough to understand while still
being rich enough to produce recognizable game behavior.

## The milestone story

The milestone story is what makes the experiment feel real.

### Update 100: the weird phase

At about update `100`, one agent would often eat another, and then the two
remaining agents drifted to the corners. That looked like early reward
hacking. The policy had learned that immediate local safety and opportunistic
eating were useful, but it had not learned to keep searching the map in a way
that looked natural.

### Update 300: the first believable phase

At about update `300`, the agents started acting in a much more natural way.
They chased smaller enemies, backed off from larger enemies, and moved through
the world with more believable intent. This was the first point where the
training curves and the on-screen behavior started telling the same story.

### Update 366: the baseline I had to preserve

By the time the checkpoint reached update `366`, the repository had a real
policy state but not a complete CSV history for the interactive phase that got
it there. I did not want to fake continuity, so I treated update `366` as a
formal milestone instead.

That checkpoint had:

- policy loss `0.0080`
- value loss `63.5176`
- entropy `2.7028`
- imitation loss `0.8676`
- total loss `31.9133`
- deterministic 5-episode average return `-2.211`

### Update 500: cleaner losses, weaker quick evaluation

I resumed the exact update `366` checkpoint to update `500` with a new
resume-aware training path and canonical logging. The final checkpoint landed
cleanly at `checkpoints/checkpoint_00500.pt`.

The final metrics were much cleaner:

- policy loss `-0.0160`
- value loss `0.2942`
- entropy `2.8653`
- imitation loss `0.9404`
- total loss `0.2905`

But the quick deterministic evaluation got worse, not better. The 5-episode
average return moved from `-2.211` at update `366` to `-6.284` at update
`500`.

That is the most honest part of the story. The optimizer stabilized, but the
small benchmark slice did not improve with it.

![Training summary chart for the update 1..200 and 367..500 series](assets/training-summary.png)

## Why that result is still useful

A worse quick benchmark does not make the experiment a failure. It means the
project is now at the point where better evaluation matters more than more
confidence.

The repo now has three things it did not have before:

1. A clean continuation path that resumes to a target update milestone.
2. Canonical logging that stops the CSV from fragmenting into duplicate runs.
3. A human-playable mode that lets me test whether the agents feel smart, not
   just whether a loss number got smaller.

That is a better engineering state than before, even if the headline metric is
messier than I would like.

## What changed in the codebase

The technical changes were straightforward but important.

- `scripts/train.py` now resumes from saved checkpoints and keeps global update
  numbering intact.
- `scripts/supervise.py` now writes interactive updates into the same canonical
  metrics CSV as headless training.
- `scripts/play.py` adds a dedicated human-vs-bots mode with mouse steering,
  split, and human-only mass eject.
- `scripts/generate_report_assets.py` builds the local charts used in the
  docs.

The trained-agent contract did not change. I kept the same observation schema,
action interface, reward shaping, and policy weights so the experiment story
would stay comparable across milestones.

## What I would do next

The next technical move is not "train longer and hope." The next move is to
improve evaluation.

- Track win rate across larger episode sets.
- Measure survival time and final mass, not only mean return.
- Compare play-mode feel across milestones instead of only reading PPO losses.
- Keep the same RL contract until I have a better benchmark story.

![Evaluation comparison between the preserved update 366 checkpoint and the new update 500 checkpoint](assets/eval-comparison.png)

## Closing

This experiment is now in a good state for sharing because it tells a real
story. The agents progressed from awkward local behavior to something more
recognizable, the tooling got better, and the final numbers forced a more
honest conclusion than a victory lap.

That is the kind of RL project I actually like reading about: one where the
behavior, the code, and the uncertainty are all visible.
