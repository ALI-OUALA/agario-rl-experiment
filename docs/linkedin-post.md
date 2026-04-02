# LinkedIn post draft

I just finished documenting my Agar.io reinforcement learning experiment and
pushed the main checkpoint from update `366` to update `500`.

The setup is a deterministic Agar.io-style world with `3` agents sharing one
PPO policy. I kept the RL contract stable the whole time: same observation
space, same continuous action interface, same reward shaping, and the same
shared-policy setup.

The behavior changes were the most interesting part:

- Around update `100`, one agent would eat another and the remaining agents
  often drifted to corners. It looked like short-term reward bias.
- Around update `300`, the agents started behaving much more naturally. They
  chased smaller enemies, backed off from larger ones, and moved through the
  map with better intent.
- I preserved update `366` as a formal milestone before continuing training.
- I then resumed that exact checkpoint to update `500` with cleaner logging and
  a dedicated human-playable mode.

One honest takeaway: the losses at update `500` look much cleaner than they did
at `366`, but my quick deterministic 5-episode evaluation did not improve with
it. That is a useful result, not a bad one. It means the next step is better
evaluation, not just more training.

I also added:

- a resume-aware training path
- canonical training metrics logging
- a Raylib observer cockpit
- a playable `1 human + 2 agents` mode

What I like most about this project is that the repo now tells the real story,
including the awkward early behavior, the better mid-training behavior, the
logging gap, and the mixed result at `500`.

If you want, I can share the repo write-up, the training charts, and the
play-mode clip next.
