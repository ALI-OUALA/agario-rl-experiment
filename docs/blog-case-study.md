# Case study: browser reset

The project started as a Python RL lab with a native visual surface. That made
simulation work possible, but the user-facing game did not feel like a modern
Agar.io-style browser game and was hard to verify with browser automation.

The reset keeps the strongest part of the project: Python remains the source of
truth for world rules, rewards, observations, PPO, opponents, and checkpoints.
Only the presentation layer moved to the browser.

## What changed

- Added FastAPI and a WebSocket runtime around `AgarioMultiAgentEnv`.
- Added a Vite + TypeScript canvas app under `web/`.
- Added `python scripts/run_game.py` as the single public game command.
- Removed the old native UI, reference UI file, and native UI tests.
- Rewrote documentation around running, training, and evaluating.

## Why this shape is useful

The browser is good at smooth canvas rendering, layout, responsive HUDs, and
automation through Browser Use or Playwright. Python is good at deterministic
simulation, reproducible training, and direct tests around rewards and world
rules. Keeping those roles separate avoids browser/Python behavior drift.

## What users can see

The game now shows the full arena context through camera smoothing, minimap,
leaderboard, off-screen threat indicators, visible viruses, pellets, ejected
mass, and live training state. The UI is meant to explain the experiment while
still feeling like a playable game.

## What remains research work

Better human-facing behavior still depends on training quality. The UI exposes
split safety and reward-related counters, but the policy needs continued mixed
opponent training and evaluation to prove that agents behave well with humans.
