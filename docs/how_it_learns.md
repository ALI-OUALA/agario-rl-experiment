# How learning works

This page is a short conceptual summary. For the complete implementation-level
explanation, formulas, and architecture diagrams, read `README.md`.

## Training loop

Shared-policy training follows a collect-then-update cycle:
1. Roll out transitions in the environment.
2. Build advantages and returns with GAE.
3. Run PPO optimization for several epochs.
4. Optionally add imitation loss from peer demonstrations.

Headless training collects rollouts and updates PPO directly. The showcase and
play surfaces render trained behavior, but they no longer own a separate
interactive training loop.

The upgraded scenario path keeps the same PPO loop. It changes the world and
training signals around the learner: viruses can split large cells, ejected
mass can feed viruses, mass can decay over time, and continuing respawn can
keep agents in the arena after elimination.

The repository now also includes a mixed-opponent training path in
`scripts/train_human_ready.py`. That path still uses PPO, but it only learns
for one tracked agent while the other slots are controlled by a frozen `500`
checkpoint and scripted opponents. The strongest scripted anchor is the
Agar-style objective bot: it flees larger threats, avoids dangerous viruses,
forages when neutral, chases smaller targets, and only splits on close weak
targets. The goal is to reduce the gap between self-play behavior and human
matches.

For presentation and smoke testing, `scripts/showcase.py` runs the large arena
without saving checkpoints or logs. It uses the human-ready checkpoint when it
can load and falls back to the scripted opponent pool when it cannot.

## PPO and imitation at a glance

- PPO uses a clipped policy objective to constrain update drift.
- Value loss regresses predicted returns.
- Entropy bonus supports exploration.
- Peer imitation stores transitions from the highest-return agent at episode
  end and injects behavior-cloning loss during updates.

## Scenario curriculum

The `agario_curriculum` preset enables a staged environment while keeping the
same public training entrypoints. The stage labels are:

- `pellet_growth`
- `evasion`
- `hunting`
- `virus_control`
- `mixed_arena`
- `full_arena`

Early stages keep the world easier to read and learn. Later stages unlock more
of the upgraded Agar.io behavior, including virus control and the full arena.
The `full_arena` preset skips the staged ramp and starts directly on the large
2000-style map with more agents, more pellets, viruses, eject support, and
continuing respawn.

## Observation and reward upgrades

The classic observation vector remains available. When
`observation_features.enabled` is true, the environment appends extra features
for nearby threats, smaller targets, viruses, split readiness, and eject
readiness.

Reward breakdowns are stored in each agent's `infos[agent_id]` payload. The
new terms can credit or penalize:

- increasing distance from a larger threat
- closing distance on a smaller target
- spending time in corners
- surviving in useful positions
- avoiding repeated split attempts
- avoiding splits near larger threats or dangerous viruses
- splitting only when a close weak target makes the attack useful
- splitting on viruses
- respawning after elimination

The Raylib cockpit reads these breakdowns through the immutable
`RenderFrame`, so you can inspect training behavior without coupling the
renderer to PPO internals.

## Where to read more

Use `README.md` for:
- Exact objective formulas.
- Network head definitions.
- Rollout and async update flows.
- Observation/action/reward schema details.
- Test-backed behavior guarantees.
Use `docs/controls_and_tuning.md` for:
- Scenario preset commands.
- Config fields for viruses, mass decay, observations, and reward terms.
- Cockpit overlay and render-detail behavior.

## Next steps

1. Run `python scripts/train.py --updates 20`.
2. Inspect `logs/train_metrics.csv`.
3. Compare runs with and without imitation (`rl.imitation_coef`).
4. If human-vs-agent performance matters, run
   `python scripts/train_human_ready.py --updates 80` and compare it with
   `python scripts/eval_human_readiness.py`.
5. If visible behavior matters, run `python scripts/showcase.py`.
