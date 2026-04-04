# How learning works

This page is a short conceptual summary. For the complete implementation-level
explanation, formulas, and architecture diagrams, read `README.md`.

## Training loop

Shared-policy training follows a collect-then-update cycle:
1. Roll out transitions in the environment.
2. Build advantages and returns with GAE.
3. Run PPO optimization for several epochs.
4. Optionally add imitation loss from peer demonstrations.

Supervisor mode can offload updates to the async worker so rendering stays
responsive while training continues.

The repository now also includes a mixed-opponent training path in
`scripts/train_human_ready.py`. That path still uses PPO, but it only learns
for one tracked agent while the other slots are controlled by a frozen `500`
checkpoint and scripted opponents. The goal is to reduce the gap between
self-play behavior and human matches.

## PPO and imitation at a glance

- PPO uses a clipped policy objective to constrain update drift.
- Value loss regresses predicted returns.
- Entropy bonus supports exploration.
- Peer imitation stores transitions from the highest-return agent at episode
  end and injects behavior-cloning loss during updates.

## Where to read more

Use `README.md` for:
- Exact objective formulas.
- Network head definitions.
- Rollout and async update flows.
- Observation/action/reward schema details.
- Test-backed behavior guarantees.

## Next steps

1. Run `python scripts/train.py --updates 20`.
2. Inspect `logs/train_metrics.csv`.
3. Compare runs with and without imitation (`rl.imitation_coef`).
4. If human-vs-agent performance matters, run
   `python scripts/train_human_ready.py --updates 80` and compare it with
   `python scripts/eval_human_readiness.py`.
