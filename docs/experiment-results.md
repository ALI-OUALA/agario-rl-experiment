# Experiment results

This page summarizes the result files currently tracked in the repository. It
separates training diagnostics from gameplay evaluation because PPO loss is not
a gameplay score.

## Data used

| Run | Training log | Logged updates | Evaluation |
| --- | --- | ---: | ---: |
| Recovered baseline | `logs/train_metrics_recovered.csv` | 500 | 20 episodes |
| Human-ready v1 | `logs/human_ready_v1_train_metrics.csv` | 80 | 20 episodes |

The original baseline CSV is preserved at `logs/train_metrics.csv`. Its missing
updates could not be reconstructed exactly, so `checkpoint_00200.pt` was resumed
and updates 201–500 were trained again on CPU. The recovered CSV combines the
original updates 1–200 with that single continuous recovered trajectory. It has
500 unique rows, no missing updates, and timing data for every recovered update.

![Training update coverage](assets/update-coverage.png)

## Gameplay evaluation

The evaluation files compare `checkpoints/checkpoint_00500.pt` with
`checkpoints/human_ready_v1/latest.pt`.

| Metric | Baseline | Human-ready v1 | Change |
| --- | ---: | ---: | ---: |
| Win rate | 0.00% | 0.00% | 0.00 points |
| Mean survival | 448.00 steps | 502.65 steps | +54.65 steps |
| Mean final mass | 96.88 | 14.00 | -82.88 |
| Time in corners | 59.46% | 64.14% | +4.68 points |
| Threat avoidance | 20.53% | 24.77% | +4.24 points |
| Small-target pressure | 20.00% | 100.00% | +80.00 points |

![Human-readiness metrics](assets/eval-comparison.png)

Human-ready v1 survived 12.20% longer, improved the threat-avoidance proxy by
4.24 percentage points, and raised small-target pressure from 20% to 100%.
Those gains came with an 85.55% drop in mean final mass and 4.68 percentage
points more corner time. Both policies recorded zero wins.

The honest conclusion: the mixed-opponent run changed behavior, but this
evaluation does not show a stronger overall player. More seeds and more
episodes are needed before treating the proxy improvements as reliable.

## Training diagnostics

The chart uses a 10-update rolling mean. Missing updates remain missing.

![PPO training diagnostics](assets/training-summary.png)

| Diagnostic | Baseline first 20 | Baseline last 20 | Human-ready first 20 | Human-ready last 20 |
| --- | ---: | ---: | ---: | ---: |
| Total loss | 0.9722 | 0.4970 | 2.5837 | 2.4654 |
| Value loss | 1.6094 | 0.6911 | 4.8545 | 4.6064 |
| Entropy | 2.5446 | 3.0330 | 2.5402 | 2.6106 |

Recovered baseline total loss fell 48.88% between its first and last 20 updates.
Human-ready total loss fell 4.58%. These values describe optimizer behavior;
they do not override the gameplay evaluation above.

The recovered 300-update segment took 2,483.62 seconds of measured rollout and
optimization time: 8.28 seconds per update on average and 5.21 seconds median.

## Reproduce

Regenerate all charts from the tracked data:

```powershell
python scripts/generate_report_assets.py
```

Recovery command used:

```powershell
python scripts/train.py --resume --updates 500 --device cpu --checkpoint checkpoints/recovery_201_500/latest.pt --checkpoint-dir checkpoints/recovery_201_500 --metrics-csv logs/train_metrics_recovered.csv
```

Run a fresh evaluation:

```powershell
python scripts/eval_human_readiness.py --checkpoint checkpoints/checkpoint_00500.pt --episodes 20
python scripts/eval_human_readiness.py --checkpoint checkpoints/human_ready_v1/latest.pt --episodes 20
```

When publishing a comparison, include the checkpoint path, seed, episode count,
scenario preset, and command output.
