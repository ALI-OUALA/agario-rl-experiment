"""Interactive supervisor mode with fixed timestep and async training."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agario_rl import load_config
from agario_rl.env.gym_env import AgarioMultiAgentEnv
from agario_rl.rendering.view_model import build_render_frame
from agario_rl.rl.async_trainer import AsyncTrainerCoordinator
from agario_rl.rl.ppo_shared import SharedPPOTrainer
from agario_rl.supervisor.controller import SupervisorController
from agario_rl.supervisor.runtime_stats import RuntimeSessionStats
from agario_rl.utils.logging import TrainingMetricsLogger, maybe_log_training_metrics
from agario_rl.utils.seeding import set_global_seeds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run interactive supervision for Agar.io RL agents.")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--load-checkpoint", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--disable-async", action="store_true")
    parser.add_argument("--action-mode", type=str, choices=["continuous", "discrete_9way"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = PROJECT_ROOT
    config = load_config(project_root / args.config)
    if args.seed is not None:
        config.seed = int(args.seed)
    if args.action_mode is not None:
        config.simulation.action_mode = args.action_mode
    set_global_seeds(config.seed)

    env = AgarioMultiAgentEnv(config=config, enable_render=True)
    trainer = SharedPPOTrainer(config=config, observation_dim=env.observation_space["shape"][0])
    controller = SupervisorController(config=config)
    metrics_logger = TrainingMetricsLogger(project_root / config.logging.train_metrics_csv)
    last_logged_update = metrics_logger.max_update()

    if args.load_checkpoint:
        trainer.load(project_root / config.supervisor.checkpoint_path)
    trainer.force_sync_with_env(env, seed=config.seed)

    use_async = (
        (not args.eval_only)
        and config.async_training.enabled
        and (not args.disable_async)
    )
    async_coordinator: AsyncTrainerCoordinator | None = None
    if use_async:
        async_coordinator = AsyncTrainerCoordinator(config=config, observation_dim=env.observation_space["shape"][0])
        async_coordinator.start()
        async_coordinator.sync_from_trainer(trainer)

    physics_hz = max(1, int(config.simulation.physics_hz))
    decision_hz = max(1, int(config.simulation.decision_hz))
    physics_dt = 1.0 / physics_hz
    decision_dt = 1.0 / decision_hz
    physics_substeps_per_decision = max(1, int(round(physics_hz / decision_hz)))
    max_decisions_per_frame = max(1, int(config.simulation.max_substeps_per_frame))

    running = True
    latest_metrics = dict(trainer.last_metrics)
    decision_accumulator = 0.0
    last_wall_time = time.perf_counter()
    physics_steps_this_second = 0
    physics_steps_per_sec = 0.0
    perf_last_reset = time.perf_counter()
    frame_stats = {"frame_ms": 0.0, "render_fps": 0.0}
    runtime_stats = RuntimeSessionStats.create(env.agent_ids)

    while running:
        now = time.perf_counter()
        wall_dt = max(0.0, now - last_wall_time)
        last_wall_time = now
        decision_accumulator += wall_dt * controller.speed_multiplier
        max_backlog = decision_dt * max_decisions_per_frame
        decision_accumulator = min(decision_accumulator, max_backlog)
        interpolation_alpha = min(1.0, max(0.0, decision_accumulator / max(decision_dt, 1e-6)))

        if async_coordinator is not None:
            async_metrics = async_coordinator.poll_updates(trainer)
            if async_metrics:
                latest_metrics = async_metrics
                last_logged_update = maybe_log_training_metrics(
                    metrics_logger,
                    latest_metrics,
                    last_logged_update,
                )

        frame_metrics = {
            **latest_metrics,
            "speed_multiplier": controller.speed_multiplier,
            "auto_train_enabled": 1.0 if controller.auto_train_enabled else 0.0,
            "physics_steps_per_sec": physics_steps_per_sec,
            "worker_queue": float(async_coordinator.queue_depth() if async_coordinator else 0),
            "policy_sync_age_steps": float(trainer.policy_sync_age_steps),
            **runtime_stats.to_extra_stats(),
            "frame_ms": float(frame_stats.get("frame_ms", latest_metrics.get("frame_ms", 0.0))),
            "render_fps": float(frame_stats.get("render_fps", latest_metrics.get("render_fps", 0.0))),
        }
        render_frame = build_render_frame(
            config=config,
            world=env.world,
            infos=env.last_infos,
            metrics=frame_metrics,
            controller=controller,
            runtime_stats=runtime_stats,
            interpolation_alpha=interpolation_alpha,
            focus_agent_index=env.focus_agent_index,
        )
        frame_stats = env.render(frame=render_frame)
        frame_metrics["frame_ms"] = float(frame_stats.get("frame_ms", 0.0))
        frame_metrics["render_fps"] = float(frame_stats.get("render_fps", 0.0))
        runtime_stats.record_frame(frame_metrics, env.last_infos)

        commands = env.poll_commands()
        controller.handle_commands(commands)
        controller.apply_runtime_overrides(config=config, trainer=trainer, env=env)
        if async_coordinator is not None and controller.events.load_requested:
            async_coordinator.sync_from_trainer(trainer)

        if controller.events.quit_requested:
            running = False
            continue

        if controller.events.reset_requested:
            trainer.force_sync_with_env(env)
            runtime_stats.reset_wins()
            decision_accumulator = 0.0
            controller.set_status("Episode reset and session wins cleared.")
            continue

        if controller.events.step_tick_once:
            step_infos = trainer.step_physics_with_last_action(env, dt=physics_dt)
            runtime_stats.record_infos(step_infos)
            physics_steps_this_second += 1
        elif controller.events.step_policy_once:
            step_infos = trainer.step_decision(
                env=env,
                substeps=physics_substeps_per_decision,
                dt=physics_dt,
                track_experience=not args.eval_only,
                deterministic=False,
            )
            runtime_stats.record_infos(step_infos)
            physics_steps_this_second += physics_substeps_per_decision
        elif not controller.paused:
            processed = 0
            while decision_accumulator >= decision_dt and processed < max_decisions_per_frame:
                step_infos = trainer.step_decision(
                    env=env,
                    substeps=physics_substeps_per_decision,
                    dt=physics_dt,
                    track_experience=not args.eval_only,
                    deterministic=False,
                )
                runtime_stats.record_infos(step_infos)
                physics_steps_this_second += physics_substeps_per_decision
                decision_accumulator -= decision_dt
                processed += 1

        if (not args.eval_only) and controller.auto_train_enabled:
            if async_coordinator is not None:
                if trainer.transitions_since_update >= config.async_training.min_rollout_transitions_per_job:
                    if async_coordinator.can_submit():
                        payload = trainer.prepare_update_job_payload()
                        if payload is not None:
                            async_coordinator.submit_update(payload)
            elif trainer.ready_to_update():
                latest_metrics = trainer.update()
                last_logged_update = maybe_log_training_metrics(
                    metrics_logger,
                    latest_metrics,
                    last_logged_update,
                )

        if async_coordinator is None:
            latest_metrics = dict(trainer.last_metrics)

        elapsed = time.perf_counter() - perf_last_reset
        if elapsed >= 1.0:
            physics_steps_per_sec = physics_steps_this_second / elapsed
            physics_steps_this_second = 0
            perf_last_reset = time.perf_counter()
            latest_metrics["frame_ms"] = float(frame_stats.get("frame_ms", 0.0))
            latest_metrics["render_fps"] = float(frame_stats.get("render_fps", 0.0))

    trainer.save(project_root / config.supervisor.checkpoint_path)
    if async_coordinator is not None:
        async_coordinator.shutdown()
    env.close()


if __name__ == "__main__":
    main()
