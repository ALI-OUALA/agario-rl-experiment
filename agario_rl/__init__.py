"""Public package interface for the Agar.io RL experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class MapConfig:
    start_size: int = 256
    max_size: int = 1024
    pellets_per_10k_area: int = 26
    pellet_mass: float = 1.0
    pellet_respawn_per_step: int = 4


@dataclass(slots=True)
class PhysicsConfig:
    radius_scale: float = 3.6
    base_speed: float = 3.5
    speed_mass_factor: float = 0.28
    drag: float = 0.82
    eat_mass_ratio: float = 1.15
    assimilation_efficiency: float = 0.94
    split_cooldown_steps: int = 20
    merge_cooldown_steps: int = 60
    split_boost: float = 5.2
    max_cells_per_agent: int = 4
    min_split_mass: float = 16.0
    enable_eject_mechanic: bool = False
    eject_mass_amount: float = 2.0
    eject_cooldown_steps: int = 8


@dataclass(slots=True)
class RewardConfig:
    mass_gain_scale: float = 0.06
    elimination_bonus: float = 8.0
    death_penalty: float = -8.0
    time_penalty: float = -0.002
    survival_bonus: float = 0.001
    survival_bonus_start_frac: float = 0.85


@dataclass(slots=True)
class RLConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    learning_rate: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 256
    steps_per_update: int = 2048
    imitation_coef: float = 0.2
    max_grad_norm: float = 0.5
    imitation_buffer_capacity: int = 12000
    imitation_batch_size: int = 256


@dataclass(slots=True)
class CurriculumConfig:
    enabled: bool = True
    stage_scale: float = 1.25
    advance_window: int = 20
    advance_survival_rate: float = 0.7
    min_stage_steps: int = 200


@dataclass(slots=True)
class RenderConfig:
    enabled: bool = True
    window_width: int = 1920
    window_height: int = 1080
    side_panel_width: int = 620
    start_fullscreen: bool = False
    window_resizable: bool = True
    fps: int = 60
    show_help_by_default: bool = True
    theme: str = "agar_reference_hybrid"
    overlay_mode_default: str = "minimal"
    grid_enabled_default: bool = True
    grid_spacing: int = 25
    grid_line_width: int = 2
    hud_refresh_hz: int = 15
    cache_grid_surface: bool = True
    show_agent_labels: bool = True
    show_score_chip: bool = True


@dataclass(slots=True)
class SimulationConfig:
    physics_hz: int = 90
    decision_hz: int = 15
    max_substeps_per_frame: int = 8
    action_mode: str = "continuous"
    camera_smoothness: float = 0.18
    zoom_smoothness: float = 0.12


@dataclass(slots=True)
class AsyncTrainingConfig:
    enabled: bool = True
    rollout_queue_size: int = 2
    min_rollout_transitions_per_job: int = 1024
    max_pending_weight_updates: int = 2


@dataclass(slots=True)
class SupervisorConfig:
    min_speed_multiplier: float = 0.25
    max_speed_multiplier: float = 16.0
    speed_step: float = 0.25
    checkpoint_path: str = "checkpoints/latest.pt"
    auto_update_when_ready: bool = True


@dataclass(slots=True)
class LoggingConfig:
    log_dir: str = "logs"
    train_metrics_csv: str = "logs/train_metrics.csv"
    print_every_updates: int = 1
    checkpoint_every_updates: int = 10


@dataclass(slots=True)
class AgarioConfig:
    seed: int = 7
    num_agents: int = 3
    max_steps: int = 1200
    nearest_pellets: int = 8
    nearest_opponents: int = 6
    map: MapConfig = field(default_factory=MapConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    async_training: AsyncTrainingConfig = field(
        default_factory=AsyncTrainingConfig)
    supervisor: SupervisorConfig = field(default_factory=SupervisorConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _default_raw_config() -> dict[str, Any]:
    return {
        "seed": 7,
        "num_agents": 3,
        "max_steps": 1200,
        "nearest_pellets": 8,
        "nearest_opponents": 6,
        "map": asdict(MapConfig()),
        "physics": asdict(PhysicsConfig()),
        "rewards": asdict(RewardConfig()),
        "rl": asdict(RLConfig()),
        "curriculum": asdict(CurriculumConfig()),
        "render": asdict(RenderConfig()),
        "simulation": asdict(SimulationConfig()),
        "async_training": asdict(AsyncTrainingConfig()),
        "supervisor": asdict(SupervisorConfig()),
        "logging": asdict(LoggingConfig()),
    }


def load_config(path: str | Path) -> AgarioConfig:
    """Load project configuration from YAML and fill missing defaults."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    raw = _merge_dicts(_default_raw_config(), loaded)
    render_raw = dict(raw["render"])
    render_raw.pop("backend", None)
    return AgarioConfig(
        seed=int(raw["seed"]),
        num_agents=int(raw["num_agents"]),
        max_steps=int(raw["max_steps"]),
        nearest_pellets=int(raw["nearest_pellets"]),
        nearest_opponents=int(raw["nearest_opponents"]),
        map=MapConfig(**raw["map"]),
        physics=PhysicsConfig(**raw["physics"]),
        rewards=RewardConfig(**raw["rewards"]),
        rl=RLConfig(**raw["rl"]),
        curriculum=CurriculumConfig(**raw["curriculum"]),
        render=RenderConfig(**render_raw),
        simulation=SimulationConfig(**raw["simulation"]),
        async_training=AsyncTrainingConfig(**raw["async_training"]),
        supervisor=SupervisorConfig(**raw["supervisor"]),
        logging=LoggingConfig(**raw["logging"]),
    )


__all__ = [
    "AgarioConfig",
    "AsyncTrainingConfig",
    "CurriculumConfig",
    "LoggingConfig",
    "MapConfig",
    "PhysicsConfig",
    "RLConfig",
    "RenderConfig",
    "RewardConfig",
    "SimulationConfig",
    "SupervisorConfig",
    "load_config",
]
