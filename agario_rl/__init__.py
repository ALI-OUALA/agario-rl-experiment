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
    eject_speed: float = 8.0


@dataclass(slots=True)
class VirusConfig:
    enabled: bool = False
    initial_count: int = 0
    mass: float = 100.0
    min_split_mass: float = 90.0
    feed_to_split: int = 7
    max_count: int = 24
    spawn_margin: float = 30.0
    split_spawn_distance: float = 42.0
    split_pieces: int = 4
    consumption_efficiency: float = 0.0


@dataclass(slots=True)
class MassDecayConfig:
    enabled: bool = False
    per_second: float = 0.0
    min_mass: float = 10.0


@dataclass(slots=True)
class ObservationFeaturesConfig:
    enabled: bool = False
    include_threats: bool = False
    include_viruses: bool = False
    include_eject_state: bool = False


@dataclass(slots=True)
class RewardTermsConfig:
    threat_escape_scale: float = 0.0
    target_pressure_scale: float = 0.0
    corner_penalty: float = 0.0
    survival_quality_scale: float = 0.0
    virus_split_bonus: float = 0.0
    respawn_penalty: float = 0.0
    split_attempt_penalty: float = 0.0
    unsafe_split_penalty: float = 0.0
    useful_split_bonus: float = 0.0


@dataclass(slots=True)
class ScenarioCurriculumConfig:
    enabled: bool = False
    preset: str = "classic"
    stage_names: list[str] = field(
        default_factory=lambda: [
            "pellet_growth",
            "evasion",
            "hunting",
            "virus_control",
            "mixed_arena",
            "full_arena",
        ]
    )


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
    split_logit_bias: float = -0.75
    unready_split_logit_penalty: float = 4.0


@dataclass(slots=True)
class CurriculumConfig:
    enabled: bool = True
    stage_scale: float = 1.25
    advance_window: int = 20
    advance_survival_rate: float = 0.7
    min_stage_steps: int = 200


@dataclass(slots=True)
class SimulationConfig:
    physics_hz: int = 90
    decision_hz: int = 15
    max_substeps_per_frame: int = 8
    action_mode: str = "continuous"
    camera_smoothness: float = 0.18
    zoom_smoothness: float = 0.12
    continuing_respawn: bool = False
    respawn_mass: float = 25.0


@dataclass(slots=True)
class AsyncTrainingConfig:
    enabled: bool = True
    rollout_queue_size: int = 2
    min_rollout_transitions_per_job: int = 1024
    max_pending_weight_updates: int = 2


@dataclass(slots=True)
class CheckpointConfig:
    latest_path: str = "checkpoints/latest.pt"


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
    viruses: VirusConfig = field(default_factory=VirusConfig)
    mass_decay: MassDecayConfig = field(default_factory=MassDecayConfig)
    observation_features: ObservationFeaturesConfig = field(
        default_factory=ObservationFeaturesConfig
    )
    reward_terms: RewardTermsConfig = field(default_factory=RewardTermsConfig)
    scenario_curriculum: ScenarioCurriculumConfig = field(
        default_factory=ScenarioCurriculumConfig
    )
    rewards: RewardConfig = field(default_factory=RewardConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    async_training: AsyncTrainingConfig = field(
        default_factory=AsyncTrainingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
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
        "viruses": asdict(VirusConfig()),
        "mass_decay": asdict(MassDecayConfig()),
        "observation_features": asdict(ObservationFeaturesConfig()),
        "reward_terms": asdict(RewardTermsConfig()),
        "scenario_curriculum": asdict(ScenarioCurriculumConfig()),
        "rewards": asdict(RewardConfig()),
        "rl": asdict(RLConfig()),
        "curriculum": asdict(CurriculumConfig()),
        "simulation": asdict(SimulationConfig()),
        "async_training": asdict(AsyncTrainingConfig()),
        "checkpoint": asdict(CheckpointConfig()),
        "logging": asdict(LoggingConfig()),
    }


def load_config(path: str | Path) -> AgarioConfig:
    """Load project configuration from YAML and fill missing defaults."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    raw = _merge_dicts(_default_raw_config(), loaded)
    return AgarioConfig(
        seed=int(raw["seed"]),
        num_agents=int(raw["num_agents"]),
        max_steps=int(raw["max_steps"]),
        nearest_pellets=int(raw["nearest_pellets"]),
        nearest_opponents=int(raw["nearest_opponents"]),
        map=MapConfig(**raw["map"]),
        physics=PhysicsConfig(**raw["physics"]),
        viruses=VirusConfig(**raw["viruses"]),
        mass_decay=MassDecayConfig(**raw["mass_decay"]),
        observation_features=ObservationFeaturesConfig(**raw["observation_features"]),
        reward_terms=RewardTermsConfig(**raw["reward_terms"]),
        scenario_curriculum=ScenarioCurriculumConfig(**raw["scenario_curriculum"]),
        rewards=RewardConfig(**raw["rewards"]),
        rl=RLConfig(**raw["rl"]),
        curriculum=CurriculumConfig(**raw["curriculum"]),
        simulation=SimulationConfig(**raw["simulation"]),
        async_training=AsyncTrainingConfig(**raw["async_training"]),
        checkpoint=CheckpointConfig(**raw["checkpoint"]),
        logging=LoggingConfig(**raw["logging"]),
    )


def apply_scenario_preset(config: AgarioConfig, preset: str) -> AgarioConfig:
    """Apply a named scenario preset while preserving the public env API."""
    preset_name = str(preset or "classic")
    config.scenario_curriculum.preset = preset_name
    if preset_name == "classic":
        return config
    if preset_name not in {"agario_curriculum", "full_arena"}:
        raise ValueError(f"Unknown scenario preset: {preset_name}")

    config.viruses.enabled = True
    config.viruses.initial_count = max(config.viruses.initial_count, 6)
    config.mass_decay.enabled = True
    config.mass_decay.per_second = max(config.mass_decay.per_second, 0.002)
    config.observation_features.enabled = True
    config.observation_features.include_threats = True
    config.observation_features.include_viruses = True
    config.observation_features.include_eject_state = True
    config.reward_terms.threat_escape_scale = max(config.reward_terms.threat_escape_scale, 0.08)
    config.reward_terms.target_pressure_scale = max(config.reward_terms.target_pressure_scale, 0.05)
    config.reward_terms.corner_penalty = min(config.reward_terms.corner_penalty, -0.015)
    config.reward_terms.survival_quality_scale = max(config.reward_terms.survival_quality_scale, 0.003)
    config.reward_terms.virus_split_bonus = max(config.reward_terms.virus_split_bonus, 0.6)
    config.reward_terms.respawn_penalty = min(config.reward_terms.respawn_penalty, -1.0)
    config.reward_terms.split_attempt_penalty = min(config.reward_terms.split_attempt_penalty, -0.015)
    config.reward_terms.unsafe_split_penalty = min(config.reward_terms.unsafe_split_penalty, -0.35)
    config.reward_terms.useful_split_bonus = max(config.reward_terms.useful_split_bonus, 0.12)
    config.scenario_curriculum.enabled = preset_name == "agario_curriculum"
    if preset_name == "full_arena":
        config.num_agents = max(config.num_agents, 6)
        config.max_steps = max(config.max_steps, 3600)
        config.nearest_pellets = max(config.nearest_pellets, 12)
        config.nearest_opponents = max(config.nearest_opponents, 5)
        config.map.start_size = max(config.map.start_size, 2000)
        config.map.max_size = max(config.map.max_size, 2000)
        config.map.pellets_per_10k_area = 2
        config.map.pellet_respawn_per_step = 5
        config.viruses.initial_count = max(config.viruses.initial_count, 18)
        config.viruses.max_count = max(config.viruses.max_count, 36)
        config.physics.enable_eject_mechanic = True
        config.physics.max_cells_per_agent = max(config.physics.max_cells_per_agent, 8)
        config.simulation.continuing_respawn = True
    return config


__all__ = [
    "AgarioConfig",
    "apply_scenario_preset",
    "AsyncTrainingConfig",
    "CheckpointConfig",
    "CurriculumConfig",
    "LoggingConfig",
    "MapConfig",
    "MassDecayConfig",
    "ObservationFeaturesConfig",
    "PhysicsConfig",
    "RLConfig",
    "RewardConfig",
    "RewardTermsConfig",
    "ScenarioCurriculumConfig",
    "SimulationConfig",
    "VirusConfig",
    "load_config",
]
