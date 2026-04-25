"""Regression tests for richer Agar.io simulator mechanics."""

from __future__ import annotations

import numpy as np

from agario_rl import AgarioConfig, apply_scenario_preset
from agario_rl.env.entities import Cell, Virus
from agario_rl.env.world import AgarioWorld


def _zero_actions(world: AgarioWorld) -> dict[str, np.ndarray]:
    return {
        agent_id: np.array([0.0, 0.0, 0.0], dtype=np.float32)
        for agent_id in world.agent_ids
    }


def test_large_cell_consuming_virus_splits_into_multiple_cells() -> None:
    config = AgarioConfig()
    config.viruses.enabled = True
    config.viruses.initial_count = 0
    config.physics.max_cells_per_agent = 8
    world = AgarioWorld(config=config, seed=11)
    agent_id = world.agent_ids[0]
    cell = world.agents[agent_id][0]
    cell.mass = 180.0
    world.viruses = [
        Virus(
            virus_id=77,
            position=cell.position.copy(),
            mass=config.viruses.mass,
        )
    ]

    outcome = world.step(_zero_actions(world))

    assert len(world.agents[agent_id]) > 1
    assert outcome.infos[agent_id]["virus_splits"] == 1
    assert len(world.viruses) == 0


def test_ejected_mass_moves_and_can_feed_virus() -> None:
    config = AgarioConfig()
    config.physics.enable_eject_mechanic = True
    config.viruses.enabled = True
    config.viruses.initial_count = 0
    config.viruses.feed_to_split = 1
    world = AgarioWorld(config=config, seed=13)
    agent_id = world.agent_ids[0]
    cell = world.agents[agent_id][0]
    cell.mass = 60.0
    direction = np.array([1.0, 0.0], dtype=np.float32)
    world.viruses = [
        Virus(
            virus_id=3,
            position=cell.position + np.array([cell.radius(config.physics.radius_scale) + 3.0, 0.0], dtype=np.float32),
            mass=config.viruses.mass,
        )
    ]

    world.eject_mass(agent_id, direction)
    assert len(world.ejected_masses) == 1
    first_position = world.ejected_masses[0].position.copy()

    world.step(_zero_actions(world), dt=1.0 / config.simulation.physics_hz)

    assert len(world.ejected_masses) == 0
    assert len(world.viruses) >= 2
    assert not np.allclose(first_position, world.viruses[-1].position)


def test_mass_decay_reduces_cell_mass_over_time() -> None:
    config = AgarioConfig()
    config.mass_decay.enabled = True
    config.mass_decay.per_second = 0.10
    world = AgarioWorld(config=config, seed=17)
    agent_id = world.agent_ids[0]
    world.agents[agent_id][0].mass = 100.0

    world.step(_zero_actions(world), dt=1.0)

    assert world.agents[agent_id][0].mass < 100.0


def test_continuing_respawn_keeps_eliminated_agent_in_world() -> None:
    config = AgarioConfig()
    config.simulation.continuing_respawn = True
    config.viruses.enabled = False
    world = AgarioWorld(config=config, seed=19)
    hunter_id, victim_id = world.agent_ids[:2]
    pos = np.array([80.0, 80.0], dtype=np.float32)
    world.agents[hunter_id] = [
        Cell(
            cell_id=101,
            agent_id=hunter_id,
            position=pos.copy(),
            velocity=np.zeros(2, dtype=np.float32),
            mass=80.0,
        )
    ]
    world.agents[victim_id] = [
        Cell(
            cell_id=102,
            agent_id=victim_id,
            position=pos.copy(),
            velocity=np.zeros(2, dtype=np.float32),
            mass=20.0,
        )
    ]
    world.agents[world.agent_ids[2]] = []

    outcome = world.step(_zero_actions(world))

    assert world.agents[victim_id]
    assert outcome.infos[victim_id]["respawned"] is True
    assert outcome.dones[victim_id] is False
    assert outcome.dones["__all__"] is False


def test_extended_observation_features_are_additive() -> None:
    config = AgarioConfig()
    base_dim = AgarioWorld(config=config, seed=23).observation_dim
    config.observation_features.enabled = True
    config.observation_features.include_viruses = True
    config.observation_features.include_threats = True
    config.observation_features.include_eject_state = True
    upgraded = AgarioWorld(config=config, seed=23)

    obs = upgraded.get_observations()[upgraded.agent_ids[0]]

    assert upgraded.observation_dim > base_dim
    assert obs.shape == (upgraded.observation_dim,)


def test_scenario_curriculum_unlocks_viruses_after_early_stages() -> None:
    config = apply_scenario_preset(AgarioConfig(), "agario_curriculum")
    world = AgarioWorld(config=config, seed=31)

    assert world.scenario_name == "pellet_growth"
    assert world.target_virus_count == 0

    world.stage = 3

    assert world.scenario_name == "virus_control"
    assert world.target_virus_count > 0


def test_reward_terms_report_threat_escape_breakdown() -> None:
    config = AgarioConfig()
    config.reward_terms.threat_escape_scale = 1.0
    config.reward_terms.corner_penalty = 0.0
    world = AgarioWorld(config=config, seed=29)
    runner_id, threat_id = world.agent_ids[:2]
    world.agents[runner_id][0].position = np.array([100.0, 100.0], dtype=np.float32)
    world.agents[runner_id][0].mass = 20.0
    world.agents[threat_id][0].position = np.array([145.0, 100.0], dtype=np.float32)
    world.agents[threat_id][0].mass = 80.0
    world.snapshots[runner_id].nearest_threat_distance = 45.0

    actions = _zero_actions(world)
    actions[runner_id] = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    outcome = world.step(actions, dt=1.0 / config.simulation.physics_hz)

    assert outcome.infos[runner_id]["reward_breakdown"]["threat_escape"] > 0.0
    assert outcome.rewards[runner_id] > config.rewards.time_penalty


def test_unsafe_split_is_penalized_near_larger_threat() -> None:
    config = AgarioConfig()
    config.map.start_size = 500
    config.map.max_size = 500
    config.reward_terms.split_attempt_penalty = -0.05
    config.reward_terms.unsafe_split_penalty = -0.4
    world = AgarioWorld(config=config, seed=41)
    splitter_id, threat_id = world.agent_ids[:2]
    world.agents[splitter_id][0].position = np.array([220.0, 250.0], dtype=np.float32)
    world.agents[splitter_id][0].mass = 80.0
    world.agents[threat_id][0].position = np.array([290.0, 250.0], dtype=np.float32)
    world.agents[threat_id][0].mass = 100.0

    actions = _zero_actions(world)
    actions[splitter_id] = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    outcome = world.step(actions, dt=1.0 / config.simulation.physics_hz)
    breakdown = outcome.infos[splitter_id]["reward_breakdown"]

    assert breakdown["split_attempt"] == config.reward_terms.split_attempt_penalty
    assert breakdown["unsafe_split"] == config.reward_terms.unsafe_split_penalty
    assert outcome.infos[splitter_id]["unsafe_splits"] == 1


def test_useful_split_gets_target_pressure_credit() -> None:
    config = AgarioConfig()
    config.map.start_size = 500
    config.map.max_size = 500
    config.reward_terms.split_attempt_penalty = -0.05
    config.reward_terms.useful_split_bonus = 0.2
    world = AgarioWorld(config=config, seed=43)
    hunter_id, target_id = world.agent_ids[:2]
    world.agents[hunter_id][0].position = np.array([220.0, 250.0], dtype=np.float32)
    world.agents[hunter_id][0].mass = 90.0
    world.agents[target_id][0].position = np.array([270.0, 250.0], dtype=np.float32)
    world.agents[target_id][0].mass = 24.0

    actions = _zero_actions(world)
    actions[hunter_id] = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    outcome = world.step(actions, dt=1.0 / config.simulation.physics_hz)
    breakdown = outcome.infos[hunter_id]["reward_breakdown"]

    assert breakdown["split_attempt"] == config.reward_terms.split_attempt_penalty
    assert breakdown["useful_split"] == config.reward_terms.useful_split_bonus
    assert outcome.infos[hunter_id]["useful_splits"] == 1
