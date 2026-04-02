"""World rule tests: pellet eating, agent eating, split and merge behavior."""

from __future__ import annotations

import numpy as np

from agario_rl import AgarioConfig
from agario_rl.env.entities import Cell, Pellet
from agario_rl.env.world import AgarioWorld


def _zero_actions(world: AgarioWorld) -> dict[str, np.ndarray]:
    if world.config.simulation.action_mode == "continuous":
        return {agent_id: np.array([0.0, 0.0, 0.0], dtype=np.float32) for agent_id in world.agent_ids}
    return {agent_id: np.array([0.0, 0.0], dtype=np.float32) for agent_id in world.agent_ids}


def test_pellet_eating_increases_mass() -> None:
    config = AgarioConfig()
    world = AgarioWorld(config=config, seed=123)
    agent_id = world.agent_ids[0]
    cell = world.agents[agent_id][0]
    before_mass = cell.mass

    world.pellets = [
        Pellet(
            pellet_id=999,
            position=cell.position.copy(),
            mass=2.0,
        )
    ]

    world.step(_zero_actions(world))
    after_mass = world.agents[agent_id][0].mass
    assert after_mass > before_mass


def test_larger_cell_eats_smaller_cell_with_ratio_threshold() -> None:
    config = AgarioConfig()
    world = AgarioWorld(config=config, seed=321)
    world.pellets = []

    pos = np.array([80.0, 80.0], dtype=np.float32)
    world.agents[world.agent_ids[0]] = [
        Cell(
            cell_id=1,
            agent_id=world.agent_ids[0],
            position=pos.copy(),
            velocity=np.zeros(2, dtype=np.float32),
            mass=36.0,
        )
    ]
    world.agents[world.agent_ids[1]] = [
        Cell(
            cell_id=2,
            agent_id=world.agent_ids[1],
            position=pos.copy(),
            velocity=np.zeros(2, dtype=np.float32),
            mass=20.0,
        )
    ]
    world.agents[world.agent_ids[2]] = []

    world.step(_zero_actions(world))

    assert len(world.agents[world.agent_ids[0]]) >= 1
    assert len(world.agents[world.agent_ids[1]]) == 0


def test_split_then_merge_after_cooldown() -> None:
    config = AgarioConfig()
    config.simulation.action_mode = "discrete_9way"
    config.physics.merge_cooldown_steps = 5
    world = AgarioWorld(config=config, seed=99)
    agent_id = world.agent_ids[0]
    cell = world.agents[agent_id][0]
    cell.mass = 40.0

    split_action = _zero_actions(world)
    split_action[agent_id] = np.array([3.0, 1.0], dtype=np.float32)
    world.step(split_action)
    assert len(world.agents[agent_id]) == 2

    for _ in range(config.physics.merge_cooldown_steps):
        world.step(_zero_actions(world))
    for split_cell in world.agents[agent_id]:
        split_cell.position = np.array([40.0, 40.0], dtype=np.float32)
        split_cell.velocity = np.zeros(2, dtype=np.float32)
        split_cell.merge_cooldown = 0

    world.step(_zero_actions(world))
    assert len(world.agents[agent_id]) == 1


def test_split_continuous_mode_after_cooldown() -> None:
    config = AgarioConfig()
    config.simulation.action_mode = "continuous"
    config.physics.merge_cooldown_steps = 5
    world = AgarioWorld(config=config, seed=101)
    agent_id = world.agent_ids[0]
    world.agents[agent_id][0].mass = 45.0

    split_action = _zero_actions(world)
    split_action[agent_id] = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    world.step(split_action, dt=1.0 / config.simulation.physics_hz)
    assert len(world.agents[agent_id]) == 2
