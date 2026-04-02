"""Compatibility tests for continuous and discrete action modes."""

from __future__ import annotations

import numpy as np

from agario_rl import AgarioConfig
from agario_rl.env.gym_env import AgarioMultiAgentEnv
from agario_rl.rl.ppo_shared import SharedPPOTrainer


def test_action_spaces_exposed_per_mode() -> None:
    cfg_cont = AgarioConfig()
    cfg_cont.simulation.action_mode = "continuous"
    env_cont = AgarioMultiAgentEnv(cfg_cont, enable_render=False)
    assert env_cont.action_space["type"] == "Box"
    assert tuple(env_cont.action_space["shape"]) == (3,)
    env_cont.close()

    cfg_disc = AgarioConfig()
    cfg_disc.simulation.action_mode = "discrete_9way"
    env_disc = AgarioMultiAgentEnv(cfg_disc, enable_render=False)
    assert env_disc.action_space["type"] == "MultiDiscrete"
    env_disc.close()


def test_trainer_predict_actions_works_for_both_modes() -> None:
    for mode, expected_size in (("continuous", 3), ("discrete_9way", 2)):
        config = AgarioConfig()
        config.simulation.action_mode = mode
        env = AgarioMultiAgentEnv(config=config, enable_render=False)
        trainer = SharedPPOTrainer(config=config, observation_dim=env.observation_space["shape"][0], device="cpu")
        obs = trainer.force_sync_with_env(env, seed=11)
        actions = trainer.predict_actions(obs, deterministic=False)
        for action in actions.values():
            assert isinstance(action, np.ndarray)
            assert action.shape == (expected_size,)
        env.close()
