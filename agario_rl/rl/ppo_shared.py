"""Shared-policy PPO trainer with peer imitation loss."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical, Normal

from agario_rl import AgarioConfig
from agario_rl.rl.buffer import RolloutBatch, RolloutSample, Transition, compute_gae
from agario_rl.rl.networks import ActorCriticNetwork
from agario_rl.rl.peer_imitation import PeerImitationBuffer


class SharedPPOTrainer:
    """Centralized PPO update loop with shared parameters for all agents."""

    def __init__(self, config: AgarioConfig, observation_dim: int, device: str | None = None) -> None:
        self.config = config
        self.action_mode = config.simulation.action_mode
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.policy = ActorCriticNetwork(observation_dim=observation_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.rl.learning_rate)
        self.imitation_buffer = PeerImitationBuffer(
            capacity=config.rl.imitation_buffer_capacity,
            seed=config.seed,
        )

        self.agent_ids: list[str] = []
        self.current_obs: dict[str, np.ndarray] | None = None
        self.trajectories: dict[str, list[Transition]] = {}
        self.pending_samples: list[RolloutSample] = []
        self.episode_returns: dict[str, float] = {}
        self.episode_demos: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}

        self.transitions_since_update = 0
        self.update_count = 0
        self.policy_sync_age_steps = 0
        self.last_actions: dict[str, np.ndarray] | None = None
        self.last_metrics: dict[str, float] = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "imitation_loss": 0.0,
            "total_loss": 0.0,
            "batch_size": 0.0,
            "update_count": 0.0,
        }

    def set_tracked_agent_ids(self, agent_ids: list[str]) -> None:
        """Restrict experience collection to a subset of environment agents."""
        self.agent_ids = list(agent_ids)
        self.trajectories = {agent_id: [] for agent_id in self.agent_ids}
        self.episode_returns = {agent_id: 0.0 for agent_id in self.agent_ids}
        self.episode_demos = {agent_id: [] for agent_id in self.agent_ids}

    def _ensure_env_state(self, env: Any) -> None:
        if not self.agent_ids:
            self.set_tracked_agent_ids(list(env.agent_ids))
        if self.current_obs is None:
            self.current_obs = env.reset(seed=self.config.seed)

    def force_sync_with_env(self, env: Any, seed: int | None = None) -> dict[str, np.ndarray]:
        """Hard reset trainer episode state to match the environment reset."""
        if not self.agent_ids:
            self.set_tracked_agent_ids(list(env.agent_ids))
        else:
            self.trajectories = {agent_id: [] for agent_id in self.agent_ids}
            self.episode_returns = {agent_id: 0.0 for agent_id in self.agent_ids}
            self.episode_demos = {agent_id: [] for agent_id in self.agent_ids}
        self.current_obs = env.reset(seed=seed)
        self.last_actions = None
        return self.current_obs

    def _policy_outputs(self, obs_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.policy(obs_tensor)

    def _continuous_action_sample(
        self,
        steer_mean: torch.Tensor,
        steer_log_std: torch.Tensor,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        steer_std = torch.exp(steer_log_std).clamp(min=1e-4, max=4.0)
        steer_dist = Normal(steer_mean, steer_std)
        pre_tanh = steer_mean if deterministic else steer_dist.rsample()
        steer = torch.tanh(pre_tanh)
        logprob = steer_dist.log_prob(pre_tanh) - torch.log(1 - steer * steer + 1e-6)
        steer_logprob = logprob.sum(dim=-1)
        steer_entropy = steer_dist.entropy().sum(dim=-1)
        return steer, steer_logprob, steer_entropy

    def _continuous_logprob_entropy_from_actions(
        self,
        steer_mean: torch.Tensor,
        steer_log_std: torch.Tensor,
        steer_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        steer_actions = torch.clamp(steer_actions, -0.999, 0.999)
        pre_tanh = 0.5 * torch.log((1 + steer_actions) / (1 - steer_actions))
        steer_std = torch.exp(steer_log_std).clamp(min=1e-4, max=4.0)
        steer_dist = Normal(steer_mean, steer_std)
        logprob = steer_dist.log_prob(pre_tanh) - torch.log(1 - steer_actions * steer_actions + 1e-6)
        steer_logprob = logprob.sum(dim=-1)
        steer_entropy = steer_dist.entropy().sum(dim=-1)
        return steer_logprob, steer_entropy

    def _policy_step(
        self,
        obs_dict: dict[str, np.ndarray],
        deterministic: bool = False,
        agent_ids: list[str] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float | np.ndarray]]]:
        ordered_agent_ids = list(agent_ids or self.agent_ids or sorted(obs_dict.keys()))
        obs_batch = torch.tensor(
            np.stack([obs_dict[agent_id] for agent_id in ordered_agent_ids], axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            outputs = self._policy_outputs(obs_batch)
            ability_dist = Categorical(logits=outputs["ability_logits"])
            ability_action = (
                torch.argmax(outputs["ability_logits"], dim=-1)
                if deterministic
                else ability_dist.sample()
            )
            ability_logprob = ability_dist.log_prob(ability_action)

            if self.action_mode == "continuous":
                steer, steer_logprob, _ = self._continuous_action_sample(
                    outputs["steer_mean"],
                    outputs["steer_log_std"],
                    deterministic=deterministic,
                )
                total_logprob = steer_logprob + ability_logprob
            else:
                direction_dist = Categorical(logits=outputs["direction_logits"])
                direction_action = (
                    torch.argmax(outputs["direction_logits"], dim=-1)
                    if deterministic
                    else direction_dist.sample()
                )
                direction_logprob = direction_dist.log_prob(direction_action)
                total_logprob = direction_logprob + ability_logprob

        actions: dict[str, np.ndarray] = {}
        cache: dict[str, dict[str, float | np.ndarray]] = {}
        for idx, agent_id in enumerate(ordered_agent_ids):
            if self.action_mode == "continuous":
                action_vec = np.array(
                    [
                        float(steer[idx, 0].item()),
                        float(steer[idx, 1].item()),
                        float(ability_action[idx].item()),
                    ],
                    dtype=np.float32,
                )
            else:
                action_vec = np.array(
                    [
                        float(direction_action[idx].item()),
                        float(ability_action[idx].item()),
                    ],
                    dtype=np.float32,
                )
            actions[agent_id] = action_vec
            cache[agent_id] = {
                "logprob": float(total_logprob[idx].item()),
                "value": float(outputs["value"][idx].item()),
                "action": action_vec,
            }
        return actions, cache

    def predict_actions(
        self,
        obs_dict: dict[str, np.ndarray],
        deterministic: bool = False,
        agent_ids: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """Get policy actions without writing to replay buffers."""
        ordered_agent_ids = list(agent_ids or self.agent_ids or sorted(obs_dict.keys()))
        actions, _ = self._policy_step(
            obs_dict,
            deterministic=deterministic,
            agent_ids=ordered_agent_ids,
        )
        return actions

    def _record_decision_transition(
        self,
        decision_obs: dict[str, np.ndarray],
        cache: dict[str, dict[str, float | np.ndarray]],
        accumulated_rewards: dict[str, float],
        dones: dict[str, bool],
        track_experience: bool,
    ) -> None:
        if not track_experience:
            return
        for agent_id in self.agent_ids:
            transition = Transition(
                obs=decision_obs[agent_id].copy(),
                action=np.asarray(cache[agent_id]["action"], dtype=np.float32).copy(),
                logprob=float(cache[agent_id]["logprob"]),
                value=float(cache[agent_id]["value"]),
                reward=float(accumulated_rewards.get(agent_id, 0.0)),
                done=bool(dones.get(agent_id, False) or dones.get("__all__", False)),
            )
            self.trajectories[agent_id].append(transition)
            self.episode_returns[agent_id] += float(accumulated_rewards.get(agent_id, 0.0))
            self.episode_demos[agent_id].append(
                (
                    decision_obs[agent_id].copy(),
                    np.asarray(cache[agent_id]["action"], dtype=np.float32).copy(),
                )
            )
            self.transitions_since_update += 1

    def step_decision(
        self,
        env: Any,
        substeps: int = 1,
        dt: float | None = None,
        track_experience: bool = True,
        deterministic: bool = False,
        action_overrides: dict[str, np.ndarray] | None = None,
        policy_agent_ids: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Run one policy decision followed by one or more physics substeps."""
        self._ensure_env_state(env)
        assert self.current_obs is not None

        tracked_agent_ids = list(self.agent_ids)
        acting_policy_ids = list(policy_agent_ids or tracked_agent_ids)
        actions, cache = self._policy_step(
            self.current_obs,
            deterministic=deterministic,
            agent_ids=acting_policy_ids,
        )
        if action_overrides:
            for agent_id, action in action_overrides.items():
                actions[agent_id] = np.asarray(action, dtype=np.float32).copy()
        self.last_actions = {agent_id: action.copy() for agent_id, action in actions.items()}
        decision_obs = {
            agent_id: self.current_obs[agent_id].copy()
            for agent_id in tracked_agent_ids
        }
        accumulated_rewards = {agent_id: 0.0 for agent_id in tracked_agent_ids}
        infos: dict[str, dict[str, Any]] = {}
        dones: dict[str, bool] = {"__all__": False}
        next_obs: dict[str, np.ndarray] | None = self.current_obs
        physics_dt = dt if dt is not None else (1.0 / max(1, self.config.simulation.physics_hz))

        total_substeps = max(1, substeps)
        for step_idx in range(total_substeps):
            compute_observations = step_idx == (total_substeps - 1)
            step_obs, rewards, dones, infos = env.step(
                actions,
                dt=physics_dt,
                compute_observations=compute_observations,
            )
            if step_obs is not None:
                next_obs = step_obs
            for agent_id in tracked_agent_ids:
                accumulated_rewards[agent_id] += float(rewards.get(agent_id, 0.0))
            if dones.get("__all__", False):
                break

        self._record_decision_transition(
            decision_obs=decision_obs,
            cache=cache,
            accumulated_rewards=accumulated_rewards,
            dones=dones,
            track_experience=track_experience,
        )
        episode_done = bool(dones.get("__all__", False))
        if episode_done:
            if track_experience:
                self._flush_ended_episode()
            self.current_obs = env.reset()
            self.last_actions = None
        else:
            assert next_obs is not None
            self.current_obs = next_obs
        self.policy_sync_age_steps += 1
        return infos

    def step_once(self, env: Any, dt: float | None = None) -> dict[str, dict[str, Any]]:
        return self.step_decision(env=env, substeps=1, dt=dt, track_experience=True)

    def collect_rollout(self, env: Any, target_transitions: int | None = None) -> int:
        """Collect experience until the target number of transitions is reached."""
        self._ensure_env_state(env)
        target = target_transitions or self.config.rl.steps_per_update
        collected = 0
        while collected < target:
            self.step_once(env)
            collected += len(self.agent_ids)
        return collected

    def step_physics_with_last_action(self, env: Any, dt: float | None = None) -> dict[str, dict[str, Any]]:
        """Advance one physics tick without creating RL transitions."""
        self._ensure_env_state(env)
        assert self.current_obs is not None
        if self.last_actions is None:
            self.last_actions = self.predict_actions(self.current_obs, deterministic=False)
        physics_dt = dt if dt is not None else (1.0 / max(1, self.config.simulation.physics_hz))
        next_obs, _, dones, infos = env.step(
            self.last_actions,
            dt=physics_dt,
            compute_observations=True,
        )
        if dones.get("__all__", False):
            self.current_obs = env.reset()
            self.last_actions = None
        else:
            assert next_obs is not None
            self.current_obs = next_obs
        return infos

    def ready_to_update(self) -> bool:
        return self.transitions_since_update >= self.config.rl.steps_per_update

    def _bootstrap_values(self) -> dict[str, float]:
        assert self.current_obs is not None
        obs_batch = torch.tensor(
            np.stack([self.current_obs[agent_id] for agent_id in self.agent_ids], axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            values = self._policy_outputs(obs_batch)["value"]
        return {agent_id: float(values[idx].item()) for idx, agent_id in enumerate(self.agent_ids)}

    def _flush_trajectory(self, agent_id: str, bootstrap_value: float) -> None:
        trajectory = self.trajectories[agent_id]
        if not trajectory:
            return
        samples = compute_gae(
            transitions=trajectory,
            bootstrap_value=bootstrap_value,
            gamma=self.config.rl.gamma,
            gae_lambda=self.config.rl.gae_lambda,
        )
        self.pending_samples.extend(samples)
        self.trajectories[agent_id] = []

    def _flush_ended_episode(self) -> None:
        for agent_id in self.agent_ids:
            self._flush_trajectory(agent_id, bootstrap_value=0.0)
        self.imitation_buffer.add_episode(self.episode_demos, self.episode_returns)
        self.episode_returns = {agent_id: 0.0 for agent_id in self.agent_ids}
        self.episode_demos = {agent_id: [] for agent_id in self.agent_ids}

    def _flush_active_trajectories(self) -> None:
        if self.current_obs is None:
            return
        bootstrap = self._bootstrap_values()
        for agent_id in self.agent_ids:
            self._flush_trajectory(agent_id, bootstrap_value=bootstrap[agent_id])

    def compute_imitation_loss(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute imitation objective on provided observations/actions."""
        outputs = self._policy_outputs(obs)
        ability_target = torch.round(actions[:, -1]).long().clamp(min=0, max=1)
        ability_loss = F.cross_entropy(outputs["ability_logits"], ability_target)
        if self.action_mode == "continuous":
            steer_target = actions[:, :2]
            steer_loss = F.mse_loss(torch.tanh(outputs["steer_mean"]), steer_target)
            return steer_loss + ability_loss
        direction_target = torch.round(actions[:, 0]).long().clamp(min=0, max=8)
        direction_loss = F.cross_entropy(outputs["direction_logits"], direction_target)
        return direction_loss + ability_loss

    def _rollout_to_numpy_payload(self, samples: list[RolloutSample]) -> dict[str, np.ndarray]:
        return {
            "obs": np.stack([sample.obs for sample in samples], axis=0).astype(np.float32),
            "actions": np.stack([sample.action for sample in samples], axis=0).astype(np.float32),
            "old_logprob": np.array([sample.logprob for sample in samples], dtype=np.float32),
            "old_value": np.array([sample.value for sample in samples], dtype=np.float32),
            "advantages": np.array([sample.advantage for sample in samples], dtype=np.float32),
            "returns": np.array([sample.ret for sample in samples], dtype=np.float32),
        }

    def build_rollout_batch_from_transitions(self, samples: list[RolloutSample]) -> RolloutBatch:
        return RolloutBatch(
            obs=torch.tensor(np.stack([sample.obs for sample in samples], axis=0), dtype=torch.float32, device=self.device),
            actions=torch.tensor(np.stack([sample.action for sample in samples], axis=0), dtype=torch.float32, device=self.device),
            old_logprob=torch.tensor([sample.logprob for sample in samples], dtype=torch.float32, device=self.device),
            old_value=torch.tensor([sample.value for sample in samples], dtype=torch.float32, device=self.device),
            advantages=torch.tensor([sample.advantage for sample in samples], dtype=torch.float32, device=self.device),
            returns=torch.tensor([sample.ret for sample in samples], dtype=torch.float32, device=self.device),
        )

    def prepare_update_job_payload(self) -> dict[str, Any] | None:
        """Build an async update payload and clear the local pending samples."""
        self._flush_active_trajectories()
        if not self.pending_samples:
            return None
        payload = {
            "rollout": self._rollout_to_numpy_payload(self.pending_samples),
            "imitation": (
                self.imitation_buffer.sample(self.config.rl.imitation_batch_size)
                if len(self.imitation_buffer) >= self.config.rl.imitation_batch_size
                else None
            ),
        }
        self.pending_samples = []
        self.transitions_since_update = 0
        return payload

    def _compute_policy_terms(
        self,
        outputs: dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ability_action = torch.round(actions[:, -1]).long().clamp(min=0, max=1)
        ability_dist = Categorical(logits=outputs["ability_logits"])
        ability_logprob = ability_dist.log_prob(ability_action)
        ability_entropy = ability_dist.entropy()

        if self.action_mode == "continuous":
            steer_actions = actions[:, :2]
            steer_logprob, steer_entropy = self._continuous_logprob_entropy_from_actions(
                outputs["steer_mean"],
                outputs["steer_log_std"],
                steer_actions,
            )
            return steer_logprob + ability_logprob, steer_entropy + ability_entropy

        direction_actions = torch.round(actions[:, 0]).long().clamp(min=0, max=8)
        direction_dist = Categorical(logits=outputs["direction_logits"])
        direction_logprob = direction_dist.log_prob(direction_actions)
        direction_entropy = direction_dist.entropy()
        return direction_logprob + ability_logprob, direction_entropy + ability_entropy

    def update_on_batch(
        self,
        rollout_payload: dict[str, np.ndarray],
        imitation_payload: dict[str, np.ndarray] | None = None,
    ) -> dict[str, float]:
        """Run PPO optimization on a provided rollout payload."""
        batch = RolloutBatch(
            obs=torch.tensor(rollout_payload["obs"], dtype=torch.float32, device=self.device),
            actions=torch.tensor(rollout_payload["actions"], dtype=torch.float32, device=self.device),
            old_logprob=torch.tensor(rollout_payload["old_logprob"], dtype=torch.float32, device=self.device),
            old_value=torch.tensor(rollout_payload["old_value"], dtype=torch.float32, device=self.device),
            advantages=torch.tensor(rollout_payload["advantages"], dtype=torch.float32, device=self.device),
            returns=torch.tensor(rollout_payload["returns"], dtype=torch.float32, device=self.device),
        )
        advantages = batch.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        num_samples = batch.obs.shape[0]
        minibatch_size = min(self.config.rl.minibatch_size, num_samples)
        stat_policy_loss = 0.0
        stat_value_loss = 0.0
        stat_entropy = 0.0
        stat_imitation_loss = 0.0
        stat_total_loss = 0.0
        stat_updates = 0

        for _ in range(self.config.rl.ppo_epochs):
            indices = torch.randperm(num_samples, device=self.device)
            for start in range(0, num_samples, minibatch_size):
                mb_idx = indices[start : start + minibatch_size]
                mb_obs = batch.obs[mb_idx]
                mb_actions = batch.actions[mb_idx]
                mb_old_logprob = batch.old_logprob[mb_idx]
                mb_returns = batch.returns[mb_idx]
                mb_adv = advantages[mb_idx]

                outputs = self._policy_outputs(mb_obs)
                new_logprob, entropy = self._compute_policy_terms(outputs, mb_actions)
                ratio = torch.exp(new_logprob - mb_old_logprob)
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - self.config.rl.clip_coef,
                    1.0 + self.config.rl.clip_coef,
                )
                policy_loss = -torch.min(ratio * mb_adv, clipped_ratio * mb_adv).mean()
                value_loss = F.mse_loss(outputs["value"], mb_returns)
                entropy_mean = entropy.mean()

                imitation_loss = torch.tensor(0.0, device=self.device)
                if imitation_payload is not None:
                    demo_obs = torch.tensor(imitation_payload["obs"], dtype=torch.float32, device=self.device)
                    demo_actions = torch.tensor(imitation_payload["actions"], dtype=torch.float32, device=self.device)
                    imitation_loss = self.compute_imitation_loss(demo_obs, demo_actions)

                total_loss = (
                    policy_loss
                    + self.config.rl.value_coef * value_loss
                    - self.config.rl.entropy_coef * entropy_mean
                    + self.config.rl.imitation_coef * imitation_loss
                )

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.rl.max_grad_norm)
                self.optimizer.step()

                stat_policy_loss += float(policy_loss.item())
                stat_value_loss += float(value_loss.item())
                stat_entropy += float(entropy_mean.item())
                stat_imitation_loss += float(imitation_loss.item())
                stat_total_loss += float(total_loss.item())
                stat_updates += 1

        denom = max(1, stat_updates)
        self.update_count += 1
        self.last_metrics = {
            "policy_loss": stat_policy_loss / denom,
            "value_loss": stat_value_loss / denom,
            "entropy": stat_entropy / denom,
            "imitation_loss": stat_imitation_loss / denom,
            "total_loss": stat_total_loss / denom,
            "batch_size": float(num_samples),
            "update_count": float(self.update_count),
        }
        self.policy_sync_age_steps = 0
        return dict(self.last_metrics)

    def update(self) -> dict[str, float]:
        """Run PPO optimization using accumulated rollouts."""
        self._flush_active_trajectories()
        if not self.pending_samples:
            return dict(self.last_metrics)

        rollout_payload = self._rollout_to_numpy_payload(self.pending_samples)
        imitation_payload = (
            self.imitation_buffer.sample(self.config.rl.imitation_batch_size)
            if len(self.imitation_buffer) >= self.config.rl.imitation_batch_size
            else None
        )
        metrics = self.update_on_batch(rollout_payload, imitation_payload=imitation_payload)
        self.pending_samples = []
        self.transitions_since_update = 0
        return metrics

    def export_training_state(self) -> dict[str, Any]:
        return {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_count": self.update_count,
            "last_metrics": dict(self.last_metrics),
        }

    def import_training_state(self, payload: dict[str, Any]) -> None:
        self.policy.load_state_dict(payload["policy"])
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        self.update_count = int(payload.get("update_count", self.update_count))
        self.last_metrics = dict(payload.get("last_metrics", self.last_metrics))
        self.policy_sync_age_steps = 0

    def save(self, path: str | Path) -> None:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.export_training_state(), file_path)

    def load(self, path: str | Path) -> bool:
        file_path = Path(path)
        if not file_path.exists():
            return False
        payload = torch.load(file_path, map_location=self.device)
        self.import_training_state(payload)
        return True
