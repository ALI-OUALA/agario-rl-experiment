"""Deterministic Agar.io-like world simulation for 3 RL agents."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from agario_rl import AgarioConfig
from agario_rl.env.entities import AgentSnapshot, Cell, Pellet
from agario_rl.utils.seeding import make_rng


DIRECTION_VECTORS = np.array(
    [
        [0.0, 0.0],    # stay
        [0.0, -1.0],   # up
        [0.707, -0.707],
        [1.0, 0.0],    # right
        [0.707, 0.707],
        [0.0, 1.0],    # down
        [-0.707, 0.707],
        [-1.0, 0.0],   # left
        [-0.707, -0.707],
    ],
    dtype=np.float32,
)


@dataclass(slots=True)
class StepOutcome:
    observations: dict[str, np.ndarray] | None
    rewards: dict[str, float]
    dones: dict[str, bool]
    infos: dict[str, dict[str, Any]]


class AgarioWorld:
    """Physics and reward engine backing the multi-agent environment."""

    def __init__(self, config: AgarioConfig, seed: int | None = None) -> None:
        self.config = config
        self.rng = make_rng(seed if seed is not None else config.seed)
        self.agent_ids = [f"agent_{idx}" for idx in range(config.num_agents)]
        self.agent_index = {agent_id: idx for idx, agent_id in enumerate(self.agent_ids)}
        self.stage = 0
        self.map_size = float(config.map.start_size)
        self.auto_curriculum = bool(config.curriculum.enabled)
        self.curriculum_scores: deque[float] = deque(maxlen=config.curriculum.advance_window)

        self.step_count = 0
        self.next_cell_id = 0
        self.next_pellet_id = 0
        self.agents: dict[str, list[Cell]] = {}
        self.pellets: list[Pellet] = []
        self.prev_cell_positions: dict[int, np.ndarray] = {}
        self.snapshots: dict[str, AgentSnapshot] = {
            agent_id: AgentSnapshot() for agent_id in self.agent_ids
        }
        self.last_winner: str | None = None
        self.observation_dim = self._compute_observation_dim()
        self.reset(seed=config.seed if seed is None else seed)

    def _compute_observation_dim(self) -> int:
        self_features = 8
        pellet_features = self.config.nearest_pellets * 3
        opp_features = self.config.nearest_opponents * 4
        global_features = 4
        return self_features + pellet_features + opp_features + global_features

    @property
    def alive_agents(self) -> list[str]:
        return [agent_id for agent_id, cells in self.agents.items() if cells]

    @property
    def target_pellet_count(self) -> int:
        area = self.map_size * self.map_size
        base = (area / 10_000.0) * float(self.config.map.pellets_per_10k_area)
        return max(20, int(base))

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        """Reset the world and return initial observations."""
        if seed is not None:
            self.rng = make_rng(seed)

        self.step_count = 0
        self.last_winner = None
        self.next_cell_id = 0
        self.next_pellet_id = 0
        self.agents = {agent_id: [] for agent_id in self.agent_ids}

        positions = self._sample_spawn_positions(self.config.num_agents)
        for idx, agent_id in enumerate(self.agent_ids):
            cell = Cell(
                cell_id=self._new_cell_id(),
                agent_id=agent_id,
                position=positions[idx].copy(),
                velocity=np.zeros(2, dtype=np.float32),
                mass=25.0,
                split_cooldown=0,
                merge_cooldown=0,
                eject_cooldown=0,
            )
            self.agents[agent_id].append(cell)
            self.snapshots[agent_id] = AgentSnapshot(total_mass=cell.mass, alive=True)

        self.pellets = []
        self._respawn_pellets(force_full=True)
        self._sync_prev_cell_positions()
        return self.get_observations()

    def _sample_spawn_positions(self, count: int) -> list[np.ndarray]:
        positions: list[np.ndarray] = []
        min_dist = 30.0
        margin = 20.0
        for _ in range(count):
            for _attempt in range(400):
                candidate = self.rng.uniform(
                    low=margin,
                    high=max(margin + 1.0, self.map_size - margin),
                    size=(2,),
                ).astype(np.float32)
                if all(np.sum((candidate - other) ** 2) >= min_dist * min_dist for other in positions):
                    positions.append(candidate)
                    break
            else:
                positions.append(
                    self.rng.uniform(low=0.0, high=self.map_size, size=(2,)).astype(np.float32)
                )
        return positions

    def _new_cell_id(self) -> int:
        self.next_cell_id += 1
        return self.next_cell_id

    def _new_pellet_id(self) -> int:
        self.next_pellet_id += 1
        return self.next_pellet_id

    def step(
        self,
        actions: dict[str, np.ndarray],
        dt: float = 1.0,
        compute_observations: bool = True,
    ) -> StepOutcome:
        """Advance one simulation tick."""
        self._capture_prev_cell_positions()
        self.step_count += 1
        dt_scale = max(0.01, float(dt) * float(self.config.simulation.physics_hz))

        for agent_id in self.agent_ids:
            action = actions.get(agent_id)
            direction, split_enabled, direction_bucket = self._decode_action(action)
            self.snapshots[agent_id].record_direction(direction_bucket)
            self._apply_agent_action(
                agent_id=agent_id,
                direction=direction,
                split_enabled=split_enabled,
                dt_scale=dt_scale,
            )

        self._consume_pellets()
        elimination_pairs = self._resolve_cell_eating()
        self._merge_cells_if_ready()
        self._respawn_pellets(force_full=False)

        rewards, dones, infos = self._compute_rewards_and_info(elimination_pairs)
        observations = self.get_observations() if compute_observations else None
        return StepOutcome(observations=observations, rewards=rewards, dones=dones, infos=infos)

    def _capture_prev_cell_positions(self) -> None:
        self.prev_cell_positions = {
            cell.cell_id: cell.position.copy()
            for cells in self.agents.values()
            for cell in cells
        }

    def _sync_prev_cell_positions(self) -> None:
        self.prev_cell_positions = {
            cell.cell_id: cell.position.copy()
            for cells in self.agents.values()
            for cell in cells
        }

    def previous_cell_position(self, cell: Cell) -> np.ndarray:
        return self.prev_cell_positions.get(cell.cell_id, cell.position)

    def _decode_action(self, action: np.ndarray | None) -> tuple[np.ndarray, bool, int]:
        mode = self.config.simulation.action_mode
        if mode == "continuous":
            raw = np.asarray(action if action is not None else [0.0, 0.0, 0.0], dtype=np.float32).reshape(-1)
            steer = np.zeros(2, dtype=np.float32)
            if raw.size >= 2:
                steer[0] = float(np.clip(raw[0], -1.0, 1.0))
                steer[1] = float(np.clip(raw[1], -1.0, 1.0))
            norm = float(np.sqrt(np.sum(steer * steer)))
            if norm > 1.0:
                steer = steer / norm
            split_enabled = bool(raw.size >= 3 and raw[2] >= 0.5)
            return steer, split_enabled, self._vector_to_direction_bucket(steer)

        raw = np.asarray(action if action is not None else [0, 0], dtype=np.int64).reshape(-1)
        direction_idx = int(np.clip(raw[0] if raw.size > 0 else 0, 0, len(DIRECTION_VECTORS) - 1))
        split_enabled = bool(raw.size > 1 and int(raw[1]) == 1)
        return DIRECTION_VECTORS[direction_idx].copy(), split_enabled, direction_idx

    def _vector_to_direction_bucket(self, vector: np.ndarray) -> int:
        norm = float(np.sqrt(np.sum(vector * vector)))
        if norm <= 1e-6:
            return 0
        unit = vector / norm
        dots = DIRECTION_VECTORS[1:] @ unit
        return int(np.argmax(dots)) + 1

    def _apply_agent_action(
        self,
        agent_id: str,
        direction: np.ndarray,
        split_enabled: bool,
        dt_scale: float,
    ) -> None:
        cells = self.agents.get(agent_id, [])
        if not cells:
            return

        if split_enabled:
            self._try_split(agent_id, direction)

        drag = float(np.clip(self.config.physics.drag, 0.0, 0.999))
        effective_drag = drag ** dt_scale
        for cell in cells:
            speed = self.config.physics.base_speed / (
                1.0 + self.config.physics.speed_mass_factor * np.sqrt(max(cell.mass, 0.1))
            )
            acceleration = direction * (speed * 0.9 * dt_scale)
            cell.velocity = effective_drag * cell.velocity + acceleration
            velocity_norm = float(np.sqrt(np.sum(cell.velocity * cell.velocity)))
            max_velocity = speed * 2.2
            if velocity_norm > max_velocity and velocity_norm > 1e-6:
                cell.velocity = (cell.velocity / velocity_norm) * max_velocity

            cell.position = cell.position + cell.velocity * dt_scale
            self._apply_boundary_collision(cell)

            if cell.split_cooldown > 0:
                cell.split_cooldown -= 1
            if cell.merge_cooldown > 0:
                cell.merge_cooldown -= 1
            if cell.eject_cooldown > 0:
                cell.eject_cooldown -= 1

    def _apply_boundary_collision(self, cell: Cell) -> None:
        if cell.position[0] < 0.0:
            cell.position[0] = 0.0
            cell.velocity[0] = 0.0
        elif cell.position[0] > self.map_size:
            cell.position[0] = self.map_size
            cell.velocity[0] = 0.0

        if cell.position[1] < 0.0:
            cell.position[1] = 0.0
            cell.velocity[1] = 0.0
        elif cell.position[1] > self.map_size:
            cell.position[1] = self.map_size
            cell.velocity[1] = 0.0

        cell.position = cell.position.astype(np.float32)

    def _try_split(self, agent_id: str, direction: np.ndarray) -> None:
        cells = self.agents[agent_id]
        if len(cells) >= self.config.physics.max_cells_per_agent:
            return

        largest = max(cells, key=lambda c: c.mass)
        if largest.mass < self.config.physics.min_split_mass:
            return
        if largest.split_cooldown > 0:
            return

        norm = float(np.sqrt(np.sum(direction * direction)))
        if norm <= 1e-6:
            angle = self.rng.uniform(0.0, 2.0 * np.pi)
            direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        else:
            direction = (direction / norm).astype(np.float32)

        original_radius = largest.radius(self.config.physics.radius_scale)
        new_mass = largest.mass * 0.5
        largest.mass = new_mass
        largest.split_cooldown = self.config.physics.split_cooldown_steps
        largest.merge_cooldown = self.config.physics.merge_cooldown_steps
        largest.velocity = largest.velocity + direction * self.config.physics.split_boost

        new_position = largest.position + direction * (original_radius + 2.0)
        new_position = np.clip(new_position, 0.0, self.map_size).astype(np.float32)
        new_cell = Cell(
            cell_id=self._new_cell_id(),
            agent_id=agent_id,
            position=new_position,
            velocity=direction * self.config.physics.split_boost,
            mass=new_mass,
            split_cooldown=self.config.physics.split_cooldown_steps,
            merge_cooldown=self.config.physics.merge_cooldown_steps,
            eject_cooldown=0,
        )
        cells.append(new_cell)

    def eject_mass(self, agent_id: str, direction: np.ndarray) -> None:
        """Optional mass ejection mechanic (disabled by default)."""
        if not self.config.physics.enable_eject_mechanic:
            return

        cells = self.agents.get(agent_id, [])
        if not cells:
            return

        largest = max(cells, key=lambda c: c.mass)
        if largest.eject_cooldown > 0 or largest.mass <= self.config.physics.eject_mass_amount + 1.0:
            return
        largest.mass -= self.config.physics.eject_mass_amount
        largest.eject_cooldown = self.config.physics.eject_cooldown_steps

        norm = float(np.sqrt(np.sum(direction * direction)))
        if norm <= 1e-6:
            return
        direction = direction / norm

        pellet_position = largest.position + direction * (largest.radius(self.config.physics.radius_scale) + 2.0)
        pellet = Pellet(
            pellet_id=self._new_pellet_id(),
            position=np.clip(pellet_position, 0.0, self.map_size).astype(np.float32),
            mass=self.config.physics.eject_mass_amount,
        )
        self.pellets.append(pellet)

    def _consume_pellets(self) -> None:
        if not self.pellets:
            return

        cell_refs: list[Cell] = []
        for agent_cells in self.agents.values():
            cell_refs.extend(agent_cells)
        if not cell_refs:
            return

        pellet_positions = np.stack([pellet.position for pellet in self.pellets], axis=0)
        pellet_masses = np.array([pellet.mass for pellet in self.pellets], dtype=np.float32)
        cell_positions = np.stack([cell.position for cell in cell_refs], axis=0)
        radii_sq = np.array(
            [cell.radius(self.config.physics.radius_scale) ** 2 for cell in cell_refs],
            dtype=np.float32,
        )

        deltas = pellet_positions[:, None, :] - cell_positions[None, :, :]
        dist_sq = np.sum(deltas * deltas, axis=2)
        can_eat = dist_sq <= radii_sq[None, :]
        eaten_mask = np.any(can_eat, axis=1)
        if not np.any(eaten_mask):
            return

        eater_indices = np.argmax(can_eat[eaten_mask], axis=1)
        eaten_pellet_indices = np.nonzero(eaten_mask)[0]
        mass_gain = np.zeros(len(cell_refs), dtype=np.float32)
        np.add.at(mass_gain, eater_indices, pellet_masses[eaten_pellet_indices])
        for idx, gain in enumerate(mass_gain):
            if gain > 0:
                cell_refs[idx].mass += float(gain)

        self.pellets = [pellet for idx, pellet in enumerate(self.pellets) if not eaten_mask[idx]]

    def _resolve_cell_eating(self) -> list[tuple[str, str]]:
        eat_ratio = self.config.physics.eat_mass_ratio
        assimilation = self.config.physics.assimilation_efficiency

        cells: list[Cell] = []
        for agent_cells in self.agents.values():
            cells.extend(agent_cells)
        cells.sort(key=lambda c: c.mass, reverse=True)

        consumed_ids: set[int] = set()
        elimination_sources: dict[str, list[str]] = defaultdict(list)

        for eater in cells:
            if eater.cell_id in consumed_ids:
                continue
            eater_radius_sq = eater.radius(self.config.physics.radius_scale) ** 2
            for target in cells:
                if target.cell_id == eater.cell_id or target.cell_id in consumed_ids:
                    continue
                if eater.agent_id == target.agent_id:
                    continue
                if eater.mass < eat_ratio * target.mass:
                    continue
                dist_sq = float(np.sum((eater.position - target.position) ** 2))
                if dist_sq <= eater_radius_sq:
                    consumed_ids.add(target.cell_id)
                    eater.mass += target.mass * assimilation
                    elimination_sources[target.agent_id].append(eater.agent_id)

        for agent_id, agent_cells in self.agents.items():
            self.agents[agent_id] = [cell for cell in agent_cells if cell.cell_id not in consumed_ids]

        elimination_pairs: list[tuple[str, str]] = []
        for victim_id, killers in elimination_sources.items():
            if self.agents[victim_id]:
                continue
            killer = killers[-1]
            elimination_pairs.append((killer, victim_id))
        return elimination_pairs

    def _merge_cells_if_ready(self) -> None:
        for agent_id, cells in self.agents.items():
            if len(cells) <= 1:
                continue
            if any(cell.merge_cooldown > 0 for cell in cells):
                continue

            total_mass = sum(cell.mass for cell in cells)
            weighted_position = sum(cell.position * cell.mass for cell in cells) / max(total_mass, 1e-6)
            weighted_velocity = sum(cell.velocity * cell.mass for cell in cells) / max(total_mass, 1e-6)

            merged = Cell(
                cell_id=self._new_cell_id(),
                agent_id=agent_id,
                position=weighted_position.astype(np.float32),
                velocity=weighted_velocity.astype(np.float32),
                mass=total_mass,
                split_cooldown=0,
                merge_cooldown=0,
                eject_cooldown=0,
            )
            self.agents[agent_id] = [merged]

    def _respawn_pellets(self, force_full: bool) -> None:
        target = self.target_pellet_count
        if force_full:
            to_add = max(0, target - len(self.pellets))
        else:
            to_add = min(
                self.config.map.pellet_respawn_per_step,
                max(0, target - len(self.pellets)),
            )

        for _ in range(to_add):
            pellet = Pellet(
                pellet_id=self._new_pellet_id(),
                position=self.rng.uniform(0.0, self.map_size, size=(2,)).astype(np.float32),
                mass=float(self.config.map.pellet_mass),
            )
            self.pellets.append(pellet)

    def _compute_rewards_and_info(
        self,
        elimination_pairs: list[tuple[str, str]],
    ) -> tuple[dict[str, float], dict[str, bool], dict[str, dict[str, Any]]]:
        elimination_by_agent: dict[str, int] = defaultdict(int)
        eliminated_agents: set[str] = set()
        for killer, victim in elimination_pairs:
            elimination_by_agent[killer] += 1
            eliminated_agents.add(victim)

        rewards: dict[str, float] = {}
        infos: dict[str, dict[str, Any]] = {}

        alive_count = len(self.alive_agents)
        global_done = self.step_count >= self.config.max_steps or alive_count <= 1
        time_frac = self.step_count / max(1, self.config.max_steps)

        winner: str | None = None
        if global_done and alive_count == 1:
            winner = self.alive_agents[0]
            self.last_winner = winner
        elif global_done:
            self.last_winner = None

        for agent_id in self.agent_ids:
            total_mass = self._agent_total_mass(agent_id)
            alive = bool(self.agents[agent_id])
            prev_mass = self.snapshots[agent_id].total_mass
            delta_mass = total_mass - prev_mass

            reward = (
                self.config.rewards.mass_gain_scale * delta_mass
                + self.config.rewards.time_penalty
                + elimination_by_agent.get(agent_id, 0) * self.config.rewards.elimination_bonus
            )
            if agent_id in eliminated_agents:
                reward += self.config.rewards.death_penalty
            if alive and time_frac >= self.config.rewards.survival_bonus_start_frac:
                reward += self.config.rewards.survival_bonus

            rewards[agent_id] = float(reward)
            self.snapshots[agent_id].total_mass = total_mass
            self.snapshots[agent_id].alive = alive
            self.snapshots[agent_id].episode_return += reward
            self.snapshots[agent_id].eliminated_opponents += elimination_by_agent.get(agent_id, 0)

            infos[agent_id] = {
                "alive": alive,
                "total_mass": total_mass,
                "delta_mass": delta_mass,
                "episode_return": self.snapshots[agent_id].episode_return,
                "eliminations": self.snapshots[agent_id].eliminated_opponents,
                "recent_direction_counts": list(self.snapshots[agent_id].recent_direction_counts),
                "winner": winner,
            }

        dones: dict[str, bool] = {}
        for agent_id in self.agent_ids:
            agent_done = global_done or not self.snapshots[agent_id].alive
            dones[agent_id] = agent_done
        dones["__all__"] = global_done

        infos["__global__"] = {
            "step": self.step_count,
            "alive_count": alive_count,
            "stage": self.stage,
            "map_size": self.map_size,
            "winner": winner,
            "auto_curriculum": self.auto_curriculum,
        }

        if global_done:
            score = 1.0 if winner is not None else 0.0
            self.curriculum_scores.append(score)
            if self.auto_curriculum:
                self._maybe_advance_curriculum()

        return rewards, dones, infos

    def _maybe_advance_curriculum(self) -> None:
        if not self.config.curriculum.enabled:
            return
        if len(self.curriculum_scores) < self.config.curriculum.advance_window:
            return
        if self.step_count < self.config.curriculum.min_stage_steps:
            return

        avg_score = float(np.mean(self.curriculum_scores))
        if avg_score < self.config.curriculum.advance_survival_rate:
            return

        current_size = self.map_size
        next_size = min(
            float(self.config.map.max_size),
            current_size * float(self.config.curriculum.stage_scale),
        )
        if next_size <= current_size + 1e-6:
            return
        self.stage += 1
        self._set_map_size(next_size)
        self.curriculum_scores.clear()

    def adjust_map_size(self, increase: bool) -> None:
        """Manual map scaling from supervisor controls."""
        factor = float(self.config.curriculum.stage_scale)
        candidate = self.map_size * factor if increase else self.map_size / factor
        candidate = float(np.clip(candidate, self.config.map.start_size, self.config.map.max_size))
        if abs(candidate - self.map_size) < 1e-6:
            return
        self.stage = max(0, self.stage + (1 if increase else -1))
        self._set_map_size(candidate)

    def _set_map_size(self, new_size: float) -> None:
        old_size = self.map_size
        self.map_size = float(new_size)
        if old_size <= 1e-6:
            return
        scale = self.map_size / old_size
        for cells in self.agents.values():
            for cell in cells:
                cell.position = np.clip(cell.position * scale, 0.0, self.map_size).astype(np.float32)
        for pellet in self.pellets:
            pellet.position = np.clip(pellet.position * scale, 0.0, self.map_size).astype(np.float32)
        self._respawn_pellets(force_full=False)
        self._sync_prev_cell_positions()

    def _agent_total_mass(self, agent_id: str) -> float:
        return float(sum(cell.mass for cell in self.agents[agent_id]))

    def _agent_largest_cell(self, agent_id: str) -> Cell | None:
        cells = self.agents[agent_id]
        if not cells:
            return None
        return max(cells, key=lambda c: c.mass)

    def _agent_center(self, agent_id: str) -> np.ndarray:
        cells = self.agents[agent_id]
        if not cells:
            return np.array([self.map_size * 0.5, self.map_size * 0.5], dtype=np.float32)
        masses = np.array([cell.mass for cell in cells], dtype=np.float32)
        stacked = np.stack([cell.position for cell in cells], axis=0)
        return (stacked * masses[:, None]).sum(axis=0) / max(float(masses.sum()), 1e-6)

    def _norm_mass(self, mass: float) -> float:
        return float(mass / 250.0)

    def get_observations(self) -> dict[str, np.ndarray]:
        pellet_positions = (
            np.stack([pellet.position for pellet in self.pellets], axis=0)
            if self.pellets
            else np.zeros((0, 2), dtype=np.float32)
        )
        pellet_masses = (
            np.array([pellet.mass for pellet in self.pellets], dtype=np.float32)
            if self.pellets
            else np.zeros((0,), dtype=np.float32)
        )

        cell_positions: list[np.ndarray] = []
        cell_masses: list[float] = []
        cell_owners: list[int] = []
        for agent_id, cells in self.agents.items():
            owner_idx = self.agent_index[agent_id]
            for cell in cells:
                cell_positions.append(cell.position)
                cell_masses.append(cell.mass)
                cell_owners.append(owner_idx)
        all_cell_positions = (
            np.stack(cell_positions, axis=0) if cell_positions else np.zeros((0, 2), dtype=np.float32)
        )
        all_cell_masses = (
            np.array(cell_masses, dtype=np.float32) if cell_masses else np.zeros((0,), dtype=np.float32)
        )
        all_cell_owners = (
            np.array(cell_owners, dtype=np.int32) if cell_owners else np.zeros((0,), dtype=np.int32)
        )

        return {
            agent_id: self._build_observation(
                agent_id=agent_id,
                pellet_positions=pellet_positions,
                pellet_masses=pellet_masses,
                all_cell_positions=all_cell_positions,
                all_cell_masses=all_cell_masses,
                all_cell_owners=all_cell_owners,
            )
            for agent_id in self.agent_ids
        }

    def _top_k_indices(self, dist_sq: np.ndarray, k: int) -> np.ndarray:
        if dist_sq.size == 0 or k <= 0:
            return np.zeros((0,), dtype=np.int64)
        if dist_sq.size <= k:
            return np.argsort(dist_sq)
        part = np.argpartition(dist_sq, k - 1)[:k]
        return part[np.argsort(dist_sq[part])]

    def _build_observation(
        self,
        agent_id: str,
        pellet_positions: np.ndarray,
        pellet_masses: np.ndarray,
        all_cell_positions: np.ndarray,
        all_cell_masses: np.ndarray,
        all_cell_owners: np.ndarray,
    ) -> np.ndarray:
        obs = np.zeros(self.observation_dim, dtype=np.float32)
        cells = self.agents[agent_id]
        alive = len(cells) > 0

        cursor = 0
        if alive:
            largest = self._agent_largest_cell(agent_id)
            assert largest is not None
            total_mass = self._agent_total_mass(agent_id)
            speed_norm = self.config.physics.base_speed + self.config.physics.split_boost + 1e-6
            obs[cursor : cursor + 8] = np.array(
                [
                    largest.position[0] / self.map_size,
                    largest.position[1] / self.map_size,
                    largest.velocity[0] / speed_norm,
                    largest.velocity[1] / speed_norm,
                    self._norm_mass(total_mass),
                    len(cells) / max(1.0, float(self.config.physics.max_cells_per_agent)),
                    largest.split_cooldown / max(1.0, float(self.config.physics.split_cooldown_steps)),
                    largest.merge_cooldown / max(1.0, float(self.config.physics.merge_cooldown_steps)),
                ],
                dtype=np.float32,
            )
            center = self._agent_center(agent_id)
        else:
            center = np.array([self.map_size * 0.5, self.map_size * 0.5], dtype=np.float32)
        cursor += 8

        pellet_rel = pellet_positions - center[None, :] if pellet_positions.size > 0 else np.zeros((0, 2), dtype=np.float32)
        pellet_dist_sq = np.sum(pellet_rel * pellet_rel, axis=1) if pellet_rel.size > 0 else np.zeros((0,), dtype=np.float32)
        pellet_indices = self._top_k_indices(pellet_dist_sq, self.config.nearest_pellets)
        for idx in pellet_indices:
            delta = pellet_rel[idx] / max(self.map_size, 1e-6)
            obs[cursor : cursor + 3] = np.array(
                [delta[0], delta[1], self._norm_mass(float(pellet_masses[idx]))],
                dtype=np.float32,
            )
            cursor += 3
        cursor += 3 * max(0, self.config.nearest_pellets - pellet_indices.size)

        owner_idx = self.agent_index[agent_id]
        opponent_mask = all_cell_owners != owner_idx
        opp_positions = all_cell_positions[opponent_mask]
        opp_masses = all_cell_masses[opponent_mask]

        own_mass = max(self._agent_total_mass(agent_id), 1e-6)
        opp_rel = opp_positions - center[None, :] if opp_positions.size > 0 else np.zeros((0, 2), dtype=np.float32)
        opp_dist_sq = np.sum(opp_rel * opp_rel, axis=1) if opp_rel.size > 0 else np.zeros((0,), dtype=np.float32)
        opp_indices = self._top_k_indices(opp_dist_sq, self.config.nearest_opponents)
        for idx in opp_indices:
            delta = opp_rel[idx] / max(self.map_size, 1e-6)
            obs[cursor : cursor + 4] = np.array(
                [
                    delta[0],
                    delta[1],
                    float(opp_masses[idx] / own_mass),
                    1.0,
                ],
                dtype=np.float32,
            )
            cursor += 4
        cursor += 4 * max(0, self.config.nearest_opponents - opp_indices.size)

        alive_fraction = len(self.alive_agents) / max(1.0, float(self.config.num_agents))
        obs[cursor : cursor + 4] = np.array(
            [
                self.map_size / max(1.0, float(self.config.map.max_size)),
                1.0 - (self.step_count / max(1.0, float(self.config.max_steps))),
                alive_fraction,
                self.stage / 10.0,
            ],
            dtype=np.float32,
        )
        return obs
