"""Deterministic Agar.io-like world simulation for 3 RL agents."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from agario_rl import AgarioConfig
from agario_rl.env.entities import AgentSnapshot, Cell, EjectedMass, Pellet, Virus
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
        self.next_virus_id = 0
        self.next_ejected_id = 0
        self.agents: dict[str, list[Cell]] = {}
        self.pellets: list[Pellet] = []
        self.viruses: list[Virus] = []
        self.ejected_masses: list[EjectedMass] = []
        self.prev_cell_positions: dict[int, np.ndarray] = {}
        self._center_cache: dict[str, np.ndarray] = {}
        self._center_cache_step: int = -1
        self.step_split_attempts: dict[str, int] = {}
        self.step_successful_splits: dict[str, int] = {}
        self.step_unsafe_splits: dict[str, int] = {}
        self.step_useful_splits: dict[str, int] = {}
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
        extra_features = 0
        if self.config.observation_features.enabled:
            if self.config.observation_features.include_threats:
                extra_features += 8
            if self.config.observation_features.include_viruses:
                extra_features += 3
            if self.config.observation_features.include_eject_state:
                extra_features += 2
        return self_features + pellet_features + opp_features + global_features + extra_features

    @property
    def alive_agents(self) -> list[str]:
        return [agent_id for agent_id, cells in self.agents.items() if cells]

    @property
    def target_pellet_count(self) -> int:
        area = self.map_size * self.map_size
        base = (area / 10_000.0) * float(self.config.map.pellets_per_10k_area)
        return max(20, int(base))

    @property
    def target_virus_count(self) -> int:
        if not self.config.viruses.enabled:
            return 0
        target = min(self.config.viruses.initial_count, self.config.viruses.max_count)
        if not self.config.scenario_curriculum.enabled:
            return target
        scenario = self.scenario_name
        if scenario in {"pellet_growth", "evasion", "hunting"}:
            return 0
        if scenario == "virus_control":
            return max(1, target // 2)
        return target

    @property
    def scenario_name(self) -> str:
        if not self.config.scenario_curriculum.enabled:
            return str(self.config.scenario_curriculum.preset)
        names = self.config.scenario_curriculum.stage_names
        if not names:
            return str(self.config.scenario_curriculum.preset)
        return names[min(self.stage, len(names) - 1)]

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        """Reset the world and return initial observations."""
        if seed is not None:
            self.rng = make_rng(seed)

        self.step_count = 0
        self.last_winner = None
        self.next_cell_id = 0
        self.next_pellet_id = 0
        self.next_virus_id = 0
        self.next_ejected_id = 0
        self.agents = {agent_id: [] for agent_id in self.agent_ids}
        self._center_cache = {}
        self._center_cache_step = -1

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
        self.viruses = []
        self.ejected_masses = []
        self._respawn_pellets(force_full=True)
        self._respawn_viruses(force_full=True)
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

    def _new_virus_id(self) -> int:
        self.next_virus_id += 1
        return self.next_virus_id

    def _new_ejected_id(self) -> int:
        self.next_ejected_id += 1
        return self.next_ejected_id

    def step(
        self,
        actions: dict[str, np.ndarray],
        dt: float = 1.0,
        compute_observations: bool = True,
    ) -> StepOutcome:
        """Advance one simulation tick."""
        self._capture_prev_cell_positions()
        self.step_count += 1
        self.step_split_attempts = defaultdict(int)
        self.step_successful_splits = defaultdict(int)
        self.step_unsafe_splits = defaultdict(int)
        self.step_useful_splits = defaultdict(int)
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

        self._move_ejected_masses(dt_scale)
        self._consume_pellets()
        self._consume_ejected_masses()
        virus_splits = self._resolve_virus_interactions()
        elimination_pairs = self._resolve_cell_eating()
        self._merge_cells_if_ready()
        self._apply_mass_decay(float(dt))
        respawned_agents = self._respawn_eliminated_agents(elimination_pairs)
        self._constrain_cells_to_bounds()
        if respawned_agents:
            self._center_cache = {}
        self._respawn_pellets(force_full=False)
        self._respawn_viruses(force_full=False)

        rewards, dones, infos = self._compute_rewards_and_info(
            elimination_pairs=elimination_pairs,
            virus_splits=virus_splits,
            respawned_agents=respawned_agents,
        )
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
            self.step_split_attempts[agent_id] += 1
            useful_split, unsafe_split = self._classify_split_context(agent_id, direction)
            if self._try_split(agent_id, direction):
                self.step_successful_splits[agent_id] += 1
                if useful_split:
                    self.step_useful_splits[agent_id] += 1
                if unsafe_split:
                    self.step_unsafe_splits[agent_id] += 1

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

            # The split/eject impulse decays on its own schedule rather than
            # being clamped to normal movement speed, so a split still reads
            # as a fast outward lunge (like Agar.io) instead of being capped
            # to the target cell's cruising speed the instant it happens.
            cell.boost_velocity = (effective_drag * cell.boost_velocity).astype(np.float32)

            cell.position = cell.position + (cell.velocity + cell.boost_velocity) * dt_scale
            self._apply_boundary_collision(cell)

            if cell.split_cooldown > 0:
                cell.split_cooldown -= 1
            if cell.merge_cooldown > 0:
                cell.merge_cooldown -= 1
            if cell.eject_cooldown > 0:
                cell.eject_cooldown -= 1

    def _apply_boundary_collision(self, cell: Cell) -> None:
        radius = min(cell.radius(self.config.physics.radius_scale), self.map_size * 0.5)
        lower = radius
        upper = self.map_size - radius
        if cell.position[0] < lower:
            cell.position[0] = lower
            cell.velocity[0] = 0.0
            cell.boost_velocity[0] = 0.0
        elif cell.position[0] > upper:
            cell.position[0] = upper
            cell.velocity[0] = 0.0
            cell.boost_velocity[0] = 0.0

        if cell.position[1] < lower:
            cell.position[1] = lower
            cell.velocity[1] = 0.0
            cell.boost_velocity[1] = 0.0
        elif cell.position[1] > upper:
            cell.position[1] = upper
            cell.velocity[1] = 0.0
            cell.boost_velocity[1] = 0.0

        cell.position = cell.position.astype(np.float32)

    def _constrain_cells_to_bounds(self) -> None:
        for cells in self.agents.values():
            for cell in cells:
                self._apply_boundary_collision(cell)

    def _try_split(self, agent_id: str, direction: np.ndarray) -> bool:
        cells = self.agents[agent_id]
        if len(cells) >= self.config.physics.max_cells_per_agent:
            return False

        largest = max(cells, key=lambda c: c.mass)
        if largest.mass < self.config.physics.min_split_mass:
            return False
        if largest.split_cooldown > 0:
            return False

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
        # Only the newly spawned piece gets the outward lunge; the original
        # keeps its current velocity so the two visibly separate (Agar.io's
        # split "shoots" one piece away from the other rather than moving
        # both cells in lockstep at an identical boosted speed).

        new_position = largest.position + direction * (original_radius + 2.0)
        new_position = np.clip(new_position, 0.0, self.map_size).astype(np.float32)
        new_cell = Cell(
            cell_id=self._new_cell_id(),
            agent_id=agent_id,
            position=new_position,
            velocity=np.zeros(2, dtype=np.float32),
            mass=new_mass,
            split_cooldown=self.config.physics.split_cooldown_steps,
            merge_cooldown=self.config.physics.merge_cooldown_steps,
            eject_cooldown=0,
            boost_velocity=(direction * self.config.physics.split_boost).astype(np.float32),
        )
        cells.append(new_cell)
        return True

    def _classify_split_context(self, agent_id: str, direction: np.ndarray) -> tuple[bool, bool]:
        cells = self.agents.get(agent_id, [])
        if not cells:
            return False, False

        largest = max(cells, key=lambda c: c.mass)
        own_mass = max(self._agent_total_mass(agent_id), 1e-6)
        split_range = max(
            largest.radius(self.config.physics.radius_scale) * 4.8,
            self.config.physics.split_boost * 10.0,
        )
        useful = False
        unsafe = False
        center = self._agent_center(agent_id)
        norm = float(np.linalg.norm(direction))
        aim = direction / norm if norm > 1e-6 else np.zeros((2,), dtype=np.float32)

        for other_id in self.agent_ids:
            if other_id == agent_id or not self.agents[other_id]:
                continue
            other_center = self._agent_center(other_id)
            delta = other_center - center
            distance = float(np.linalg.norm(delta))
            other_mass = self._agent_total_mass(other_id)
            ratio = other_mass / own_mass
            if ratio >= 0.95 and distance <= self.map_size * 0.18:
                unsafe = True
            if ratio <= 0.58 and distance <= split_range:
                if norm <= 1e-6:
                    useful = True
                else:
                    target_norm = float(np.linalg.norm(delta))
                    if target_norm <= 1e-6 or float(np.dot(aim, delta / target_norm)) >= 0.45:
                        useful = True

        if self.config.viruses.enabled and self.viruses and own_mass >= self.config.viruses.min_split_mass:
            for virus in self.viruses:
                distance = float(np.linalg.norm(virus.position - center))
                if distance <= self.map_size * 0.10:
                    unsafe = True
                    break
        return useful, unsafe

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

        norm = float(np.sqrt(np.sum(direction * direction)))
        if norm <= 1e-6:
            return
        direction = direction / norm
        largest.mass -= self.config.physics.eject_mass_amount
        largest.eject_cooldown = self.config.physics.eject_cooldown_steps

        pellet_position = largest.position + direction * (largest.radius(self.config.physics.radius_scale) + 2.0)
        pellet = Pellet(
            pellet_id=self._new_pellet_id(),
            position=np.clip(pellet_position, 0.0, self.map_size).astype(np.float32),
            mass=self.config.physics.eject_mass_amount,
        )
        ejected = EjectedMass(
            ejected_id=self._new_ejected_id(),
            owner_id=agent_id,
            position=pellet.position.copy(),
            velocity=(direction * self.config.physics.eject_speed).astype(np.float32),
            mass=float(pellet.mass),
        )
        self.ejected_masses.append(ejected)

    def _move_ejected_masses(self, dt_scale: float) -> None:
        if not self.ejected_masses:
            return
        drag = float(np.clip(self.config.physics.drag, 0.0, 0.999))
        effective_drag = drag ** max(1.0, dt_scale)
        kept: list[EjectedMass] = []
        for ejected in self.ejected_masses:
            ejected.position = (ejected.position + ejected.velocity * dt_scale).astype(np.float32)
            ejected.velocity = (ejected.velocity * effective_drag).astype(np.float32)
            ejected.age_steps += 1
            if ejected.age_steps > 240:
                continue
            if np.any(ejected.position < 0.0) or np.any(ejected.position > self.map_size):
                continue
            kept.append(ejected)
        self.ejected_masses = kept

    def _consume_ejected_masses(self) -> None:
        if not self.ejected_masses:
            return

        consumed: set[int] = set()
        if self.config.viruses.enabled and self.viruses:
            for ejected in self.ejected_masses:
                for virus in self.viruses:
                    radius = virus.radius(self.config.physics.radius_scale)
                    dist_sq = float(np.sum((ejected.position - virus.position) ** 2))
                    if dist_sq > radius * radius:
                        continue
                    consumed.add(ejected.ejected_id)
                    virus.mass += ejected.mass
                    virus.fed_count += 1
                    self._maybe_split_fed_virus(virus, ejected.velocity)
                    break

        cell_refs = [cell for cells in self.agents.values() for cell in cells]
        if cell_refs:
            for ejected in self.ejected_masses:
                if ejected.ejected_id in consumed:
                    continue
                for cell in cell_refs:
                    radius = cell.radius(self.config.physics.radius_scale)
                    dist_sq = float(np.sum((ejected.position - cell.position) ** 2))
                    if dist_sq > radius * radius:
                        continue
                    cell.mass += ejected.mass
                    consumed.add(ejected.ejected_id)
                    break

        if consumed:
            self.ejected_masses = [
                ejected
                for ejected in self.ejected_masses
                if ejected.ejected_id not in consumed
            ]

    def _maybe_split_fed_virus(self, virus: Virus, feed_velocity: np.ndarray) -> None:
        if virus.fed_count < self.config.viruses.feed_to_split:
            return
        if len(self.viruses) >= self.config.viruses.max_count:
            virus.fed_count = 0
            virus.mass = float(self.config.viruses.mass)
            return

        norm = float(np.linalg.norm(feed_velocity))
        if norm <= 1e-6:
            angle = self.rng.uniform(0.0, 2.0 * np.pi)
            direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        else:
            direction = (feed_velocity / norm).astype(np.float32)
        spawn_position = virus.position + direction * self.config.viruses.split_spawn_distance
        spawn_position = np.clip(spawn_position, 0.0, self.map_size).astype(np.float32)
        self.viruses.append(
            Virus(
                virus_id=self._new_virus_id(),
                position=spawn_position,
                mass=float(self.config.viruses.mass),
            )
        )
        virus.fed_count = 0
        virus.mass = float(self.config.viruses.mass)

    def _resolve_virus_interactions(self) -> dict[str, int]:
        if not self.config.viruses.enabled or not self.viruses:
            return {}

        consumed_viruses: set[int] = set()
        splits_by_agent: dict[str, int] = defaultdict(int)
        for virus in self.viruses:
            if virus.virus_id in consumed_viruses:
                continue
            virus_radius = virus.radius(self.config.physics.radius_scale)
            for agent_id, cells in self.agents.items():
                for cell in list(cells):
                    if cell.mass < self.config.viruses.min_split_mass:
                        continue
                    dist_sq = float(np.sum((cell.position - virus.position) ** 2))
                    if dist_sq > max(virus_radius, cell.radius(self.config.physics.radius_scale)) ** 2:
                        continue
                    consumed_viruses.add(virus.virus_id)
                    cell.mass += virus.mass * self.config.viruses.consumption_efficiency
                    self._burst_cell_from_virus(agent_id, cell)
                    splits_by_agent[agent_id] += 1
                    break
                if virus.virus_id in consumed_viruses:
                    break

        if consumed_viruses:
            self.viruses = [
                virus
                for virus in self.viruses
                if virus.virus_id not in consumed_viruses
            ]
        return dict(splits_by_agent)

    def _burst_cell_from_virus(self, agent_id: str, cell: Cell) -> None:
        cells = self.agents[agent_id]
        available_slots = max(0, self.config.physics.max_cells_per_agent - len(cells))
        pieces_to_add = min(max(0, self.config.viruses.split_pieces - 1), available_slots)
        if pieces_to_add <= 0:
            return

        total_pieces = pieces_to_add + 1
        piece_mass = max(1.0, cell.mass / total_pieces)
        cell.mass = piece_mass
        cell.merge_cooldown = self.config.physics.merge_cooldown_steps
        cell.split_cooldown = self.config.physics.split_cooldown_steps

        for piece_idx in range(pieces_to_add):
            angle = (2.0 * np.pi * piece_idx / max(1, pieces_to_add)) + self.rng.uniform(-0.12, 0.12)
            direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
            new_position = cell.position + direction * (cell.radius(self.config.physics.radius_scale) + 3.0)
            cells.append(
                Cell(
                    cell_id=self._new_cell_id(),
                    agent_id=agent_id,
                    position=np.clip(new_position, 0.0, self.map_size).astype(np.float32),
                    velocity=(direction * self.config.physics.split_boost).astype(np.float32),
                    mass=piece_mass,
                    split_cooldown=self.config.physics.split_cooldown_steps,
                    merge_cooldown=self.config.physics.merge_cooldown_steps,
                )
            )

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
            weighted_boost = sum(cell.boost_velocity * cell.mass for cell in cells) / max(total_mass, 1e-6)

            merged = Cell(
                cell_id=self._new_cell_id(),
                agent_id=agent_id,
                position=weighted_position.astype(np.float32),
                velocity=weighted_velocity.astype(np.float32),
                mass=total_mass,
                split_cooldown=0,
                merge_cooldown=0,
                eject_cooldown=0,
                boost_velocity=weighted_boost.astype(np.float32),
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

    def _respawn_viruses(self, force_full: bool) -> None:
        if not self.config.viruses.enabled:
            return
        target = self.target_virus_count
        if target <= 0:
            return
        to_add = max(0, target - len(self.viruses)) if force_full else min(1, max(0, target - len(self.viruses)))
        for _ in range(to_add):
            self.viruses.append(
                Virus(
                    virus_id=self._new_virus_id(),
                    position=self._sample_open_position(
                        margin=float(self.config.viruses.spawn_margin),
                        min_dist=36.0,
                    ),
                    mass=float(self.config.viruses.mass),
                )
            )

    def _sample_open_position(self, margin: float, min_dist: float) -> np.ndarray:
        existing: list[np.ndarray] = []
        for cells in self.agents.values():
            existing.extend(cell.position for cell in cells)
        existing.extend(pellet.position for pellet in self.pellets[:64])
        existing.extend(virus.position for virus in self.viruses)
        low = max(0.0, margin)
        high = max(low + 1.0, self.map_size - margin)
        for _ in range(200):
            candidate = self.rng.uniform(low=low, high=high, size=(2,)).astype(np.float32)
            if all(float(np.sum((candidate - point) ** 2)) >= min_dist * min_dist for point in existing):
                return candidate
        return self.rng.uniform(0.0, self.map_size, size=(2,)).astype(np.float32)

    def _apply_mass_decay(self, dt: float) -> None:
        if not self.config.mass_decay.enabled or self.config.mass_decay.per_second <= 0.0:
            return
        decay = max(0.0, 1.0 - float(self.config.mass_decay.per_second) * max(0.0, dt))
        min_mass = float(self.config.mass_decay.min_mass)
        for cells in self.agents.values():
            for cell in cells:
                cell.mass = max(min_mass, cell.mass * decay)

    def _respawn_eliminated_agents(self, elimination_pairs: list[tuple[str, str]]) -> set[str]:
        if not self.config.simulation.continuing_respawn:
            return set()
        victims = {victim for _, victim in elimination_pairs}
        respawned: set[str] = set()
        for agent_id in victims:
            if self.agents.get(agent_id):
                continue
            self.agents[agent_id] = [
                Cell(
                    cell_id=self._new_cell_id(),
                    agent_id=agent_id,
                    position=self._sample_open_position(margin=20.0, min_dist=48.0),
                    velocity=np.zeros(2, dtype=np.float32),
                    mass=float(self.config.simulation.respawn_mass),
                )
            ]
            self.snapshots[agent_id].respawns += 1
            respawned.add(agent_id)
        if respawned:
            self._sync_prev_cell_positions()
        return respawned

    def _compute_rewards_and_info(
        self,
        elimination_pairs: list[tuple[str, str]],
        virus_splits: dict[str, int],
        respawned_agents: set[str],
    ) -> tuple[dict[str, float], dict[str, bool], dict[str, dict[str, Any]]]:
        elimination_by_agent: dict[str, int] = defaultdict(int)
        eliminated_agents: set[str] = set()
        for killer, victim in elimination_pairs:
            elimination_by_agent[killer] += 1
            eliminated_agents.add(victim)

        rewards: dict[str, float] = {}
        infos: dict[str, dict[str, Any]] = {}

        alive_count = len(self.alive_agents)
        if self.config.simulation.continuing_respawn:
            global_done = self.step_count >= self.config.max_steps
        else:
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

            behavior_breakdown = self._compute_behavior_reward_breakdown(agent_id)
            reward = (
                self.config.rewards.mass_gain_scale * delta_mass
                + self.config.rewards.time_penalty
                + elimination_by_agent.get(agent_id, 0) * self.config.rewards.elimination_bonus
            )
            if agent_id in eliminated_agents:
                reward += self.config.rewards.death_penalty
                reward += self.config.reward_terms.respawn_penalty
            if alive and time_frac >= self.config.rewards.survival_bonus_start_frac:
                reward += self.config.rewards.survival_bonus
            if virus_splits.get(agent_id, 0):
                behavior_breakdown["virus_split"] = (
                    virus_splits[agent_id] * self.config.reward_terms.virus_split_bonus
                )
            if self.step_split_attempts.get(agent_id, 0):
                behavior_breakdown["split_attempt"] = (
                    self.step_split_attempts[agent_id] * self.config.reward_terms.split_attempt_penalty
                )
            if self.step_unsafe_splits.get(agent_id, 0):
                behavior_breakdown["unsafe_split"] = (
                    self.step_unsafe_splits[agent_id] * self.config.reward_terms.unsafe_split_penalty
                )
            if self.step_useful_splits.get(agent_id, 0):
                behavior_breakdown["useful_split"] = (
                    self.step_useful_splits[agent_id] * self.config.reward_terms.useful_split_bonus
                )
            reward += sum(behavior_breakdown.values())

            rewards[agent_id] = float(reward)
            self.snapshots[agent_id].total_mass = total_mass
            self.snapshots[agent_id].alive = alive
            self.snapshots[agent_id].episode_return += reward
            self.snapshots[agent_id].eliminated_opponents += elimination_by_agent.get(agent_id, 0)
            self.snapshots[agent_id].virus_splits += virus_splits.get(agent_id, 0)
            self.snapshots[agent_id].split_attempts += self.step_split_attempts.get(agent_id, 0)
            self.snapshots[agent_id].successful_splits += self.step_successful_splits.get(agent_id, 0)
            self.snapshots[agent_id].unsafe_splits += self.step_unsafe_splits.get(agent_id, 0)
            self.snapshots[agent_id].useful_splits += self.step_useful_splits.get(agent_id, 0)
            self.snapshots[agent_id].last_reward_breakdown = {
                "mass_gain": self.config.rewards.mass_gain_scale * delta_mass,
                "time": self.config.rewards.time_penalty,
                "elimination": elimination_by_agent.get(agent_id, 0) * self.config.rewards.elimination_bonus,
                "death": self.config.rewards.death_penalty if agent_id in eliminated_agents else 0.0,
                **behavior_breakdown,
            }

            infos[agent_id] = {
                "alive": alive,
                "total_mass": total_mass,
                "delta_mass": delta_mass,
                "episode_return": self.snapshots[agent_id].episode_return,
                "eliminations": self.snapshots[agent_id].eliminated_opponents,
                "virus_splits": self.snapshots[agent_id].virus_splits,
                "split_attempts": self.snapshots[agent_id].split_attempts,
                "successful_splits": self.snapshots[agent_id].successful_splits,
                "unsafe_splits": self.snapshots[agent_id].unsafe_splits,
                "useful_splits": self.snapshots[agent_id].useful_splits,
                "respawned": agent_id in respawned_agents,
                "respawns": self.snapshots[agent_id].respawns,
                "reward_breakdown": dict(self.snapshots[agent_id].last_reward_breakdown),
                "recent_direction_counts": list(self.snapshots[agent_id].recent_direction_counts),
                "winner": winner,
            }

        dones: dict[str, bool] = {}
        for agent_id in self.agent_ids:
            agent_done = global_done or (
                (not self.config.simulation.continuing_respawn)
                and not self.snapshots[agent_id].alive
            )
            dones[agent_id] = agent_done
        dones["__all__"] = global_done

        infos["__global__"] = {
            "step": self.step_count,
            "alive_count": alive_count,
            "stage": self.stage,
            "map_size": self.map_size,
            "winner": winner,
            "auto_curriculum": self.auto_curriculum,
            "scenario": self.scenario_name,
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
        """Manual map scaling used by debugging and scenario tools."""
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
        self._center_cache = {}

    def _agent_total_mass(self, agent_id: str) -> float:
        return float(sum(cell.mass for cell in self.agents[agent_id]))

    def _agent_largest_cell(self, agent_id: str) -> Cell | None:
        cells = self.agents[agent_id]
        if not cells:
            return None
        return max(cells, key=lambda c: c.mass)

    def agent_center(self, agent_id: str) -> np.ndarray:
        """Public, cached accessor for an agent's mass-weighted center."""
        return self._agent_center(agent_id)

    def _agent_center(self, agent_id: str) -> np.ndarray:
        if self._center_cache_step != self.step_count:
            self._center_cache = {}
            self._center_cache_step = self.step_count
        cached = self._center_cache.get(agent_id)
        if cached is not None:
            return cached

        cells = self.agents[agent_id]
        if not cells:
            center = np.array([self.map_size * 0.5, self.map_size * 0.5], dtype=np.float32)
        else:
            masses = np.array([cell.mass for cell in cells], dtype=np.float32)
            stacked = np.stack([cell.position for cell in cells], axis=0)
            center = (stacked * masses[:, None]).sum(axis=0) / max(float(masses.sum()), 1e-6)
        self._center_cache[agent_id] = center
        return center

    def _nearest_relation(
        self,
        agent_id: str,
        *,
        want_threat: bool,
    ) -> tuple[str, np.ndarray, float, float] | None:
        center = self._agent_center(agent_id)
        own_mass = max(self._agent_total_mass(agent_id), 1e-6)
        candidates: list[tuple[str, np.ndarray, float, float]] = []
        for other_id in self.agent_ids:
            if other_id == agent_id or not self.agents[other_id]:
                continue
            other_center = self._agent_center(other_id)
            delta = other_center - center
            distance = float(np.linalg.norm(delta))
            ratio = self._agent_total_mass(other_id) / own_mass
            if want_threat and ratio >= self.config.physics.eat_mass_ratio:
                candidates.append((other_id, delta, distance, ratio))
            elif (not want_threat) and ratio <= (1.0 / max(self.config.physics.eat_mass_ratio, 1e-6)):
                candidates.append((other_id, delta, distance, ratio))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[2])

    def _compute_behavior_reward_breakdown(self, agent_id: str) -> dict[str, float]:
        breakdown: dict[str, float] = {}
        if not self.agents[agent_id]:
            self.snapshots[agent_id].nearest_threat_distance = None
            self.snapshots[agent_id].nearest_target_distance = None
            return breakdown

        threat = self._nearest_relation(agent_id, want_threat=True)
        previous_threat = self.snapshots[agent_id].nearest_threat_distance
        if threat is not None:
            if previous_threat is not None and self.config.reward_terms.threat_escape_scale:
                delta = max(0.0, threat[2] - previous_threat) / max(self.map_size, 1e-6)
                breakdown["threat_escape"] = delta * self.config.reward_terms.threat_escape_scale
            self.snapshots[agent_id].nearest_threat_distance = threat[2]
        else:
            self.snapshots[agent_id].nearest_threat_distance = None

        target = self._nearest_relation(agent_id, want_threat=False)
        previous_target = self.snapshots[agent_id].nearest_target_distance
        if target is not None:
            if previous_target is not None and self.config.reward_terms.target_pressure_scale:
                delta = max(0.0, previous_target - target[2]) / max(self.map_size, 1e-6)
                breakdown["target_pressure"] = delta * self.config.reward_terms.target_pressure_scale
            self.snapshots[agent_id].nearest_target_distance = target[2]
        else:
            self.snapshots[agent_id].nearest_target_distance = None

        center = self._agent_center(agent_id)
        corner_margin = self.map_size * 0.18
        in_corner = (
            (center[0] <= corner_margin or center[0] >= self.map_size - corner_margin)
            and (center[1] <= corner_margin or center[1] >= self.map_size - corner_margin)
        )
        if in_corner and self.config.reward_terms.corner_penalty:
            breakdown["corner"] = self.config.reward_terms.corner_penalty

        if self.config.reward_terms.survival_quality_scale:
            distance_from_center = float(
                np.linalg.norm(center - np.array([self.map_size * 0.5, self.map_size * 0.5], dtype=np.float32))
            )
            center_score = 1.0 - min(1.0, distance_from_center / max(self.map_size * 0.707, 1e-6))
            breakdown["survival_quality"] = center_score * self.config.reward_terms.survival_quality_scale
        return breakdown

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
            observed_velocity = largest.total_velocity()
            obs[cursor : cursor + 8] = np.array(
                [
                    largest.position[0] / self.map_size,
                    largest.position[1] / self.map_size,
                    observed_velocity[0] / speed_norm,
                    observed_velocity[1] / speed_norm,
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
        cursor += 4

        if self.config.observation_features.enabled:
            if self.config.observation_features.include_threats:
                cursor = self._write_relation_observation(
                    obs=obs,
                    cursor=cursor,
                    agent_id=agent_id,
                    center=center,
                    want_threat=True,
                )
                cursor = self._write_relation_observation(
                    obs=obs,
                    cursor=cursor,
                    agent_id=agent_id,
                    center=center,
                    want_threat=False,
                )
            if self.config.observation_features.include_viruses:
                cursor = self._write_virus_observation(obs=obs, cursor=cursor, center=center)
            if self.config.observation_features.include_eject_state:
                largest = self._agent_largest_cell(agent_id)
                split_ready = 0.0
                eject_ready = 0.0
                if largest is not None:
                    split_ready = float(
                        largest.mass >= self.config.physics.min_split_mass
                        and largest.split_cooldown <= 0
                    )
                    eject_ready = float(
                        self.config.physics.enable_eject_mechanic
                        and largest.eject_cooldown <= 0
                        and largest.mass > self.config.physics.eject_mass_amount + 1.0
                    )
                obs[cursor : cursor + 2] = np.array([split_ready, eject_ready], dtype=np.float32)
        return obs

    def _write_relation_observation(
        self,
        *,
        obs: np.ndarray,
        cursor: int,
        agent_id: str,
        center: np.ndarray,
        want_threat: bool,
    ) -> int:
        relation = self._nearest_relation(agent_id, want_threat=want_threat)
        if relation is None:
            return cursor + 4
        _, delta, distance, mass_ratio = relation
        obs[cursor : cursor + 4] = np.array(
            [
                delta[0] / max(self.map_size, 1e-6),
                delta[1] / max(self.map_size, 1e-6),
                distance / max(self.map_size, 1e-6),
                mass_ratio,
            ],
            dtype=np.float32,
        )
        return cursor + 4

    def _write_virus_observation(self, *, obs: np.ndarray, cursor: int, center: np.ndarray) -> int:
        if not self.viruses:
            return cursor + 3
        nearest = min(self.viruses, key=lambda virus: float(np.sum((virus.position - center) ** 2)))
        delta = nearest.position - center
        distance = float(np.linalg.norm(delta))
        obs[cursor : cursor + 3] = np.array(
            [
                delta[0] / max(self.map_size, 1e-6),
                delta[1] / max(self.map_size, 1e-6),
                distance / max(self.map_size, 1e-6),
            ],
            dtype=np.float32,
        )
        return cursor + 3
