"""JSON frame serialization for the browser runtime."""

from __future__ import annotations

from pathlib import Path
import csv
from typing import Any

import numpy as np

from agario_rl.env.world import AgarioWorld


AGENT_COLORS: tuple[str, ...] = (
    "#35c3ff",
    "#ff5f7e",
    "#54d36b",
    "#ffd166",
    "#a78bfa",
    "#f97316",
    "#22d3ee",
    "#fb7185",
)


def _xy(value: np.ndarray) -> dict[str, float]:
    return {"x": round(float(value[0]), 3), "y": round(float(value[1]), 3)}


def _agent_center(world: AgarioWorld, agent_id: str) -> np.ndarray:
    cells = world.agents[agent_id]
    if not cells:
        return np.array([world.map_size * 0.5, world.map_size * 0.5], dtype=np.float32)
    masses = np.array([cell.mass for cell in cells], dtype=np.float32)
    positions = np.stack([cell.position for cell in cells], axis=0)
    return (positions * masses[:, None]).sum(axis=0) / max(float(masses.sum()), 1e-6)


def _agent_mass(world: AgarioWorld, agent_id: str) -> float:
    return float(sum(cell.mass for cell in world.agents[agent_id]))


def _nearest_relation(
    world: AgarioWorld,
    agent_id: str,
    *,
    want_threat: bool,
) -> dict[str, Any] | None:
    own_mass = max(_agent_mass(world, agent_id), 1e-6)
    center = _agent_center(world, agent_id)
    candidates: list[tuple[str, np.ndarray, float, float]] = []
    for other_id in world.agent_ids:
        if other_id == agent_id or not world.agents[other_id]:
            continue
        other_center = _agent_center(world, other_id)
        delta = other_center - center
        distance = float(np.linalg.norm(delta))
        mass_ratio = _agent_mass(world, other_id) / own_mass
        if want_threat and mass_ratio >= 1.12:
            candidates.append((other_id, delta, distance, mass_ratio))
        elif (not want_threat) and mass_ratio <= 0.85:
            candidates.append((other_id, delta, distance, mass_ratio))
    if not candidates:
        return None
    other_id, delta, distance, mass_ratio = min(candidates, key=lambda item: item[2])
    return {
        "id": other_id,
        "delta": _xy(delta),
        "distance": round(distance, 3),
        "massRatio": round(mass_ratio, 3),
    }


def latest_training_metrics(metrics_csv: Path) -> dict[str, float]:
    """Read the most recent training metrics row without rewriting logs."""
    if not metrics_csv.exists() or metrics_csv.stat().st_size == 0:
        return {}
    with metrics_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    latest = rows[-1]
    metrics: dict[str, float] = {}
    for key, value in latest.items():
        if value in (None, ""):
            continue
        try:
            metrics[key] = float(value)
        except ValueError:
            continue
    return metrics


def build_browser_frame(
    *,
    world: AgarioWorld,
    infos: dict[str, dict[str, Any]],
    mode: str,
    tick: int,
    player_id: str | None,
    policy_source: str,
    checkpoint: str,
    metrics: dict[str, float],
    fps: float,
) -> dict[str, Any]:
    """Build the compact JSON shape consumed by the TypeScript renderer."""
    agents: list[dict[str, Any]] = []
    for index, agent_id in enumerate(world.agent_ids):
        cells = world.agents[agent_id]
        info = infos.get(agent_id, {})
        reward_breakdown = info.get("reward_breakdown", {})
        agents.append(
            {
                "id": agent_id,
                "name": "You" if agent_id == player_id else f"Agent {index + 1}",
                "color": AGENT_COLORS[index % len(AGENT_COLORS)],
                "alive": bool(cells),
                "totalMass": round(_agent_mass(world, agent_id), 3),
                "center": _xy(_agent_center(world, agent_id)),
                "cells": [
                    {
                        "id": cell.cell_id,
                        "x": round(float(cell.position[0]), 3),
                        "y": round(float(cell.position[1]), 3),
                        "vx": round(float(cell.velocity[0]), 3),
                        "vy": round(float(cell.velocity[1]), 3),
                        "mass": round(float(cell.mass), 3),
                        "radius": round(float(cell.radius(world.config.physics.radius_scale)), 3),
                    }
                    for cell in cells
                ],
                "rewardBreakdown": {
                    key: round(float(value), 4)
                    for key, value in reward_breakdown.items()
                    if isinstance(value, int | float)
                },
                "split": {
                    "attempts": int(info.get("split_attempts", 0)),
                    "successful": int(info.get("successful_splits", 0)),
                    "unsafe": int(info.get("unsafe_splits", 0)),
                    "useful": int(info.get("useful_splits", 0)),
                },
                "threat": _nearest_relation(world, agent_id, want_threat=True),
                "target": _nearest_relation(world, agent_id, want_threat=False),
            }
        )

    leaderboard = sorted(
        (
            {"id": agent["id"], "name": agent["name"], "mass": agent["totalMass"], "color": agent["color"]}
            for agent in agents
        ),
        key=lambda item: float(item["mass"]),
        reverse=True,
    )

    return {
        "type": "frame",
        "mode": mode,
        "tick": tick,
        "fps": round(fps, 2),
        "scenario": world.scenario_name,
        "mapSize": round(float(world.map_size), 3),
        "playerId": player_id,
        "agents": agents,
        "leaderboard": leaderboard,
        "pellets": [
            {"id": pellet.pellet_id, "x": round(float(pellet.position[0]), 3), "y": round(float(pellet.position[1]), 3), "mass": round(float(pellet.mass), 3)}
            for pellet in world.pellets
        ],
        "viruses": [
            {
                "id": virus.virus_id,
                "x": round(float(virus.position[0]), 3),
                "y": round(float(virus.position[1]), 3),
                "mass": round(float(virus.mass), 3),
                "radius": round(float(virus.radius(world.config.physics.radius_scale)), 3),
                "fed": int(virus.fed_count),
            }
            for virus in world.viruses
        ],
        "ejected": [
            {
                "id": ejected.ejected_id,
                "ownerId": ejected.owner_id,
                "x": round(float(ejected.position[0]), 3),
                "y": round(float(ejected.position[1]), 3),
                "mass": round(float(ejected.mass), 3),
            }
            for ejected in world.ejected_masses
        ],
        "training": {
            "policySource": policy_source,
            "checkpoint": checkpoint,
            "updateCount": int(metrics.get("update", metrics.get("update_count", 0))),
            "metrics": metrics,
            "humanReadiness": {
                "splitSafety": sum(agent["split"]["useful"] for agent in agents)
                - sum(agent["split"]["unsafe"] for agent in agents),
                "unsafeSplits": sum(agent["split"]["unsafe"] for agent in agents),
                "usefulSplits": sum(agent["split"]["useful"] for agent in agents),
                "finalMassLeader": leaderboard[0]["mass"] if leaderboard else 0.0,
            },
        },
        "controls": {
            "move": "Mouse steers your cell",
            "split": "Space",
            "reset": "R",
        },
    }
