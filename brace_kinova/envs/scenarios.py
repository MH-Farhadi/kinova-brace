"""Scenario configurations for curriculum-based training difficulty levels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScenarioConfig:
    """Defines a single difficulty scenario for the reach-and-grasp environment."""

    name: str
    n_objects: int = 1
    n_obstacles: int = 0

    object_x_range: tuple[float, float] = (0.25, 0.55)
    object_y_range: tuple[float, float] = (-0.25, 0.40)
    min_object_separation: float = 0.10

    obstacle_x_range: tuple[float, float] = (0.25, 0.55)
    obstacle_y_range: tuple[float, float] = (-0.25, 0.40)
    min_obstacle_object_separation: float = 0.08
    min_obstacle_separation: float = 0.08

    tight_gaps: bool = False
    gap_width_range: tuple[float, float] = (0.06, 0.10)

    min_episodes: int = 100
    success_threshold: float = 0.80
    max_collision_rate: Optional[float] = None
    plateau_window: Optional[int] = None


SCENARIOS: dict[str, ScenarioConfig] = {
    "basic_reaching": ScenarioConfig(
        name="basic_reaching",
        n_objects=1,
        n_obstacles=0,
        min_episodes=100,
        success_threshold=0.80,
    ),
    "collision_avoidance": ScenarioConfig(
        name="collision_avoidance",
        n_objects=1,
        n_obstacles=3,
        min_episodes=200,
        success_threshold=0.75,
        max_collision_rate=0.15,
    ),
    "challenging_obstacles": ScenarioConfig(
        name="challenging_obstacles",
        n_objects=1,
        n_obstacles=4,
        tight_gaps=True,
        gap_width_range=(0.06, 0.10),
        min_episodes=300,
        success_threshold=0.70,
    ),
    "goal_ambiguity": ScenarioConfig(
        name="goal_ambiguity",
        n_objects=3,
        n_obstacles=3,
        min_episodes=400,
        success_threshold=0.65,
    ),
    "full_complexity": ScenarioConfig(
        name="full_complexity",
        n_objects=3,
        n_obstacles=4,
        tight_gaps=True,
        plateau_window=200,
    ),
}


def get_scenario(name: str) -> ScenarioConfig:
    """Retrieve a named scenario configuration."""
    if name not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{name}'. Available: {list(SCENARIOS.keys())}"
        )
    return SCENARIOS[name]
