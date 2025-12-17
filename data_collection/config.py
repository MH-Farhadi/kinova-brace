from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class EpisodeConfig:
    """Episode-loop settings."""

    num_episodes: int = 10
    max_steps_per_phase: int = 2000


@dataclass
class TaskConfig:
    """Task-specific knobs for reach-to-grasp."""

    target_label: Optional[str] = None
    pregrasp_offset_m: float = 0.10
    grasp_depth_m: float = 0.00
    lift_height_m: float = 0.15


@dataclass
class PlannerConfig:
    """Planner settings (scripted/rmpflow/curobo/...)."""

    type: str = "scripted"
    linear_speed_mps: float = 0.7
    tolerance_m: float = 0.005


@dataclass
class ObjectsConfig:
    """Object spawning configuration."""

    dataset_dirs: List[str]
    num_objects: int
    spawn_min_xyz: Tuple[float, float, float]
    spawn_max_xyz: Tuple[float, float, float]


@dataclass
class LoggingConfig:
    logs_root: str = "logs/data_collection"


@dataclass
class RunConfig:
    """Top-level configuration passed to episode runners."""

    episode: EpisodeConfig
    task: TaskConfig
    planner: PlannerConfig
    objects: ObjectsConfig
    logging: LoggingConfig


