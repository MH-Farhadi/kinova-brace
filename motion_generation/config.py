from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any


@dataclass
class EpisodeConfig:
    num_episodes: int = 10
    max_steps_per_phase: int = 400
    timeout_s: float = 30.0
    settle_time_s: float = 1.0


@dataclass
class TaskConfig:
    target_label: Optional[str] = None  # if None, pick first object
    pregrasp_offset_m: float = 0.10
    grasp_depth_m: float = 0.00
    lift_height_m: float = 0.15
    approach_axis_b: Tuple[float, float, float] = (0.0, 0.0, 1.0)  # along +Z of base


@dataclass
class PlannerConfig:
    type: str = "scripted"  # placeholder for future extensions
    linear_speed_mps: float = 0.20
    rot_speed_rps: float = 0.5
    tolerance_m: float = 0.005


@dataclass
class GraspConfig:
    type: str = "aabb"  # "aabb" | "replicator"
    # Replicator-specific options:
    rep_gripper_prim_path: Optional[str] = None
    rep_config_yaml_path: Optional[str] = None
    rep_sampler_config: Optional[Dict[str, Any]] = None
    rep_max_candidates: int = 16


@dataclass
class ObjectsConfig:
    dataset_dirs: List[str] = field(default_factory=list)
    num_objects: int = 1
    min_distance_m: float = 0.08
    spawn_min_xyz: Tuple[float, float, float] = (0.30, -0.20, 0.02)
    spawn_max_xyz: Tuple[float, float, float] = (0.55, 0.20, 0.05)
    scale_min: Optional[float] = None
    scale_max: Optional[float] = None


@dataclass
class LoggingConfig:
    logs_root: str = "logs/assist"
    log_rate_hz: int = 10
    window_len_s: float = 2.0


@dataclass
class RunConfig:
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    grasp: GraspConfig = field(default_factory=GraspConfig)
    objects: ObjectsConfig = field(default_factory=ObjectsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


