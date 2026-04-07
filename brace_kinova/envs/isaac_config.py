"""Configuration dataclasses for the Isaac Sim BRACE environment.

Defines scene layout, robot parameters, workspace bounds, and episode
settings for 2D planar reach-and-grasp training with the Kinova Jaco2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class IsaacBraceSceneConfig:
    """Scene layout for the Kinova Jaco2 tabletop RL environment."""

    table_height: float = 0.80
    robot_base_height: float = 0.80

    table_usd_path: str = ""  # Resolved at runtime via ISAAC_NUCLEUS_DIR
    table_scale: Tuple[float, float, float] = (1.5, 2.0, 1.0)
    table_translation: Tuple[float, float, float] = (0.0, 0.0, 0.80)

    robot_prim_path: str = "/World/Origin1/Robot"
    objects_prim_root: str = "/World/Origin1/BraceObjects"

    robot_default_joint_pos: Optional[dict] = None

    goal_box_size: Tuple[float, float, float] = (0.04, 0.04, 0.04)
    goal_color: Tuple[float, float, float] = (0.1, 0.8, 0.1)

    obstacle_box_size: Tuple[float, float, float] = (0.05, 0.05, 0.05)
    obstacle_color: Tuple[float, float, float] = (0.8, 0.1, 0.1)

    object_spawn_z: float = 0.825


@dataclass
class IsaacBraceEnvConfig:
    """Full configuration for the Isaac BRACE RL environment.

    Observation and action semantics intentionally match
    ``brace_kinova.envs.reach_grasp_env.ReachGraspEnv`` so that all
    BRACE models (expert, belief, arbitration) transfer without changes.
    """

    # Robot link / joint identifiers
    ee_link_name: str = "j2n6s300_end_effector"
    arm_joint_regex: str = "j2n6s300_joint_[1-6]"
    gripper_joint_regex: str = ".*_joint_finger_.*|.*_joint_finger_tip_.*"

    # Workspace bounds (robot base frame, metres)
    x_min: float = 0.20
    x_max: float = 0.60
    y_min: float = -0.30
    y_max: float = 0.45
    z_fixed: float = 0.03  # Fixed EE height in base frame (≈ table surface)

    # End effector
    max_velocity: float = 0.15
    ee_initial_x: float = 0.30
    ee_initial_y: float = 0.0

    # Object / obstacle limits (max across all curriculum stages)
    max_n_objects: int = 3
    max_n_obstacles: int = 4
    grasp_threshold: float = 0.04
    collision_threshold: float = 0.045  # ee_radius + obstacle_radius

    # Spawn ranges (base frame XY)
    object_x_range: Tuple[float, float] = (0.25, 0.55)
    object_y_range: Tuple[float, float] = (-0.25, 0.40)
    min_object_separation: float = 0.10
    min_obstacle_object_separation: float = 0.08

    # Episode
    max_steps: int = 200
    control_dt: float = 0.05
    physics_dt: float = 1.0 / 240.0
    decimation: int = 12  # physics_dt * decimation ≈ control_dt
    stabilize_steps: int = 60

    # IK
    ik_method: str = "dls"

    # Gripper positions
    gripper_open_pos: float = 0.0
    gripper_close_pos: float = 1.2

    # Reward weights (same semantics as ExpertRewardConfig)
    reward_progress_weight: float = 4.0
    reward_heading_penalty: float = 0.6
    reward_obstacle_penalty: float = 1.2
    reward_d_safe: float = 0.06
    reward_near_goal_weight: float = 0.6
    reward_near_goal_scale: float = 0.12
    reward_time_penalty: float = 0.002
    reward_stagnation_penalty: float = 0.01
    reward_goal_bonus: float = 5.0
    reward_collision_penalty: float = 10.0

    # Scene
    scene: IsaacBraceSceneConfig = field(default_factory=IsaacBraceSceneConfig)

    seed: int = 42
