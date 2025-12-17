from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple


# Compact record types for logging and LM inputs


@dataclass
class Pose:
    position_m: Tuple[float, float, float]
    orientation_wxyz: Tuple[float, float, float, float]


@dataclass
class RobotState:
    t_ms: int
    ee_pose: Pose
    ee_linear_vel_mps: Tuple[float, float, float]
    ee_angular_vel_rps: Tuple[float, float, float]
    gripper_open_frac: float
    mode: str


@dataclass
class UserInput:
    t_ms: int
    cartesian_vel_cmd: Tuple[float, float, float, float, float, float]
    speed_scale: float
    mode: str


@dataclass
class DetectedObject:
    id: str
    label: str
    color: Optional[str]
    pose: Pose
    bbox_xywh: Optional[Tuple[float, float, float, float]]
    confidence: float


@dataclass
class Event:
    t_ms: int
    type: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class RollingWindow:
    window_ms: int
    robot_states: List[RobotState]
    user_inputs: List[UserInput]
    objects: List[DetectedObject]
    events: List[Event]


def to_json(obj: Any) -> Dict[str, Any]:
    """Convert dataclass (possibly nested) to a plain JSON-serializable dict."""
    return asdict(obj)


