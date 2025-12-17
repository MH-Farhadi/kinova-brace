from __future__ import annotations

from typing import List, Tuple


def generate_waypoints_base(
    *,
    target_pos_b: Tuple[float, float, float],
    pregrasp_offset_m: float,
    grasp_depth_m: float,
    lift_height_m: float,
) -> List[Tuple[float, float, float]]:
    """Generate simple position-only waypoints in the robot base frame.

    This mirrors the basic strategy used by the scripted planner:
    - pregrasp: above the target by pregrasp_offset_m
    - grasp: near the target by grasp_depth_m
    - lift: above the target by lift_height_m
    """

    x, y, z = float(target_pos_b[0]), float(target_pos_b[1]), float(target_pos_b[2])
    pre = (x, y, z + float(pregrasp_offset_m))
    grasp = (x, y, z + float(grasp_depth_m))
    lift = (x, y, z + float(lift_height_m))
    return [pre, grasp, lift]


