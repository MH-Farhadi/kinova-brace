from __future__ import annotations

from typing import List, Tuple


def generate_waypoints_base(
    *,
    target_pos_b: Tuple[float, float, float],
    pregrasp_offset_m: float = 0.10,
    grasp_depth_m: float = 0.00,
    lift_height_m: float = 0.15,
) -> List[Tuple[float, float, float]]:
    """Return [pregrasp, grasp, lift] waypoints in base frame.

    - pregrasp: hover above target by pregrasp_offset_m (along +Z of base)
    - grasp: descend to target (minus grasp_depth_m)
    - lift: raise by lift_height_m after grasp
    """
    x, y, z = float(target_pos_b[0]), float(target_pos_b[1]), float(target_pos_b[2])
    pre = (x, y, z + float(pregrasp_offset_m))
    grasp = (x, y, z + float(grasp_depth_m))
    lift = (x, y, z + float(lift_height_m))
    return [pre, grasp, lift]


