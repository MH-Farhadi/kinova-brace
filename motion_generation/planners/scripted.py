from __future__ import annotations

from typing import List, Tuple

from .base import BasePlanner


class ScriptedPlanner(BasePlanner):
    def plan_waypoints_b(
        self,
        *,
        target_pos_b: Tuple[float, float, float],
        pregrasp_offset_m: float,
        grasp_depth_m: float,
        lift_height_m: float,
    ) -> List[Tuple[float, float, float]]:
        x, y, z = float(target_pos_b[0]), float(target_pos_b[1]), float(target_pos_b[2])
        pre = (x, y, z + float(pregrasp_offset_m))
        grasp = (x, y, z + float(grasp_depth_m))
        lift = (x, y, z + float(lift_height_m))
        return [pre, grasp, lift]


