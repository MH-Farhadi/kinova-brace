from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class PlannerContext:
    base_frame: str
    ee_link_name: str
    urdf_path: Optional[str]
    config_dir: str


class BasePlanner:
    def __init__(self, ctx: PlannerContext) -> None:
        self.ctx = ctx

    def plan_to_pose_b(
        self,
        *,
        target_pos_b: Tuple[float, float, float],
        target_quat_b_wxyz: Optional[Tuple[float, float, float, float]],
        pregrasp_offset_m: float,
        grasp_depth_m: float,
        lift_height_m: float,
    ) -> List[Tuple[float, float, float]]:
        """Optional 6D planning interface. Default falls back to position-only waypoints."""
        return self.plan_waypoints_b(
            target_pos_b=target_pos_b,
            pregrasp_offset_m=pregrasp_offset_m,
            grasp_depth_m=grasp_depth_m,
            lift_height_m=lift_height_m,
        )

    def plan_waypoints_b(
        self,
        *,
        target_pos_b: Tuple[float, float, float],
        pregrasp_offset_m: float,
        grasp_depth_m: float,
        lift_height_m: float,
    ) -> List[Tuple[float, float, float]]:
        raise NotImplementedError


