from __future__ import annotations

from typing import List, Tuple
import os

from .base import BasePlanner
from .scripted import ScriptedPlanner


class RmpFlowPlanner(BasePlanner):
    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        # Validate presence of motion generation config files
        cfg_dir = self.ctx.config_dir
        expected = [
            os.path.join(cfg_dir, "j2n6s300.srdf"),
            os.path.join(cfg_dir, "kinematics.yaml"),
            os.path.join(cfg_dir, "joint_limits.yaml"),
            os.path.join(cfg_dir, "ompl_planning.yaml"),
        ]
        missing = [p for p in expected if not os.path.exists(p)]
        if missing:
            print(f"[MG][RMP][WARN] Missing required config files: {missing}. Falling back to scripted planner.")
            self._fallback = ScriptedPlanner(ctx)
        else:
            self._fallback = None
            print(f"[MG][RMP] Found MoveIt configs in: {cfg_dir}. Using scripted path generation; RMP collision checks TBD.")

    def plan_waypoints_b(
        self,
        *,
        target_pos_b: Tuple[float, float, float],
        pregrasp_offset_m: float,
        grasp_depth_m: float,
        lift_height_m: float,
    ) -> List[Tuple[float, float, float]]:
        if self._fallback is not None:
            return self._fallback.plan_waypoints_b(
                target_pos_b=target_pos_b,
                pregrasp_offset_m=pregrasp_offset_m,
                grasp_depth_m=grasp_depth_m,
                lift_height_m=lift_height_m,
            )
        # TODO: Integrate LULA RMP collision-aware path sampling when Kinova descriptor and RMP config are available
        return ScriptedPlanner(self.ctx).plan_waypoints_b(
            target_pos_b=target_pos_b,
            pregrasp_offset_m=pregrasp_offset_m,
            grasp_depth_m=grasp_depth_m,
            lift_height_m=lift_height_m,
        )

    def plan_to_pose_b(
        self,
        *,
        target_pos_b: Tuple[float, float, float],
        target_quat_b_wxyz: Tuple[float, float, float, float] | None,
        pregrasp_offset_m: float,
        grasp_depth_m: float,
        lift_height_m: float,
    ) -> List[Tuple[float, float, float]]:
        # Until full RMP integration is added, use position-only fallback
        return self.plan_waypoints_b(
            target_pos_b=target_pos_b,
            pregrasp_offset_m=pregrasp_offset_m,
            grasp_depth_m=grasp_depth_m,
            lift_height_m=lift_height_m,
        )


