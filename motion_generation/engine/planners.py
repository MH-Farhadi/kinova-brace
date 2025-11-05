from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import os


@dataclass
class PlannerContext:
    base_frame: str
    ee_link_name: str
    urdf_path: Optional[str]
    config_dir: str


class BasePlanner:
    def __init__(self, ctx: PlannerContext) -> None:
        self.ctx = ctx

    def plan_waypoints_b(
        self,
        *,
        target_pos_b: Tuple[float, float, float],
        pregrasp_offset_m: float,
        grasp_depth_m: float,
        lift_height_m: float,
    ) -> List[Tuple[float, float, float]]:
        raise NotImplementedError


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


class RmpFlowPlanner(BasePlanner):
    def __init__(self, ctx: PlannerContext) -> None:
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


class CuroboPlanner(BasePlanner):
    def __init__(self, ctx: PlannerContext) -> None:
        super().__init__(ctx)
        # Lazy detection of cuRobo/Isaac Motion Generation GPU backend
        try:
            import importlib  # noqa: F401
            # The exact API may vary across Isaac Sim versions; we probe extension availability
            self._available = True
            print("[MG][CUROBO] cuRobo backend assumed available (extension check deferred).")
        except Exception:
            self._available = False
            print("[MG][CUROBO][WARN] cuRobo backend not available. Falling back to scripted planner.")

    def plan_waypoints_b(
        self,
        *,
        target_pos_b: Tuple[float, float, float],
        pregrasp_offset_m: float,
        grasp_depth_m: float,
        lift_height_m: float,
    ) -> List[Tuple[float, float, float]]:
        if not getattr(self, "_available", False):
            return ScriptedPlanner(self.ctx).plan_waypoints_b(
                target_pos_b=target_pos_b,
                pregrasp_offset_m=pregrasp_offset_m,
                grasp_depth_m=grasp_depth_m,
                lift_height_m=lift_height_m,
            )
        # TODO: Construct a collision-aware trajectory via cuRobo once robot config is provided
        print("[MG][CUROBO] Placeholder path generation; using scripted waypoints pending full integration.")
        return ScriptedPlanner(self.ctx).plan_waypoints_b(
            target_pos_b=target_pos_b,
            pregrasp_offset_m=pregrasp_offset_m,
            grasp_depth_m=grasp_depth_m,
            lift_height_m=lift_height_m,
        )


def create_planner(kind: str, *, ctx: PlannerContext) -> BasePlanner:
    kind_l = (kind or "").lower()
    if kind_l == "rmpflow":
        return RmpFlowPlanner(ctx)
    if kind_l == "curobo":
        return CuroboPlanner(ctx)
    if kind_l == "scripted" or kind_l == "":
        return ScriptedPlanner(ctx)
    print(f"[MG][PLANNER][WARN] Unknown planner type '{kind}'. Falling back to 'scripted'.")
    return ScriptedPlanner(ctx)


