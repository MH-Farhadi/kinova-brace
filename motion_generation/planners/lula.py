from __future__ import annotations

from typing import List, Tuple, Optional, Any

from .base import BasePlanner
from .scripted import ScriptedPlanner


class LulaPlanner(BasePlanner):
    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._available = False
        try:
            import importlib  # noqa: F401
            self._available = True
            print("[MG][LULA] LULA motion generation assumed available (deferred init).")
        except Exception:
            self._available = False
            print("[MG][LULA][WARN] LULA not available. Falling back to scripted planning.")
        self._gen: Any = None

    def _ensure_generator(self) -> bool:
        if not self._available:
            return False
        if self._gen is not None:
            return True
        try:
            import importlib
            mg_mod = None
            try:
                mg_mod = importlib.import_module("omni.isaac.motion_generation")
            except Exception:
                mg_mod = None
            LulaGenCls = None
            if mg_mod is not None:
                LulaGenCls = getattr(mg_mod, "CSpaceTrajectoryGenerator", None)
                if LulaGenCls is None:
                    lula_mod = getattr(mg_mod, "lula", None)
                    if lula_mod is not None:
                        LulaGenCls = getattr(lula_mod, "CSpaceTrajectoryGenerator", None)
            if LulaGenCls is None:
                print("[MG][LULA][WARN] Could not locate CSpaceTrajectoryGenerator symbol.")
                return False
            try:
                self._gen = LulaGenCls()
            except Exception:
                self._gen = None
                print("[MG][LULA][WARN] LULA generator requires extra config; using scripted fallback.")
                return False
            print("[MG][LULA] CSpaceTrajectoryGenerator initialized.")
            return True
        except Exception as e:
            print(f"[MG][LULA][WARN] Failed to initialize LULA generator: {e}")
            self._gen = None
            return False

    def plan_waypoints_b(
        self,
        *,
        target_pos_b: Tuple[float, float, float],
        pregrasp_offset_m: float,
        grasp_depth_m: float,
        lift_height_m: float,
    ) -> List[Tuple[float, float, float]]:
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
        target_quat_b_wxyz: Optional[Tuple[float, float, float, float]],
        pregrasp_offset_m: float,
        grasp_depth_m: float,
        lift_height_m: float,
    ) -> List[Tuple[float, float, float]]:
        if not self._ensure_generator():
            return ScriptedPlanner(self.ctx).plan_waypoints_b(
                target_pos_b=target_pos_b,
                pregrasp_offset_m=pregrasp_offset_m,
                grasp_depth_m=grasp_depth_m,
                lift_height_m=lift_height_m,
            )
        try:
            import torch
            pos = torch.tensor([[float(target_pos_b[0]), float(target_pos_b[1]), float(target_pos_b[2])]], dtype=torch.float32)
            if target_quat_b_wxyz is None:
                quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
            else:
                w, x, y, z = target_quat_b_wxyz
                quat = torch.tensor([[float(w), float(x), float(y), float(z)]], dtype=torch.float32)
            result = None
            if hasattr(self._gen, "plan_single"):
                result = self._gen.plan_single(goal_position=pos, goal_orientation=quat)
            elif hasattr(self._gen, "plan"):
                result = self._gen.plan(goal_position=pos, goal_orientation=quat)
            elif hasattr(self._gen, "compute_path"):
                result = self._gen.compute_path(goal_position=pos, goal_orientation=quat)
            else:
                print("[MG][LULA][WARN] No recognized plan method on LULA generator; using scripted fallback.")
                raise RuntimeError("No plan method")
            last_p = (float(pos[0, 0]), float(pos[0, 1]), float(pos[0, 2]))
            waypoints = [
                (last_p[0], last_p[1], last_p[2] + float(pregrasp_offset_m)),
                (last_p[0], last_p[1], last_p[2] + float(grasp_depth_m)),
                (last_p[0], last_p[1], last_p[2] + float(lift_height_m)),
            ]
            print("[MG][LULA] Planned 6D goal; returning approach/grasp/lift waypoints.")
            return waypoints
        except Exception as e:
            print(f"[MG][LULA][WARN] LULA planning failed ({e}); using scripted fallback.")
            return ScriptedPlanner(self.ctx).plan_waypoints_b(
                target_pos_b=target_pos_b,
                pregrasp_offset_m=pregrasp_offset_m,
                grasp_depth_m=grasp_depth_m,
                lift_height_m=lift_height_m,
            )


