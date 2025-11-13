from __future__ import annotations

from typing import Tuple, Optional
import importlib
import math

from .base import GraspPoseProvider


class AabbGraspPoseProvider(GraspPoseProvider):
    """Compute grasp pose using world AABB: top-center position; optional min-width yaw alignment."""

    def __init__(self, *, align_to_min_width: bool = True) -> None:
        self._align_to_min_width = align_to_min_width

    def _compute_world_aabb_top_center_and_xy_extents(self, *, prim_path: str) -> Tuple[Tuple[float, float, float], float, float]:
        """Return (pos_w_m, extent_x, extent_y) from world AABB."""
        try:
            UsdGeom = importlib.import_module("pxr.UsdGeom")
            omni_usd = importlib.import_module("omni.usd")  # type: ignore[attr-defined]
        except Exception as e:
            raise RuntimeError(f"[MG][GRASP] USD modules not available: {e}")

        stage = omni_usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"[MG][GRASP] Invalid prim path: {prim_path}")

        bbox_cache = UsdGeom.BBoxCache(0.0, ["default"], useExtentsHint=True)
        bbox = bbox_cache.ComputeWorldBound(prim)
        rng = bbox.ComputeAlignedRange()
        mn = rng.GetMin()
        mx = rng.GetMax()

        cx = 0.5 * (mn[0] + mx[0])
        cy = 0.5 * (mn[1] + mx[1])
        cz = mx[2]
        pos = (float(cx), float(cy), float(cz))
        ex = float(mx[0] - mn[0])
        ey = float(mx[1] - mn[1])
        return pos, ex, ey

    def compute_object_topdown_grasp_pose_w(self, *, prim_path: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        """Estimate a top-down grasp pose in world frame using the object's world AABB.

        Returns:
            (position_w_m, orientation_wxyz)

        - Position: (center_x, center_y, top_z) where top_z is AABB max z.
        - Orientation:
            - If align_to_min_width=False: identity quaternion (w=1).
            - If align_to_min_width=True: yaw aligns with the smaller of (extent_x, extent_y).
        """
        pos, ex, ey = self._compute_world_aabb_top_center_and_xy_extents(prim_path=prim_path)

        if self._align_to_min_width:
            yaw = 0.0 if ex <= ey else math.pi / 2.0
            half_yaw = 0.5 * yaw
            quat_wxyz = (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))
            print(f"[MG][GRASP] AABB pos=({pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f}) ex={ex:.4f} ey={ey:.4f} yaw={yaw:.3f} prim={prim_path}")
        else:
            print(f"[MG][GRASP] AABB pos=({pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f}) ex={ex:.4f} ey={ey:.4f} prim={prim_path}")
            # Keep current EE orientation unchanged by returning identity here
            quat_wxyz = (1.0, 0.0, 0.0, 0.0)
        return pos, quat_wxyz

    def get_grasp_pose_w(self, *, object_prim_path: str, robot_prim_path: Optional[str]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        return self.compute_object_topdown_grasp_pose_w(prim_path=object_prim_path)
