from __future__ import annotations

from typing import Tuple, Optional
import importlib

from .base import GraspPoseProvider


def compute_object_topdown_grasp_pose_w(*, prim_path: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    """Estimate a top-down grasp pose in world frame using the object's world AABB.

    Returns:
        (position_w_m, orientation_wxyz)

    Position: (center_x, center_y, top_z) where top_z is the AABB max z.
    Orientation: identity quaternion (w=1) as a placeholder; controller can hold orientation.
    """
    # Lazy imports to require active USD/Kit
    try:
        UsdGeom = importlib.import_module("pxr.UsdGeom")
        omni_usd = importlib.import_module("omni.usd")  # type: ignore[attr-defined]
    except Exception as e:
        raise RuntimeError(f"[MG][GRASP] USD modules not available: {e}")

    stage = omni_usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"[MG][GRASP] Invalid prim path for grasp estimation: {prim_path}")

    bbox_cache = UsdGeom.BBoxCache(0.0, ["default"], useExtentsHint=True)
    bbox = bbox_cache.ComputeWorldBound(prim)
    rng = bbox.ComputeAlignedRange()
    # min/max as GfVec3d; convert to floats
    mn = rng.GetMin()
    mx = rng.GetMax()
    cx = 0.5 * (mn[0] + mx[0])
    cy = 0.5 * (mn[1] + mx[1])
    cz = mx[2]
    pos = (float(cx), float(cy), float(cz))
    try:
        print(f"[MG][GRASP] AABB center=({cx:.4f},{cy:.4f}) top_z={cz:.4f} for prim={prim_path}")
    except Exception:
        pass
    # Keep current EE orientation unchanged by returning identity here
    quat_wxyz = (1.0, 0.0, 0.0, 0.0)
    return pos, quat_wxyz


class AabbTopGraspProvider(GraspPoseProvider):
    """Compute grasp pose using world AABB: position at top center, identity orientation."""

    def __init__(self) -> None:
        self._compute = compute_object_topdown_grasp_pose_w

    def get_grasp_pose_w(self, *, object_prim_path: str, robot_prim_path: Optional[str]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        pos, quat = self._compute(prim_path=object_prim_path)
        return pos, quat


