from __future__ import annotations

from typing import Tuple


def compute_object_topdown_grasp_pose_w(*, prim_path: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    """Estimate a top-down grasp pose in world frame using the object's world AABB.

    Returns:
        (position_w_m, orientation_wxyz)

    Position: (center_x, center_y, top_z) where top_z is the AABB max z.
    Orientation: identity quaternion (w=1) as a placeholder; controller can hold orientation.
    """
    # Lazy imports to require active USD/Kit
    Usd = __import__("pxr.Usd", fromlist=["Usd"]).Usd
    UsdGeom = __import__("pxr.UsdGeom", fromlist=["UsdGeom"]).UsdGeom
    omni_usd = __import__("omni.usd", fromlist=["get_context"])  # type: ignore[attr-defined]

    stage = omni_usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Invalid prim path for grasp estimation: {prim_path}")

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"], useExtentsHint=True)
    bbox = bbox_cache.ComputeWorldBound(prim)
    rng = bbox.ComputeAlignedRange()
    # min/max as GfVec3d; convert to floats
    mn = rng.GetMin()
    mx = rng.GetMax()
    cx = 0.5 * (mn[0] + mx[0])
    cy = 0.5 * (mn[1] + mx[1])
    cz = mx[2]
    pos = (float(cx), float(cy), float(cz))
    # Keep current EE orientation unchanged by returning identity here
    quat_wxyz = (1.0, 0.0, 0.0, 0.0)
    return pos, quat_wxyz


