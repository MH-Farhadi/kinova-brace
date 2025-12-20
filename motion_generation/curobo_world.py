from __future__ import annotations

"""Utilities to sync an Isaac USD scene into cuRobo's world collision model.

cuRobo plans against its own "world model" (typically collision primitives in the
robot base frame). Isaac/Usd is the physics ground truth, but cuRobo does not
automatically read the live stage.

This module provides a minimal MVP: represent each obstacle prim by a base-frame
axis-aligned bounding box (AABB) and export it as cuRobo "cuboid" primitives.
"""

from typing import Any, Dict, List, Tuple


def _prim_world_corners(prim_path: str) -> List[Tuple[float, float, float]]:
    """Return 8 world-space corners of the prim's world bound (OBB range transformed)."""
    try:
        import importlib

        Usd = importlib.import_module("pxr.Usd")  # type: ignore[attr-defined]
        UsdGeom = importlib.import_module("pxr.UsdGeom")  # type: ignore[attr-defined]
        Gf = importlib.import_module("pxr.Gf")  # type: ignore[attr-defined]
        omni_usd = importlib.import_module("omni.usd")  # type: ignore[attr-defined]
    except Exception as e:
        raise RuntimeError(f"USD/Omni modules not available for bounds query: {e}")

    stage = omni_usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return []

    # Note: useExtentsHint=True is typically faster and robust for authored bounds.
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"], useExtentsHint=True)
    bbox = bbox_cache.ComputeWorldBound(prim)  # GfBBox3d

    box = bbox.GetBox()  # range in bbox-local frame
    mat = bbox.GetMatrix()  # bbox-local -> world

    mn = box.GetMin()
    mx = box.GetMax()

    xs = (float(mn[0]), float(mx[0]))
    ys = (float(mn[1]), float(mx[1]))
    zs = (float(mn[2]), float(mx[2]))

    corners_w: List[Tuple[float, float, float]] = []
    for x in xs:
        for y in ys:
            for z in zs:
                p = mat.Transform(Gf.Vec3d(x, y, z))
                corners_w.append((float(p[0]), float(p[1]), float(p[2])))
    return corners_w


def prim_to_base_aabb(sim, robot, prim_path: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Compute (center_b, dims_xyz) for a prim, expressed in the robot base frame."""
    from utilities import world_to_base_pos

    corners_w = _prim_world_corners(prim_path)
    if not corners_w:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

    corners_b = [world_to_base_pos(sim, robot, p) for p in corners_w]
    xs = [p[0] for p in corners_b]
    ys = [p[1] for p in corners_b]
    zs = [p[2] for p in corners_b]

    mn = (min(xs), min(ys), min(zs))
    mx = (max(xs), max(ys), max(zs))

    center = ((mn[0] + mx[0]) * 0.5, (mn[1] + mx[1]) * 0.5, (mn[2] + mx[2]) * 0.5)
    dims = (max(0.0, mx[0] - mn[0]), max(0.0, mx[1] - mn[1]), max(0.0, mx[2] - mn[2]))
    return center, dims


def build_curobo_world_cuboids(
    *,
    sim,
    robot,
    prim_paths: List[str],
    name_prefix: str = "obs",
) -> Any:
    """Build a cuRobo world model representing obstacles as cuboids in base frame.

    Important: cuRobo MotionGen APIs differ across releases:
    - Some versions accept a plain python dict like {"cuboid": {...}}
    - Others expect a typed object (e.g. WorldConfig / WorldCollision) constructed from that dict.

    We return the *most compatible* representation we can build at runtime.
    """
    cuboids: Dict[str, Any] = {}
    for i, p in enumerate(list(prim_paths)):
        center_b, dims = prim_to_base_aabb(sim, robot, p)
        if dims[0] <= 1e-6 or dims[1] <= 1e-6 or dims[2] <= 1e-6:
            continue

        # cuRobo expects pose as [x,y,z,qw,qx,qy,qz]
        cuboids[f"{name_prefix}_{i:03d}"] = {
            "dims": [float(dims[0]), float(dims[1]), float(dims[2])],
            "pose": [float(center_b[0]), float(center_b[1]), float(center_b[2]), 1.0, 0.0, 0.0, 0.0],
            "metadata": {"source_prim": str(p)},
        }

    world_dict: Dict[str, Any] = {"cuboid": cuboids}

    # Best-effort: convert dict -> cuRobo WorldConfig/WorldCollision if available.
    # We keep imports lazy and tolerant because cuRobo package namespaces vary.
    try:
        import importlib

        for mod_name in (
            "curobo.geom.sdf.world",
            "curobo.geom.world",
            "curobo.geom.types",
            "nvidia.curobo.geom.sdf.world",
            "nvidia.curobo.geom.world",
            "nvidia.curobo.geom.types",
        ):
            try:
                mod = importlib.import_module(mod_name)
            except Exception:
                continue

            # Prefer WorldConfig
            WC = getattr(mod, "WorldConfig", None)
            if WC is not None:
                for ctor in ("from_dict", "load_from_dict", "from_world_dict"):
                    fn = getattr(WC, ctor, None)
                    if fn is not None:
                        try:
                            return fn(world_dict)
                        except Exception:
                            pass

            # Some builds use WorldCollision as the type carrier
            WCol = getattr(mod, "WorldCollision", None)
            if WCol is not None:
                for ctor in ("from_dict", "load_from_dict", "from_world_dict"):
                    fn = getattr(WCol, ctor, None)
                    if fn is not None:
                        try:
                            return fn(world_dict)
                        except Exception:
                            pass
    except Exception:
        pass

    # Fallback: dict (still useful for some cuRobo plan_single signatures)
    return world_dict


