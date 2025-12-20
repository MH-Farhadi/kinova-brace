from __future__ import annotations

from typing import Tuple, Optional
import importlib
import math

from .base import GraspPoseProvider


class ObbGraspPoseProvider(GraspPoseProvider):
    """Compute grasp pose using world OBB: top-center position; optional min-width yaw alignment.

    This implementation computes an oriented bounding box (OBB) from the USD world's
    bounding box corner points and aligns yaw continuously with the object's minor
    axis in the XY plane when enabled.
    """

    def __init__(self, *, align_to_min_width: bool = True) -> None:
        self._align_to_min_width = align_to_min_width

    def _compute_world_obb_xy(self, *, prim_path: str) -> Tuple[Tuple[float, float, float], float, float, float]:
        """Return (pos_w_m, yaw_rad, extent_major, extent_minor) from world OBB in XY.

        - Computes world-space bbox corner points
        - Projects to XY, runs 2D PCA to get principal axes
        - Extents are measured along principal axes; yaw is angle of the minor axis
        - pos_w_m is rectangle center at top_z (max Z among corners)
        """
        try:
            Usd = importlib.import_module("pxr.Usd")  # type: ignore[attr-defined]
            UsdGeom = importlib.import_module("pxr.UsdGeom")  # type: ignore[attr-defined]
            Gf = importlib.import_module("pxr.Gf")  # type: ignore[attr-defined]
            omni_usd = importlib.import_module("omni.usd")  # type: ignore[attr-defined]
        except Exception as e:
            raise RuntimeError(f"[MG][GRASP] USD modules not available: {e}")

        stage = omni_usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"[MG][GRASP] Invalid prim path: {prim_path}")

        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"], useExtentsHint=True)
        bbox = bbox_cache.ComputeWorldBound(prim)  # GfBBox3d

        # GfBBox3d encodes an oriented box as (GfRange3d box, GfMatrix4d transform)
        box = bbox.GetBox()
        mat = bbox.GetMatrix()

        # Local box center and half-sizes
        mn = box.GetMin()
        mx = box.GetMax()
        cx = 0.5 * (mn[0] + mx[0])
        cy = 0.5 * (mn[1] + mx[1])
        cz = 0.5 * (mn[2] + mx[2])
        hx = 0.5 * (mx[0] - mn[0])
        hy = 0.5 * (mx[1] - mn[1])
        hz = 0.5 * (mx[2] - mn[2])

        # World-space center
        c_world = mat.Transform(Gf.Vec3d(cx, cy, cz))

        # World-space axis directions (unit) for local box axes
        ax_w = mat.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
        ay_w = mat.TransformDir(Gf.Vec3d(0.0, 1.0, 0.0))
        az_w = mat.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0))
        def _norm(v):
            l = math.sqrt(float(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))
            return (float(v[0]) / l, float(v[1]) / l, float(v[2]) / l) if l > 1e-12 else (1.0, 0.0, 0.0)
        ax = _norm(ax_w)
        ay = _norm(ay_w)
        az = _norm(az_w)

        # Project base axes (ax, ay) to XY to get top-down extents
        def _proj_xy(u):
            return (u[0], u[1])
        def _len2d(u2):
            return math.hypot(u2[0], u2[1])
        ax_xy = _proj_xy(ax)
        ay_xy = _proj_xy(ay)
        lx = 2.0 * hx * _len2d(ax_xy)
        ly = 2.0 * hy * _len2d(ay_xy)

        # Pick minor axis in XY
        if lx <= ly:
            u_minor_xy = ax_xy
            extent_minor = lx
            extent_major = ly
        else:
            u_minor_xy = ay_xy
            extent_minor = ly
            extent_major = lx

        # Fallback for degenerate OBB (e.g. infinite thinness or projection failure)
        # This often happens if the object is rotated such that one axis is purely vertical
        # and the OBB computation assumes local axes align with width/height.
        if extent_minor < 1e-4 and extent_major < 1e-4:
            # Fall back to World AABB
            try:
                aligned_range = bbox.ComputeAlignedRange()
                amn = aligned_range.GetMin()
                amx = aligned_range.GetMax()
                adx = float(amx[0] - amn[0])
                ady = float(amx[1] - amn[1])
                # Center from AABB
                cx_a = 0.5 * (amn[0] + amx[0])
                cy_a = 0.5 * (amn[1] + amx[1])
                cz_a = amx[2] # Top Z
                pos = (float(cx_a), float(cy_a), float(cz_a))
                
                if adx <= ady:
                    extent_minor = adx
                    extent_major = ady
                    yaw = 0.0
                else:
                    extent_minor = ady
                    extent_major = adx
                    yaw = 0.5 * math.pi
                return pos, yaw, extent_major, extent_minor
            except Exception:
                pass

        # Yaw from minor axis projection
        yaw = math.atan2(u_minor_xy[1], u_minor_xy[0]) if _len2d(u_minor_xy) > 1e-12 else 0.0
        if yaw > math.pi:
            yaw -= 2.0 * math.pi
        if yaw <= -math.pi:
            yaw += 2.0 * math.pi

        # Top Z in world: center_z + sum_i |axis_i.z| * half_i
        top_z = float(c_world[2]) + (abs(az[2]) * hz + abs(ax[2]) * hx + abs(ay[2]) * hy)
        pos = (float(c_world[0]), float(c_world[1]), float(top_z))

        return pos, yaw, float(extent_major), float(extent_minor)

    def compute_object_topdown_grasp_pose_w(self, *, prim_path: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        """Top-down grasp pose using world OBB for continuous yaw alignment.

        Returns:
            (position_w_m, orientation_wxyz)

        - Position: (center_x, center_y, top_z) where top_z is OBB top face z.
        - Orientation:
            - If align_to_min_width=False: identity quaternion (w=1).
            - If align_to_min_width=True: yaw aligns with the OBB minor axis in XY.
        """
        pos, yaw, extent_major, extent_minor = self._compute_world_obb_xy(prim_path=prim_path)

        if self._align_to_min_width:
            half_yaw = 0.5 * yaw
            quat_wxyz = (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))
            print(f"[MG][GRASP][OBB] pos=({pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f}) major={extent_major:.4f} minor={extent_minor:.4f} yaw={yaw:.3f} prim={prim_path}")
        else:
            print(f"[MG][GRASP][OBB] pos=({pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f}) major={extent_major:.4f} minor={extent_minor:.4f} prim={prim_path}")
            # Keep current EE orientation unchanged by returning identity here
            quat_wxyz = (1.0, 0.0, 0.0, 0.0)
        return pos, quat_wxyz

    def get_grasp_pose_w(self, *, object_prim_path: str, robot_prim_path: Optional[str]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        return self.compute_object_topdown_grasp_pose_w(prim_path=object_prim_path)
