from __future__ import annotations

from typing import List, Optional, Tuple

from .schemas import DetectedObject, Pose


class ObjectsTracker:
    """Lightweight object tracker over spawned prims.

    For v0, we read world transforms from USD stage at each snapshot.
    If USD access fails, returns an empty list.
    """

    def __init__(self, prim_paths: Optional[List[str]] = None) -> None:
        self.prim_paths = prim_paths or []

    def _read_prim_pose(self, prim_path: str) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]]:
        try:
            # Lazy imports to avoid hard dependency in tests
            from pxr import UsdGeom
            import omni.usd

            stage = omni.usd.get_context().get_stage()  # type: ignore[attr-defined]
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                return None
            xf_cache = UsdGeom.XformCache()
            mat = xf_cache.GetLocalToWorldTransform(prim)
            # Decompose
            trans = mat.ExtractTranslation()
            rotation = mat.ExtractRotation()  # Gf.Rotation
            quat = rotation.GetQuat()  # Gf.Quatd (w, imag)
            pos = (float(trans[0]), float(trans[1]), float(trans[2]))
            w = float(quat.GetReal())
            x, y, z = quat.GetImaginary()
            ori = (w, float(x), float(y), float(z))
            return pos, ori
        except Exception:
            return None

    def snapshot(self) -> List[DetectedObject]:
        objects: List[DetectedObject] = []
        for path in self.prim_paths:
            pose = self._read_prim_pose(path)
            if pose is None:
                continue
            pos, ori = pose
            obj = DetectedObject(
                id=path.split("/")[-1],
                label="object",
                color=None,
                pose=Pose(position_m=pos, orientation_wxyz=ori),
                bbox_xywh=None,
                confidence=1.0,
            )
            objects.append(obj)
        return objects


