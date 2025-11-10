from __future__ import annotations

from typing import Optional, Tuple, Any, List


class GraspPoseProvider:
    """Abstract provider for grasp pose generation in world frame."""

    def get_grasp_pose_w(self, *, object_prim_path: str, robot_prim_path: Optional[str]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        raise NotImplementedError


class AabbTopGraspProvider(GraspPoseProvider):
    """Compute grasp pose using world AABB: position at top center, identity orientation."""

    def __init__(self) -> None:
        # Lazy import to avoid USD deps during module import
        from ..samplers.grasp_estimation import compute_object_topdown_grasp_pose_w  # noqa: WPS433

        self._compute = compute_object_topdown_grasp_pose_w

    def get_grasp_pose_w(self, *, object_prim_path: str, robot_prim_path: Optional[str]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        pos, quat = self._compute(prim_path=object_prim_path)
        return pos, quat


class ReplicatorGraspProvider(GraspPoseProvider):
    """Use isaacsim.replicator.grasping.GraspingManager to generate/evaluate 6D grasp poses.

    Notes:
    - Requires the extension `isaacsim.replicator.grasping` to be enabled.
    - Accepts either a YAML config or an inline sampler_config dict (or both).
    - Returns the first available grasp pose in world frame by default.
    """

    def __init__(
        self,
        *,
        gripper_prim_path: Optional[str],
        config_yaml_path: Optional[str] = None,
        sampler_config: Optional[dict] = None,
        max_candidates: int = 16,
    ) -> None:
        import omni.kit.app  # noqa: WPS433

        self._gripper_prim_path = gripper_prim_path
        self._config_yaml_path = config_yaml_path
        self._sampler_config = sampler_config
        self._max_candidates = int(max_candidates)
        # Enable extension if needed
        ext_manager = omni.kit.app.get_app().get_extension_manager()
        if not ext_manager.is_extension_enabled("isaacsim.replicator.grasping"):
            ext_manager.set_extension_enabled_immediate("isaacsim.replicator.grasping", True)
        # Lazy import after enabling
        from isaacsim.replicator.grasping.grasping_manager import GraspingManager  # noqa: WPS433

        self._gm = GraspingManager()

    def _ensure_config(self, object_prim_path: str) -> None:
        # Load config once if provided
        if getattr(self, "_loaded_cfg", False) is not True:
            if self._config_yaml_path:
                status = self._gm.load_config(self._config_yaml_path)
                print(f"[MG][GRASP][REP] Loaded config '{self._config_yaml_path}': {status}")
            self._loaded_cfg = True
        # Object to grasp (prefer setter methods to avoid read-only properties)
        def _try_set(gm, names, value):
            for name in names:
                fn = getattr(gm, name, None)
                if callable(fn):
                    try:
                        fn(value)
                        return True
                    except Exception:
                        pass
            return False
        if object_prim_path:
            ok_obj = _try_set(self._gm, ["set_object_prim_path", "set_object_path"], object_prim_path)
            if not ok_obj:
                try:
                    setattr(self._gm, "object_path", object_prim_path)
                    ok_obj = True
                except Exception:
                    ok_obj = False
            print(f"[MG][GRASP][REP] object_prim_path set: {ok_obj} path='{object_prim_path}'")
        # Gripper path (optional override)
        if self._gripper_prim_path:
            ok_gr = _try_set(self._gm, ["set_gripper_prim_path", "set_gripper_path"], self._gripper_prim_path)
            if not ok_gr:
                try:
                    setattr(self._gm, "gripper_path", self._gripper_prim_path)
                    ok_gr = True
                except Exception:
                    ok_gr = False
            print(f"[MG][GRASP][REP] gripper_path set: {ok_gr} path='{self._gripper_prim_path}'")

        # Sampler config (inline) if provided and manager lacks it
        if self._sampler_config is not None:
            # Prefer a setter if present
            if not _try_set(self._gm, ["set_sampler_config"], dict(self._sampler_config)):
                try:
                    setattr(self._gm, "sampler_config", dict(self._sampler_config))
                except Exception:
                    print("[MG][GRASP][REP][WARN] Could not set sampler_config; relying on YAML or defaults.")

    @staticmethod
    def _to_pos_quat(pose_w: Any) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]]:
        """Convert various Replicator/Isaac pose formats to (pos, quat_wxyz)."""
        def as_float3(v: Any) -> Optional[Tuple[float, float, float]]:
            try:
                return (float(v[0]), float(v[1]), float(v[2]))
            except Exception:
                return None

        def quat_xyzw_to_wxyz(q: Any) -> Optional[Tuple[float, float, float, float]]:
            try:
                x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
                return (w, x, y, z)
            except Exception:
                return None

        def quat_wxyz(q: Any) -> Optional[Tuple[float, float, float, float]]:
            try:
                return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
            except Exception:
                return None

        def mat3_to_quat_wxyz(m3: Any) -> Optional[Tuple[float, float, float, float]]:
            # Robust, dependency-free 3x3 to quaternion (wxyz)
            try:
                m00, m01, m02 = float(m3[0][0]), float(m3[0][1]), float(m3[0][2])
                m10, m11, m12 = float(m3[1][0]), float(m3[1][1]), float(m3[1][2])
                m20, m21, m22 = float(m3[2][0]), float(m3[2][1]), float(m3[2][2])
                tr = m00 + m11 + m22
                if tr > 0.0:
                    s = (tr + 1.0) ** 0.5 * 2.0
                    w = 0.25 * s
                    x = (m21 - m12) / s
                    y = (m02 - m20) / s
                    z = (m10 - m01) / s
                elif (m00 > m11) and (m00 > m22):
                    s = (1.0 + m00 - m11 - m22) ** 0.5 * 2.0
                    w = (m21 - m12) / s
                    x = 0.25 * s
                    y = (m01 + m10) / s
                    z = (m02 + m20) / s
                elif m11 > m22:
                    s = (1.0 + m11 - m00 - m22) ** 0.5 * 2.0
                    w = (m02 - m20) / s
                    x = (m01 + m10) / s
                    y = 0.25 * s
                    z = (m12 + m21) / s
                else:
                    s = (1.0 + m22 - m00 - m11) ** 0.5 * 2.0
                    w = (m10 - m01) / s
                    x = (m02 + m20) / s
                    y = (m12 + m21) / s
                    z = 0.25 * s
                return (float(w), float(x), float(y), float(z))
            except Exception:
                return None

        try:
            # dict forms
            if isinstance(pose_w, dict):
                # Common: { position: [x,y,z], orientation: [w,x,y,z] } or xyzw
                pos = pose_w.get("position", pose_w.get("position_m") or pose_w.get("pos"))
                ori = pose_w.get("orientation", pose_w.get("orientation_wxyz") or pose_w.get("quat") or pose_w.get("quat_wxyz"))
                if pos is not None and ori is not None and len(pos) == 3 and len(ori) == 4:
                    p = as_float3(pos)
                    # Try wxyz first, else assume xyzw
                    q = quat_wxyz(ori) if abs(float(ori[0])) <= 1.0 and abs(float(ori[3])) <= 1.0 else None
                    q = q or quat_wxyz(ori)
                    if p is not None and q is not None:
                        return p, q
                # Alternative keys: orientation_xyzw, rotation_xyzw, etc.
                ori_xyzw = pose_w.get("orientation_xyzw", pose_w.get("quat_xyzw") or pose_w.get("rotation_xyzw"))
                if pos is not None and ori_xyzw is not None and len(pos) == 3 and len(ori_xyzw) == 4:
                    p = as_float3(pos)
                    q = quat_xyzw_to_wxyz(ori_xyzw)
                    if p is not None and q is not None:
                        return p, q
                # Homogeneous matrix embedded
                mat = pose_w.get("matrix", pose_w.get("transform") or pose_w.get("T"))
                if mat is not None:
                    # Try 4x4 nested lists
                    if isinstance(mat, (list, tuple)) and len(mat) == 4 and all(len(r) == 4 for r in mat):
                        p = (float(mat[0][3]), float(mat[1][3]), float(mat[2][3]))
                        q = mat3_to_quat_wxyz([[mat[0][0], mat[0][1], mat[0][2]], [mat[1][0], mat[1][1], mat[1][2]], [mat[2][0], mat[2][1], mat[2][2]]])
                        if q is not None:
                            return p, q
            # Objects with common attributes
            for attr in ("transform", "matrix", "world_pose", "pose", "T", "tf", "mat"):
                if hasattr(pose_w, attr):
                    sub = getattr(pose_w, attr)
                    conv = ReplicatorGraspProvider._to_pos_quat(sub)
                    if conv is not None:
                        return conv
            # USD/pxr matrix with ExtractTranslation/ExtractRotation
            if hasattr(pose_w, "ExtractTranslation") and hasattr(pose_w, "ExtractRotation"):
                trans = pose_w.ExtractTranslation()
                rot = pose_w.ExtractRotation()
                quat = rot.GetQuat()
                p = (float(trans[0]), float(trans[1]), float(trans[2]))
                w = float(quat.GetReal())
                x, y, z = quat.GetImaginary()
                return p, (w, float(x), float(y), float(z))
            # Tuple of (pxr.Gf.Vec3*, pxr.Gf.Quat*) or similar
            if isinstance(pose_w, (list, tuple)) and len(pose_w) == 2:
                pos_obj, quat_obj = pose_w[0], pose_w[1]
                # Try pxr quaternion
                if hasattr(quat_obj, "GetReal") and hasattr(quat_obj, "GetImaginary"):
                    w = float(quat_obj.GetReal())
                    xi, yi, zi = quat_obj.GetImaginary()
                    # pos object with x/y/z or indexable
                    if hasattr(pos_obj, "x") and hasattr(pos_obj, "y") and hasattr(pos_obj, "z"):
                        p = (float(pos_obj.x), float(pos_obj.y), float(pos_obj.z))
                        return p, (w, float(xi), float(yi), float(zi))
                    try:
                        p = (float(pos_obj[0]), float(pos_obj[1]), float(pos_obj[2]))
                        return p, (w, float(xi), float(yi), float(zi))
                    except Exception:
                        pass
            # numpy 4x4
            if hasattr(pose_w, "shape") and tuple(getattr(pose_w, "shape", ())) == (4, 4):
                m = pose_w
                p = (float(m[0, 3]), float(m[1, 3]), float(m[2, 3]))
                q = mat3_to_quat_wxyz([[m[0, 0], m[0, 1], m[0, 2]], [m[1, 0], m[1, 1], m[1, 2]], [m[2, 0], m[2, 1], m[2, 2]]])
                if q is not None:
                    return p, q
            # tuple/list ((x,y,z), (w,x,y,z)) or ((x,y,z), (x,y,z,w))
            if isinstance(pose_w, (list, tuple)):
                if len(pose_w) == 2:
                    pos, quat = pose_w
                    if len(pos) == 3 and len(quat) == 4:
                        p = as_float3(pos)
                        # Try interpret as wxyz, else xyzw
                        q = quat_wxyz(quat) or quat_xyzw_to_wxyz(quat)
                        if p is not None and q is not None:
                            return p, q
                # 7D vector [px,py,pz,qx,qy,qz,qw] or [px,py,pz,qw,qx,qy,qz]
                if len(pose_w) == 7:
                    px, py, pz = float(pose_w[0]), float(pose_w[1]), float(pose_w[2])
                    # Heuristic: if last element has largest magnitude, assume qw last (qx,qy,qz,qw)
                    qx, qy, qz, q4 = float(pose_w[3]), float(pose_w[4]), float(pose_w[5]), float(pose_w[6])
                    # Prefer qw-last
                    return (px, py, pz), (q4, qx, qy, qz)
                # Flattened 4x4 (16)
                if len(pose_w) == 16:
                    m = [list(pose_w[i*4:(i+1)*4]) for i in range(4)]
                    p = (float(m[0][3]), float(m[1][3]), float(m[2][3]))
                    q = mat3_to_quat_wxyz([[m[0][0], m[0][1], m[0][2]], [m[1][0], m[1][1], m[1][2]], [m[2][0], m[2][1], m[2][2]]])
                    if q is not None:
                        return p, q
        except Exception:
            return None
        return None

    def get_grasp_pose_w(self, *, object_prim_path: str, robot_prim_path: Optional[str]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        # Prepare manager
        self._ensure_config(object_prim_path)
        # If config had pre-populated grasp locations, skip generation; else generate
        poses_ready = False
        try:
            poses_ready = bool(getattr(self._gm, "grasp_locations", None))
        except Exception:
            poses_ready = False
        if not poses_ready:
            print("[MG][GRASP][REP] Generating grasp poses via Replicator...")
            success = self._gm.generate_grasp_poses()
            if not success:
                raise RuntimeError("[MG][GRASP][REP] generate_grasp_poses() returned failure.")
        # Retrieve world-frame poses
        poses_w = self._gm.get_grasp_poses(in_world_frame=True)
        if not poses_w:
            raise RuntimeError("[MG][GRASP][REP] No grasp poses available from Replicator.")
        # Select first N candidates and pick the first for now (could sort by metric if available)
        pose = poses_w[0]
        # Try direct conversion
        conv = self._to_pos_quat(pose)
        # If conversion failed, try common attribute containers (pose.transform, pose.matrix, pose.world_pose)
        if conv is None:
            for attr in ("transform", "matrix", "world_pose", "pose", "T", "tf", "mat"):
                try:
                    if hasattr(pose, attr):
                        sub = getattr(pose, attr)
                        conv = self._to_pos_quat(sub)
                        if conv is not None:
                            break
                except Exception:
                    pass
        if conv is None:
            try:
                print(f"[MG][GRASP][REP][ERROR] Unsupported grasp pose type={type(pose)} keys={list(pose.keys()) if isinstance(pose, dict) else 'n/a'}")
            except Exception:
                print(f"[MG][GRASP][REP][ERROR] Unsupported grasp pose type={type(pose)} (no introspection)")
            raise RuntimeError("[MG][GRASP][REP] Unsupported grasp pose format returned.")
        return conv



