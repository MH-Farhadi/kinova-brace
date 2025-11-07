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
        # Object to grasp
        if not self._gm.get_object_prim_path():
            self._gm.object_path = object_prim_path
        else:
            self._gm.object_path = object_prim_path
        # Gripper path
        if self._gripper_prim_path:
            self._gm.gripper_path = self._gripper_prim_path

        # Sampler config (inline) if provided and manager lacks it
        if (self._sampler_config is not None) and (not getattr(self._gm, "sampler_config", None)):
            self._gm.sampler_config = dict(self._sampler_config)

    @staticmethod
    def _to_pos_quat(pose_w: Any) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]]:
        # Try common representations: dict with position/orientation, 4x4 matrix, tuple pairs
        try:
            # dict
            if isinstance(pose_w, dict):
                pos = pose_w.get("position", pose_w.get("position_m"))
                quat = pose_w.get("orientation", pose_w.get("orientation_wxyz"))
                if pos is not None and quat is not None and len(pos) == 3 and len(quat) == 4:
                    return (float(pos[0]), float(pos[1]), float(pos[2])), (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
            # homogeneous matrix (4x4)
            if hasattr(pose_w, "shape") and tuple(getattr(pose_w, "shape", ())) == (4, 4):
                import numpy as np  # noqa: WPS433

                m = np.asarray(pose_w)
                px, py, pz = float(m[0, 3]), float(m[1, 3]), float(m[2, 3])
                # Convert rotation to quaternion (wxyz)
                import scipy.spatial.transform as sst  # type: ignore  # noqa: WPS433

                r = sst.Rotation.from_matrix(m[0:3, 0:3])
                x, y, z, w = r.as_quat()  # xyzw
                return (px, py, pz), (float(w), float(x), float(y), float(z))
            # tuple/list ((x,y,z), (w,x,y,z))
            if isinstance(pose_w, (list, tuple)) and len(pose_w) == 2:
                pos, quat = pose_w
                if len(pos) == 3 and len(quat) == 4:
                    return (float(pos[0]), float(pos[1]), float(pos[2])), (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
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
        conv = self._to_pos_quat(pose)
        if conv is None:
            raise RuntimeError("[MG][GRASP][REP] Unsupported grasp pose format returned.")
        return conv



