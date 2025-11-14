from __future__ import annotations

"""High-level motion generation agent.

This helper coordinates:
- object label → prim path resolution via ObjectLoader
- grasp pose estimation (e.g. OBB, Replicator)
- world → base frame transforms

It is intentionally light-weight: it does NOT own the episode loop or controller,
but provides reusable utilities that `demo.py` and other scripts can call to
fetch the latest grasp pose for a given object label.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from utils import world_to_base_pos, world_to_base_quat
from utils.objects_tracker import ObjectsTracker
from .grasp_estimation.base import GraspPoseProvider

if TYPE_CHECKING:
    # Imported only for type checking; avoids importing heavy sim/omni deps at module import time.
    from environments.object_loader import ObjectLoader


@dataclass
class MotionGenerationAgent:
    """Helper for label-based target selection and grasp pose queries."""

    sim: object
    robot: object
    controller: object
    planner: object
    grasp_provider: GraspPoseProvider
    loader: "ObjectLoader"
    robot_prim_path: Optional[str]
    tracker: Optional[ObjectsTracker] = None

    def _label_map(self) -> Dict[str, str]:
        """Return prim_path -> label mapping for last spawn."""
        try:
            return self.loader.get_last_spawn_labels()
        except Exception:
            return {}

    def select_target_prim(
        self,
        *,
        label: Optional[str],
        prim_paths: List[str],
    ) -> Optional[str]:
        """Select a target prim given an optional label and candidate prim paths.

        Selection strategy:
        - If label is None: return the first prim in prim_paths (if any).
        - Otherwise:
          - Prefer exact case-insensitive match on label.
          - Fallback to substring match on label.
          - Fallback to first prim in prim_paths.
        """
        if not prim_paths:
            return None

        label_map = self._label_map()  # prim_path -> label
        if label is None:
            return prim_paths[0]

        label_l = label.lower()

        # Exact match
        for p in prim_paths:
            lbl = str(label_map.get(p, "")).lower()
            if lbl == label_l:
                return p

        # Substring match
        for p in prim_paths:
            lbl = str(label_map.get(p, "")).lower()
            if label_l in lbl:
                return p

        # Fallback
        return prim_paths[0]

    def compute_current_grasp_for_label(
        self,
        *,
        label: Optional[str],
        prim_paths: List[str],
    ) -> Tuple[str, Tuple[float, float, float], Tuple[float, float, float, float], Tuple[float, float, float], Optional[Tuple[float, float, float, float]]]:
        """Resolve target prim by label and compute the latest grasp pose.

        Returns:
            (target_prim, pos_w, quat_wxyz_w, pos_b, quat_b_wxyz_or_None)
        """
        target_prim = self.select_target_prim(label=label, prim_paths=prim_paths)
        if target_prim is None:
            raise RuntimeError("[MG][AGENT] No target prim available for grasp computation.")

        # World-frame grasp pose from provider (e.g. OBB or Replicator)
        pos_w, quat_wxyz_w = self.grasp_provider.get_grasp_pose_w(
            object_prim_path=target_prim,
            robot_prim_path=self.robot_prim_path,
        )

        # Convert to base frame
        pos_b = world_to_base_pos(self.sim, self.robot, pos_w)
        quat_b = world_to_base_quat(self.sim, self.robot, quat_wxyz_w)

        return target_prim, pos_w, quat_wxyz_w, pos_b, quat_b

    def compute_scene_safe_z(
        self,
        prim_paths: List[str],
        clearance: float = 0.05,
    ) -> Optional[float]:
        """Compute a safe Z height (in base frame) based on all objects in the scene.

        Uses ObjectsTracker to read the latest PhysX poses for all provided prims
        and returns max_z + clearance, where max_z is the maximum Z (in base frame)
        among all tracked objects' grasp poses (top surface).

        Args:
            prim_paths: List of all object prim paths in the scene
            clearance: Extra safety margin above the highest object (meters)

        Returns:
            Safe Z height in base frame, or None if no objects are trackable
        """
        if not prim_paths:
            return None

        # Lazily initialize or update tracker to cover all prims
        if self.tracker is None:
            self.tracker = ObjectsTracker(list(prim_paths))
        else:
            self.tracker.prim_paths = list(prim_paths)

        max_z_b: Optional[float] = None
        for prim_path in prim_paths:
            try:
                # Get grasp pose (top surface) for this object
                pos_w, _ = self.grasp_provider.get_grasp_pose_w(
                    object_prim_path=prim_path,
                    robot_prim_path=self.robot_prim_path,
                )
                pos_b = world_to_base_pos(self.sim, self.robot, pos_w)
                z_b = float(pos_b[2])
                if max_z_b is None or z_b > max_z_b:
                    max_z_b = z_b
            except Exception:
                continue

        if max_z_b is None:
            return None
        return max_z_b + clearance

    def compute_current_grasp_for_prim(
        self,
        prim_path: str,
        all_prim_paths: Optional[List[str]] = None,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float], Tuple[float, float, float], Optional[Tuple[float, float, float, float]], Optional[float]]:
        """Compute latest grasp pose for a specific prim path.

        This is useful when the caller has already selected a target prim (e.g.
        random choice from spawned objects) and simply wants the up-to-date
        grasp pose in world and base frames.

        Args:
            prim_path: The target object prim path
            all_prim_paths: All spawned object prim paths (for computing safe Z)

        Returns:
            (pos_w, quat_wxyz_w, pos_b, quat_b, safe_z_b)
            where safe_z_b is the scene-wide safe height in base frame or None
        """
        # World-frame grasp pose from provider (uses current USD/PhysX state)
        pos_w, quat_wxyz_w = self.grasp_provider.get_grasp_pose_w(
            object_prim_path=prim_path,
            robot_prim_path=self.robot_prim_path,
        )

        # Convert to base frame
        pos_b = world_to_base_pos(self.sim, self.robot, pos_w)
        quat_b = world_to_base_quat(self.sim, self.robot, quat_wxyz_w)

        # Compute scene-wide safe Z if all prim paths are provided
        safe_z_b = None
        if all_prim_paths:
            safe_z_b = self.compute_scene_safe_z(all_prim_paths)

        return pos_w, quat_wxyz_w, pos_b, quat_b, safe_z_b


