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

from utilities import world_to_base_pos, world_to_base_quat
from .grasp_estimation.base import GraspPoseProvider

if TYPE_CHECKING:
    # Imported only for type checking; avoids importing heavy sim/omni deps at module import time.
    from environments.utils.object_loader import ObjectLoader


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

    def compute_current_grasp_for_prim(
        self,
        prim_path: str,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float], Tuple[float, float, float], Optional[Tuple[float, float, float, float]]]:
        """Compute latest grasp pose for a specific prim path.

        This is useful when the caller has already selected a target prim (e.g.
        random choice from spawned objects) and simply wants the up-to-date
        grasp pose in world and base frames.
        """
        # World-frame grasp pose from provider
        pos_w, quat_wxyz_w = self.grasp_provider.get_grasp_pose_w(
            object_prim_path=prim_path,
            robot_prim_path=self.robot_prim_path,
        )

        # Convert to base frame
        pos_b = world_to_base_pos(self.sim, self.robot, pos_w)
        quat_b = world_to_base_quat(self.sim, self.robot, quat_wxyz_w)

        return pos_w, quat_wxyz_w, pos_b, quat_b


