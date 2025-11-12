from __future__ import annotations

from typing import Optional, Tuple


class GraspPoseProvider:
    """Abstract provider for grasp pose generation in world frame."""

    def get_grasp_pose_w(self, *, object_prim_path: str, robot_prim_path: Optional[str]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
        raise NotImplementedError


