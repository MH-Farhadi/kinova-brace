from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from controllers.base import InputProvider


class WaypointFollowerInput(InputProvider):
    """Programmatic input that drives the EE toward a sequence of waypoints (base frame).

    - Produces per-step Cartesian deltas [dx, dy, dz, 0, 0, 0, g].
    - Step magnitudes are bounded by `step_pos_m` per step.
    - Supports queued gripper commands by returning [0..0, g] for a fixed number of steps.
    - The runner must update the current EE pose each step via `set_current_pose_b`.
    """

    def __init__(
        self,
        *,
        step_pos_m: float,
        tol_m: float = 0.005,
        device: str = "cpu",
    ) -> None:
        self.step_pos_m = float(step_pos_m)
        self.tol_m = float(tol_m)
        self.device = torch.device(device)
        self._waypoints_b: List[torch.Tensor] = []
        self._current_ee_pos_b: Optional[torch.Tensor] = None
        self._gripper_steps_left: int = 0
        self._gripper_value: float = 0.0
        self._last_cmd: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self._waypoints_b = []
        self._current_ee_pos_b = None
        self._gripper_steps_left = 0
        self._gripper_value = 0.0
        self._last_cmd = None

    def set_current_pose_b(self, ee_pos_b: torch.Tensor) -> None:
        """Update the current EE position in base frame: shape (3,) or (1,3)."""
        if ee_pos_b.ndim == 2:
            ee_pos_b = ee_pos_b.view(-1)
        self._current_ee_pos_b = ee_pos_b.to(self.device)

    def set_waypoints_b(self, points_b: List[Tuple[float, float, float]]) -> None:
        self._waypoints_b = [torch.tensor(p, dtype=torch.float32, device=self.device) for p in points_b]

    def queue_gripper(self, g_value: float, steps: int) -> None:
        self._gripper_value = float(g_value)
        self._gripper_steps_left = max(0, int(steps))

    def advance(self) -> torch.Tensor:
        # Priority: gripper commands
        if self._gripper_steps_left > 0:
            self._gripper_steps_left -= 1
            cmd = torch.zeros(1, 7, dtype=torch.float32, device=self.device)
            cmd[0, 6] = float(self._gripper_value)
            self._last_cmd = cmd
            return cmd

        if self._current_ee_pos_b is None or len(self._waypoints_b) == 0:
            cmd = torch.zeros(1, 6, dtype=torch.float32, device=self.device)
            self._last_cmd = cmd
            return cmd

        goal = self._waypoints_b[0]
        diff = goal - self._current_ee_pos_b
        dist = torch.linalg.norm(diff).item()
        if dist <= self.tol_m:
            # Reached; pop and output zero
            self._waypoints_b.pop(0)
            cmd = torch.zeros(1, 6, dtype=torch.float32, device=self.device)
            self._last_cmd = cmd
            return cmd

        step = self.step_pos_m
        d = (diff / (dist + 1e-9)) * min(step, dist)
        cmd6 = torch.zeros(1, 6, dtype=torch.float32, device=self.device)
        cmd6[0, 0:3] = d
        self._last_cmd = cmd6
        return cmd6

    @property
    def last_cmd(self) -> Optional[torch.Tensor]:
        return self._last_cmd


