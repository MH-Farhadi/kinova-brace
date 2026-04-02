from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class ArmSafetyCfg:
    """Configuration for arm safety mechanisms."""
    # Workspace bounds in base frame (meters); None disables a dimension
    workspace_min: Optional[Tuple[Optional[float], Optional[float], Optional[float]]] = None
    workspace_max: Optional[Tuple[Optional[float], Optional[float], Optional[float]]] = None
    # Singularity avoidance
    min_sigma_thresh: Optional[float] = 0.005  # Block/project if sigma_min < threshold
    # Joint limit avoidance
    joint_limit_margin_rad: Optional[float] = 0.10  # Margin for qdot clamping


class ArmSafety:
    """Encapsulates safety checks and projections for arm control.

    Handles workspace clamping, orientation holding, singularity avoidance,
    joint limit protection, and directional recovery. Instantiated once with config.
    """

    def __init__(self, cfg: ArmSafetyCfg, num_envs: int, device: str) -> None:
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device

        # Pre-compute workspace bound tensors if defined
        self._ws_min = None
        self._ws_max = None
        if self.cfg.workspace_min is not None:
            self._ws_min = torch.tensor(
                [float("-inf") if v is None else v for v in self.cfg.workspace_min],
                device=self.device,
                dtype=torch.float32,
            ).view(1, 3)
        if self.cfg.workspace_max is not None:
            self._ws_max = torch.tensor(
                [float("inf") if v is None else v for v in self.cfg.workspace_max],
                device=self.device,
                dtype=torch.float32,
            ).view(1, 3)

        # Thresholds as floats for quick access
        self._min_sigma_thresh = float(self.cfg.min_sigma_thresh) if self.cfg.min_sigma_thresh is not None else None
        self._joint_limit_margin = float(self.cfg.joint_limit_margin_rad) if self.cfg.joint_limit_margin_rad is not None else None

    def clamp_position(self, pos_b: torch.Tensor) -> torch.Tensor:
        """Clamp position to workspace bounds in base frame."""
        if self._ws_min is None and self._ws_max is None:
            return pos_b
        result = pos_b
        if self._ws_min is not None:
            result = torch.maximum(result, self._ws_min.to(result.device))
        if self._ws_max is not None:
            result = torch.minimum(result, self._ws_max.to(result.device))
        return result

    def hold_orientation(self, current_quat: torch.Tensor, hold_quat: Optional[torch.Tensor], enabled: bool) -> torch.Tensor:
        """Hold orientation if enabled and hold_quat is provided."""
        if enabled and hold_quat is not None:
            return hold_quat
        return current_quat

    def smallest_singular_value(self, jacobian_b: torch.Tensor) -> torch.Tensor:
        """Compute smallest singular value of Jacobian per env."""
        svals = torch.linalg.svdvals(jacobian_b)
        return svals[..., -1]

    def project_twist_away_from_low_sigma(self, jacobian_b: torch.Tensor, twist_b: torch.Tensor) -> torch.Tensor:
        """Project twist away from low-sigma Jacobian directions."""
        if self._min_sigma_thresh is None:
            return twist_b
        U, S, _ = torch.linalg.svd(jacobian_b, full_matrices=False)
        low = (S < self._min_sigma_thresh).to(twist_b.dtype)  # (N,6)
        M = torch.diag_embed(low)  # (N,6,6)
        P = U @ M @ U.transpose(-1, -2)  # (N,6,6)
        twist_b_col = twist_b.unsqueeze(-1)  # (N,6,1)
        removed = P @ twist_b_col  # (N,6,1)
        result = twist_b_col - removed
        return result.squeeze(-1)

    def project_rotation_toward_quat(
        self, current_quat: torch.Tensor, target_quat: torch.Tensor, drot: torch.Tensor
    ) -> torch.Tensor:
        """Project drot to only the component reducing angular error to target_quat."""
        # Error quaternion: q_err = q_target * conj(q_current)
        q_err = self._quat_multiply(target_quat, self._quat_conjugate(current_quat))
        err_rotvec = self._quat_to_rotvec(q_err)  # (N,3)
        err_norm = torch.linalg.norm(err_rotvec, dim=-1, keepdim=True).clamp_min(1e-8)
        err_dir = err_rotvec / err_norm
        # Project drot onto direction toward target
        proj_mag = (drot * err_dir).sum(dim=-1, keepdim=True)
        # Keep only positive component (reduces error)
        proj_mag_clamped = torch.clamp(proj_mag, min=0.0)
        return err_dir * proj_mag_clamped

    def clamp_qdot_near_limits(
        self, qdot: torch.Tensor, q: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor
    ) -> torch.Tensor:
        """Zero qdot components pushing into nearby joint limits."""
        if self._joint_limit_margin is None or self._joint_limit_margin <= 0.0:
            return qdot
        dist_to_lower = q - lower
        dist_to_upper = upper - q
        near_lower = dist_to_lower < self._joint_limit_margin
        near_upper = dist_to_upper < self._joint_limit_margin
        # Block velocities moving further into limits
        block_lower = near_lower & (qdot < 0.0)
        block_upper = near_upper & (qdot > 0.0)
        block = block_lower | block_upper
        qdot_safe = qdot.clone()
        qdot_safe[block] = 0.0
        return qdot_safe

    # Private quaternion helper methods
    @staticmethod
    def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
        """Conjugate of quaternion q = [w, x, y, z]. Shape (..., 4)."""
        return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)

    @staticmethod
    def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Hamilton product q1 * q2, both [w, x, y, z]. Shape (..., 4)."""
        w1, x1, y1, z1 = q1.unbind(dim=-1)
        w2, x2, y2, z2 = q2.unbind(dim=-1)
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack([w, x, y, z], dim=-1)

    @staticmethod
    def _quat_to_rotvec(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Convert unit quaternion [w,x,y,z] to rotation vector (axis*angle). Shape (...,3)."""
        w = torch.clamp(q[..., 0], -1.0, 1.0)
        v = q[..., 1:4]
        v_norm = torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(eps)
        angle = 2.0 * torch.atan2(v_norm, w.unsqueeze(-1))
        axis = v / v_norm
        return axis * angle
