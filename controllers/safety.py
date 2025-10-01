from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class WorkspaceBounds:
    min_xyz: Optional[Tuple[Optional[float], Optional[float], Optional[float]]] = None
    max_xyz: Optional[Tuple[Optional[float], Optional[float], Optional[float]]] = None

    def clamp(self, pos_b: torch.Tensor, device: str) -> torch.Tensor:
        if self.min_xyz is None and self.max_xyz is None:
            return pos_b
        result = pos_b
        if self.min_xyz is not None:
            ws_min = torch.tensor(
                [float("-inf") if v is None else v for v in self.min_xyz],
                device=device,
                dtype=pos_b.dtype,
            ).view(1, 3)
            result = torch.maximum(result, ws_min)
        if self.max_xyz is not None:
            ws_max = torch.tensor(
                [float("inf") if v is None else v for v in self.max_xyz],
                device=device,
                dtype=pos_b.dtype,
            ).view(1, 3)
            result = torch.minimum(result, ws_max)
        return result


def hold_orientation(ee_quat_current_b: torch.Tensor, ee_quat_hold_b: torch.Tensor | None, enabled: bool) -> torch.Tensor:
    if enabled and ee_quat_hold_b is not None:
        return ee_quat_hold_b
    return ee_quat_current_b


# --- Minimal safety helpers for rotation gating ---

def smallest_singular_value(jacobian_b: torch.Tensor) -> torch.Tensor:
    """Compute the smallest singular value of the 6xN Jacobian per env.

    Args:
        jacobian_b: Tensor shaped (num_envs, 6, N)
    Returns:
        Tensor (num_envs,) of sigma_min values.
    """
    svals = torch.linalg.svdvals(jacobian_b)
    return svals[..., -1]


def near_joint_limits(joint_pos: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor, margin: float) -> torch.Tensor:
    """Check if any arm joint is within an absolute margin [rad] of its limits.

    Args:
        joint_pos: (num_envs, dof)
        lower: (num_envs, dof)
        upper: (num_envs, dof)
        margin: absolute margin in radians
    Returns:
        Bool tensor (num_envs,) True if any joint is near limits.
    """
    dist_to_lower = joint_pos - lower
    dist_to_upper = upper - joint_pos
    near_lower = dist_to_lower < margin
    near_upper = dist_to_upper < margin
    return (near_lower | near_upper).any(dim=-1)


def should_block_rotation(
    jacobian_b: torch.Tensor,
    joint_pos: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    min_sigma_thresh: Optional[float] = None,
    joint_limit_margin: Optional[float] = None,
) -> torch.Tensor:
    """Decide whether to block rotation for each env based on manipulability and joint-limit proximity."""
    num_envs = joint_pos.shape[0]
    block = torch.zeros(num_envs, dtype=torch.bool, device=joint_pos.device)

    if min_sigma_thresh is not None:
        sigma_min = smallest_singular_value(jacobian_b)
        block = block | (sigma_min < float(min_sigma_thresh))

    if joint_limit_margin is not None and float(joint_limit_margin) > 0.0:
        block = block | near_joint_limits(joint_pos, lower, upper, float(joint_limit_margin))

    return block


# --- Quaternion helpers for directional back-out ---

def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of quaternion q = [w, x, y, z]. Shape (..., 4)."""
    return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)


def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product of quaternions q1*q2, both shaped (...,4) as [w, x, y, z]."""
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)


def quat_to_rotvec(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Convert unit quaternion q=[w,x,y,z] to rotation vector (axis*angle). Shape (...,3)."""
    w = torch.clamp(q[..., 0], -1.0, 1.0)
    v = q[..., 1:4]
    v_norm = torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(eps)
    angle = 2.0 * torch.atan2(v_norm, w.unsqueeze(-1))
    axis = v / v_norm
    return axis * angle


def project_rotation_toward_quat(current_quat: torch.Tensor, target_quat: torch.Tensor, drot: torch.Tensor) -> torch.Tensor:
    """Keep only the component of drot that reduces the angular error to target_quat.

    Args:
        current_quat: (N,4) current orientation
        target_quat: (N,4) target/safe orientation to move toward
        drot: (N,3) commanded rotation vector
    Returns:
        (N,3) projected rotation vector that moves toward target; zeros otherwise.
    """
    # Error quaternion that takes current -> target: q_err = q_target * conj(q_current)
    q_err = quat_multiply(target_quat, quat_conjugate(current_quat))
    err_rotvec = quat_to_rotvec(q_err)  # (N,3)
    err_norm = torch.linalg.norm(err_rotvec, dim=-1, keepdim=True).clamp_min(1e-8)
    err_dir = err_rotvec / err_norm
    # Project drot onto direction toward target
    proj_mag = (drot * err_dir).sum(dim=-1, keepdim=True)
    # Keep only positive component (reduces error). Negative increases error
    proj_mag_clamped = torch.clamp(proj_mag, min=0.0)
    return err_dir * proj_mag_clamped


