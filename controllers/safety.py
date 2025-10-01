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
    # use svdvals for numerical stability
    svals = torch.linalg.svdvals(jacobian_b)
    # svals sorted descending; take the last one
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
    """Decide whether to block rotation for each env based on manipulability and joint-limit proximity.

    Both checks are optional. If a threshold is None, that check is skipped.

    Args:
        jacobian_b: (num_envs, 6, dof) Jacobian in base frame for EE
        joint_pos: (num_envs, dof) current arm joint positions
        lower, upper: (num_envs, dof) soft limits
        min_sigma_thresh: block if sigma_min < threshold
        joint_limit_margin: absolute radian margin to block near limits
    Returns:
        Bool tensor (num_envs,) indicating whether rotation should be blocked this step.
    """
    num_envs = joint_pos.shape[0]
    block = torch.zeros(num_envs, dtype=torch.bool, device=joint_pos.device)

    if min_sigma_thresh is not None:
        sigma_min = smallest_singular_value(jacobian_b)
        block = block | (sigma_min < float(min_sigma_thresh))

    if joint_limit_margin is not None and float(joint_limit_margin) > 0.0:
        block = block | near_joint_limits(joint_pos, lower, upper, float(joint_limit_margin))

    return block


