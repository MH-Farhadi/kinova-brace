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


