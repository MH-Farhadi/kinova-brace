from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class WorkspaceBounds:
    min_xyz: Optional[Tuple[float, float, float]] = None
    max_xyz: Optional[Tuple[float, float, float]] = None

    def clamp(self, pos_b: torch.Tensor, device: str) -> torch.Tensor:
        if self.min_xyz is None or self.max_xyz is None:
            return pos_b
        ws_min = torch.tensor(self.min_xyz, device=device, dtype=pos_b.dtype).view(1, 3)
        ws_max = torch.tensor(self.max_xyz, device=device, dtype=pos_b.dtype).view(1, 3)
        return torch.minimum(torch.maximum(pos_b, ws_min), ws_max)


def hold_orientation(ee_quat_current_b: torch.Tensor, ee_quat_hold_b: torch.Tensor | None, enabled: bool) -> torch.Tensor:
    if enabled and ee_quat_hold_b is not None:
        return ee_quat_hold_b
    return ee_quat_current_b


