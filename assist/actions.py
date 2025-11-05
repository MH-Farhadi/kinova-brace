from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from .config import AssistConfig
from .schemas import Pose


@dataclass
class ActionPlan:
    cmd_stream: List[torch.Tensor]
    gripper_events: List[Tuple[int, str]]  # (step_index, "open"|"close")


def _unit(v: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    n = torch.linalg.norm(v)
    if n < eps:
        return torch.zeros_like(v)
    return v / n


def build_align_and_grasp(
    cfg: AssistConfig,
    ee_pos_w: Tuple[float, float, float],
    object_pose_w: Pose,
    dt: float,
) -> ActionPlan:
    """Create a simple approach-pregrasp-close-retreat plan as a sequence of dpose commands per step.

    Controller interprets commands as per-step deltas: [dx, dy, dz, rx, ry, rz, g?]
    We leave rotation zeros (orientation held by controller), and inject gripper close via gripper_events.
    """
    ee = torch.tensor(ee_pos_w, dtype=torch.float32)
    obj = torch.tensor(object_pose_w.position_m, dtype=torch.float32)

    # Approach to pregrasp (back-off along direction)
    approach_dir = _unit(obj - ee)
    pregrasp_target = obj - approach_dir * float(cfg.pregrasp_offset_m)

    align_speed = float(cfg.align_speed_mps)
    step_dist = max(1e-4, align_speed * dt)

    # Phase 1: approach to pregrasp
    phase1: List[torch.Tensor] = []
    cur = ee.clone()
    max_steps = int(max(20, min(600, (torch.linalg.norm(pregrasp_target - cur) / step_dist).item() + 5)))
    for _ in range(max_steps):
        delta = pregrasp_target - cur
        dist = torch.linalg.norm(delta).item()
        if dist < 0.01:
            break
        step_vec = _unit(delta) * step_dist
        cmd = torch.zeros(1, 7)
        cmd[0, 0:3] = step_vec
        phase1.append(cmd)
        cur = cur + step_vec

    # Phase 2: final in-touch approach (small nudge)
    phase2: List[torch.Tensor] = []
    for _ in range(12):  # ~0.5s at 240 Hz
        cmd = torch.zeros(1, 7)
        cmd[0, 0:3] = approach_dir * step_dist * 0.5
        phase2.append(cmd)

    # Phase 3: gripper close (handled via gripper_events)
    gripper_events: List[Tuple[int, str]] = []
    close_at = len(phase1) + len(phase2)
    gripper_events.append((close_at, "close"))

    # Phase 4: retreat
    phase4: List[torch.Tensor] = []
    retreat_dir = -approach_dir
    retreat_steps = int(max(5, (float(cfg.retreat_distance_m) / step_dist)))
    for _ in range(retreat_steps):
        cmd = torch.zeros(1, 7)
        cmd[0, 0:3] = retreat_dir * step_dist
        phase4.append(cmd)

    stream: List[torch.Tensor] = phase1 + phase2 + phase4
    return ActionPlan(cmd_stream=stream, gripper_events=gripper_events)


