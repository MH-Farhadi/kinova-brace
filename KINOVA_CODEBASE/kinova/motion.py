from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch


@dataclass
class MotionCommandBuilder:
    """Utility to build simple Cartesian jog commands for controllers.

    Produces per-step command tensors shaped (N, D) where D >= 6 or 7.
    The convention matches CartesianVelocityJogController: [dx, dy, dz, rx, ry, rz, g?].
    """

    device: str = "cuda:0"
    dtype: torch.dtype = torch.float32

    def zero(self, num_envs: int, with_gripper: bool = True) -> torch.Tensor:
        d = 7 if with_gripper else 6
        return torch.zeros((num_envs, d), device=self.device, dtype=self.dtype)

    def move(self, num_envs: int, direction: Literal["right", "left", "up", "down", "forward", "backward"], step: float) -> torch.Tensor:
        cmd = self.zero(num_envs)
        if direction == "right":
            cmd[:, 0] = step
        elif direction == "left":
            cmd[:, 0] = -step
        elif direction == "forward":
            cmd[:, 1] = step
        elif direction == "backward":
            cmd[:, 1] = -step
        elif direction == "up":
            cmd[:, 2] = step
        elif direction == "down":
            cmd[:, 2] = -step
        return cmd

    def rotate(self, num_envs: int, axis: Literal["rx", "ry", "rz"], step_rad: float) -> torch.Tensor:
        cmd = self.zero(num_envs)
        if axis == "rx":
            cmd[:, 3] = step_rad
        elif axis == "ry":
            cmd[:, 4] = step_rad
        elif axis == "rz":
            cmd[:, 5] = step_rad
        return cmd

    def gripper(self, num_envs: int, open_cmd: bool) -> torch.Tensor:
        cmd = self.zero(num_envs, with_gripper=True)
        cmd[:, 6] = 1.0 if open_cmd else -1.0
        return cmd


class MotionPrimitives:
    """Thin helper that can drive a Cartesian controller using MotionCommandBuilder outputs."""

    def __init__(self, builder: MotionCommandBuilder, controller) -> None:
        self.builder = builder
        self.controller = controller

    def move_right(self, step: float) -> torch.Tensor:
        return self.builder.move(self.controller.num_envs, "right", step)

    def move_left(self, step: float) -> torch.Tensor:
        return self.builder.move(self.controller.num_envs, "left", step)

    def move_up(self, step: float) -> torch.Tensor:
        return self.builder.move(self.controller.num_envs, "up", step)

    def move_down(self, step: float) -> torch.Tensor:
        return self.builder.move(self.controller.num_envs, "down", step)

    def move_forward(self, step: float) -> torch.Tensor:
        return self.builder.move(self.controller.num_envs, "forward", step)

    def move_backward(self, step: float) -> torch.Tensor:
        return self.builder.move(self.controller.num_envs, "backward", step)

    def rotate_rx(self, step_rad: float) -> torch.Tensor:
        return self.builder.rotate(self.controller.num_envs, "rx", step_rad)

    def rotate_ry(self, step_rad: float) -> torch.Tensor:
        return self.builder.rotate(self.controller.num_envs, "ry", step_rad)

    def rotate_rz(self, step_rad: float) -> torch.Tensor:
        return self.builder.rotate(self.controller.num_envs, "rz", step_rad)

    def gripper_open(self) -> torch.Tensor:
        return self.builder.gripper(self.controller.num_envs, True)

    def gripper_close(self) -> torch.Tensor:
        return self.builder.gripper(self.controller.num_envs, False)


