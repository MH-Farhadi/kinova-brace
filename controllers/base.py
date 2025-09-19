from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import torch


@dataclass
class ArmControllerConfig:
    """Generic configuration for an arm controller.

    Attributes
    ----------
    ee_link_name:
        Name of the end-effector link used by the controller.
    arm_joint_regex:
        Regular expression that matches arm joint names (excludes gripper joints).
    use_relative_mode:
        If True, commands are interpreted relative to the current state.
    device:
        Torch device string used for controller tensors.
    """

    ee_link_name: str
    arm_joint_regex: str = "j2n6s300_joint_[1-6]"
    use_relative_mode: bool = True
    device: str = "cuda:0"


class InputProvider(Protocol):
    """Protocol for objects that produce control commands each step."""

    def reset(self) -> None:  # pragma: no cover - simple protocol
        ...

    def advance(self) -> torch.Tensor:  # shape context is controller-specific
        ...


class ArmController:
    """Abstract base class for arm controllers.

    Subclasses should implement reset() and step(). The controller is expected
    to perform any computation needed and optionally write targets to the passed
    robot articulation.
    """

    def __init__(self, config: ArmControllerConfig, num_envs: int = 1, device: Optional[str] = None) -> None:
        self.config = config
        self.num_envs = int(num_envs)
        self.device = device or config.device
        self._input_provider: Optional[InputProvider] = None

    def set_input_provider(self, provider: InputProvider) -> None:
        self._input_provider = provider

    def reset(self, robot) -> None:  # robot: isaaclab.assets.Articulation
        raise NotImplementedError

    def step(self, robot, dt: float) -> None:  # robot: isaaclab.assets.Articulation
        raise NotImplementedError

    def close(self) -> None:
        # Hook for controllers that own resources/devices
        pass


