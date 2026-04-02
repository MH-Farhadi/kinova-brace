"""Types and enums for control modes."""

from __future__ import annotations

from enum import Enum
from typing import Literal


class ControlMode(Enum):
    """Available control modes for robotic manipulation."""
    
    TRANSLATE = "translate"
    ROTATE = "rotate" 
    GRIPPER = "gripper"
    
    def __str__(self) -> str:
        return self.value


# Type alias for mode literals
ControlModeType = Literal["translate", "rotate", "gripper"] 