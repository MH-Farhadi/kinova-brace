"""Mode manager for handling control mode transitions."""

from __future__ import annotations

from typing import Callable, Optional

from .types import ControlMode, ControlModeType


class ModeManager:
    """Manages control mode transitions and validation."""
    
    def __init__(self, initial_mode: ControlModeType = "translate") -> None:
        self._current_mode = ControlMode(initial_mode)
        self._mode_change_callback: Optional[Callable[[ControlMode], None]] = None
        
    @property
    def current_mode(self) -> ControlMode:
        """Get the current control mode."""
        return self._current_mode
        
    def set_mode_change_callback(self, callback: Callable[[ControlMode], None]) -> None:
        """Set a callback function to be called when mode changes.
        
        Args:
            callback: Function that takes the new mode as argument
        """
        self._mode_change_callback = callback
        
    def switch_to(self, mode: ControlModeType) -> None:
        """Switch to a new control mode.
        
        Args:
            mode: The target mode to switch to
        """
        new_mode = ControlMode(mode)
        if new_mode != self._current_mode:
            self._current_mode = new_mode
            print(f"[MODE] Switched to: {self._current_mode}")
            if self._mode_change_callback:
                self._mode_change_callback(self._current_mode)
                
    def is_mode(self, mode: ControlModeType) -> bool:
        """Check if currently in a specific mode.
        
        Args:
            mode: The mode to check against
            
        Returns:
            True if currently in the specified mode
        """
        return self._current_mode == ControlMode(mode)
        
    def get_mode_callbacks(self) -> tuple[Callable, Callable, Callable]:
        """Get callback functions for mode switching via input devices.
        
        Returns:
            Tuple of (translate_fn, rotate_fn, gripper_fn) callbacks
        """
        translate_fn = lambda: self.switch_to("translate")
        rotate_fn = lambda: self.switch_to("rotate") 
        gripper_fn = lambda: self.switch_to("gripper")
        return translate_fn, rotate_fn, gripper_fn 