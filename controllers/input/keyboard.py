"""Keyboard input provider with mode switching capabilities."""

from __future__ import annotations

from typing import Callable

from ..base import InputProvider


class Se3KeyboardInput(InputProvider):
    """Wrapper around IsaacLab's Se3Keyboard with mode switching callbacks."""

    def __init__(self, pos_sensitivity_per_step: float, rot_sensitivity_rad_per_step: float) -> None:
        from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg

        self._kb = Se3Keyboard(
            Se3KeyboardCfg(
                pos_sensitivity=pos_sensitivity_per_step,
                rot_sensitivity=rot_sensitivity_rad_per_step,
                gripper_term=True,
            )
        )

    def reset(self) -> None:
        """Reset the keyboard input state."""
        self._kb.reset()

    def advance(self):
        """Get the current SE(3) command from keyboard input."""
        return self._kb.advance()

    def add_mode_callbacks(self, translate_fn: Callable, rotate_fn: Callable, gripper_fn: Callable) -> None:
        """Add keyboard callbacks for mode switching.
        
        Args:
            translate_fn: Function to call when switching to translate mode
            rotate_fn: Function to call when switching to rotate mode  
            gripper_fn: Function to call when switching to gripper mode
        """
        try:
            for k in ["I", "i"]:
                self._kb.add_callback(k, translate_fn)  # type: ignore[attr-defined]
            for k in ["O", "o"]:
                self._kb.add_callback(k, rotate_fn)     # type: ignore[attr-defined]
            for k in ["P", "p"]:
                self._kb.add_callback(k, gripper_fn)    # type: ignore[attr-defined]
        except Exception:
            print("[WARN]: Failed to add keyboard mode callbacks") 