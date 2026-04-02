"""DualSense PS5 controller interface for BRACE.

Two backends:
A. pygame-based (for simulation / user studies without ROS)
B. ROS 1-based (for real-world Kinova deployment via /joy topic)

Axis/button mapping (SDL2/pygame):
    Axis 0: Left stick X  → EE X velocity
    Axis 1: Left stick Y  → EE Y velocity
    Axis 4: L2 trigger    → Decrease gamma (manual mode)
    Axis 5: R2 trigger    → Increase gamma (manual mode)
    Button 0: Square      → Reset position
    Button 1: Cross       → Confirm / Start trial
    Button 2: Circle      → Cancel / Skip
    Button 3: Triangle    → Toggle mode
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DualSenseState:
    """Current state of the DualSense controller."""

    lx: float = 0.0
    ly: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    l2: float = 0.0
    r2: float = 0.0
    square: bool = False
    cross: bool = False
    circle: bool = False
    triangle: bool = False
    connected: bool = False


class DualSenseInput:
    """Pygame-based DualSense PS5 controller handler.

    Supports 5 control modes matching the user study:
        0.0     → No assistance (human only, gamma=0)
        0.5     → IDA-style (binary intervention)
        1.0     → Full autonomy (expert only, gamma=1)
        "manual" → User-controlled gamma via L2/R2 triggers
        "ai"    → BRACE AI-predicted gamma
    """

    AXIS_LX = 0
    AXIS_LY = 1
    AXIS_RX = 2
    AXIS_RY = 3
    AXIS_L2 = 4
    AXIS_R2 = 5

    BTN_SQUARE = 0
    BTN_CROSS = 1
    BTN_CIRCLE = 2
    BTN_TRIANGLE = 3

    def __init__(
        self,
        deadzone: float = 0.1,
        max_speed: float = 1.0,
        gamma_adjust_rate: float = 0.003,
    ):
        self.deadzone = deadzone
        self.max_speed = max_speed
        self.gamma_adjust_rate = gamma_adjust_rate

        self._joystick = None
        self._state = DualSenseState()
        self._pygame_init = False

    def init(self) -> bool:
        """Initialize pygame joystick system and connect to DualSense."""
        try:
            import pygame

            if not self._pygame_init:
                pygame.init()
                pygame.joystick.init()
                self._pygame_init = True

            if pygame.joystick.get_count() > 0:
                self._joystick = pygame.joystick.Joystick(0)
                self._joystick.init()
                self._state.connected = True
                return True

            self._state.connected = False
            return False
        except Exception:
            self._state.connected = False
            return False

    def update(self) -> DualSenseState:
        """Poll the controller and update the state."""
        if not self._state.connected or self._joystick is None:
            self._try_reconnect()
            return self._state

        import pygame
        pygame.event.pump()

        try:
            raw_lx = self._joystick.get_axis(self.AXIS_LX)
            raw_ly = self._joystick.get_axis(self.AXIS_LY)

            if abs(raw_lx) < self.deadzone and abs(raw_ly) < self.deadzone:
                self._state.lx = 0.0
                self._state.ly = 0.0
            else:
                self._state.lx = raw_lx
                self._state.ly = raw_ly

            if self._joystick.get_numaxes() >= 6:
                self._state.l2 = self._joystick.get_axis(self.AXIS_L2)
                self._state.r2 = self._joystick.get_axis(self.AXIS_R2)

            if self._joystick.get_numbuttons() >= 4:
                self._state.square = self._joystick.get_button(self.BTN_SQUARE)
                self._state.cross = self._joystick.get_button(self.BTN_CROSS)
                self._state.circle = self._joystick.get_button(self.BTN_CIRCLE)
                self._state.triangle = self._joystick.get_button(self.BTN_TRIANGLE)

        except Exception:
            self._state.connected = False

        return self._state

    def get_velocity(self) -> np.ndarray:
        """Get XY velocity command from left stick, scaled by max_speed."""
        return np.array(
            [self._state.lx * self.max_speed, self._state.ly * self.max_speed],
            dtype=np.float32,
        )

    def adjust_gamma(self, current_gamma: float) -> float:
        """Adjust gamma based on L2/R2 trigger values (manual mode)."""
        gamma = current_gamma
        if self._state.l2 > 0.1:
            gamma = max(0.0, gamma - self.gamma_adjust_rate)
        if self._state.r2 > 0.1:
            gamma = min(1.0, gamma + self.gamma_adjust_rate)
        return gamma

    def _try_reconnect(self) -> None:
        """Attempt to reconnect to a controller."""
        try:
            import pygame
            pygame.joystick.quit()
            pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                self._joystick = pygame.joystick.Joystick(0)
                self._joystick.init()
                self._state.connected = True
        except Exception:
            pass

    def close(self) -> None:
        """Cleanup pygame resources."""
        if self._pygame_init:
            import pygame
            pygame.joystick.quit()
            self._pygame_init = False
            self._state.connected = False


class DualSenseROSInterface:
    """ROS 1-based DualSense handler reading from /joy topic.

    Wraps sensor_msgs/Joy messages into the same DualSenseState format.
    """

    def __init__(
        self,
        deadzone: float = 0.1,
        max_speed: float = 0.15,
        gamma_adjust_rate: float = 0.003,
        joy_topic: str = "/joy",
    ):
        self.deadzone = deadzone
        self.max_speed = max_speed
        self.gamma_adjust_rate = gamma_adjust_rate
        self._state = DualSenseState()

        try:
            import rospy
            from sensor_msgs.msg import Joy

            self._sub = rospy.Subscriber(joy_topic, Joy, self._joy_callback, queue_size=1)
            self._state.connected = True
        except ImportError:
            raise RuntimeError("ROS 1 (rospy) required for DualSenseROSInterface")

    def _joy_callback(self, msg) -> None:
        if len(msg.axes) >= 2:
            raw_lx = msg.axes[0]
            raw_ly = msg.axes[1]

            if abs(raw_lx) < self.deadzone and abs(raw_ly) < self.deadzone:
                self._state.lx = 0.0
                self._state.ly = 0.0
            else:
                self._state.lx = raw_lx
                self._state.ly = raw_ly

        if len(msg.axes) >= 6:
            self._state.l2 = msg.axes[4]
            self._state.r2 = msg.axes[5]

        if len(msg.buttons) >= 4:
            self._state.square = bool(msg.buttons[0])
            self._state.cross = bool(msg.buttons[1])
            self._state.circle = bool(msg.buttons[2])
            self._state.triangle = bool(msg.buttons[3])

        self._state.connected = True

    @property
    def state(self) -> DualSenseState:
        return self._state

    def get_velocity(self) -> np.ndarray:
        return np.array(
            [self._state.lx * self.max_speed, self._state.ly * self.max_speed],
            dtype=np.float32,
        )

    def adjust_gamma(self, current_gamma: float) -> float:
        gamma = current_gamma
        if self._state.l2 > 0.1:
            gamma = max(0.0, gamma - self.gamma_adjust_rate)
        if self._state.r2 > 0.1:
            gamma = min(1.0, gamma + self.gamma_adjust_rate)
        return gamma
