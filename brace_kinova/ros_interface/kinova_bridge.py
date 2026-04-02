"""Bridge between BRACE 2D Cartesian velocity output and Kinova Gen3 ros_kortex commands.

Converts 2D (vx, vy) to 3D Cartesian velocity at fixed Z height,
handles gripper via ros_kortex action server, and enforces workspace safety.

IMPORTANT: Requires ROS 1 (rospy) + ros_kortex driver. Optional import.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import rospy
    from geometry_msgs.msg import TwistStamped
    import actionlib

    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False


class KinovaBridge:
    """Translates BRACE 2D velocity to Kinova Gen3 Cartesian control.

    The Kinova Gen3 ros_kortex package exposes:
    - Cartesian velocity via TwistStamped on /my_gen3/in/cartesian_velocity
    - Gripper via action server /my_gen3/robotiq_2f_85_gripper_controller/gripper_cmd
    """

    DEFAULT_WORKSPACE = {
        "x_min": 0.20, "x_max": 0.60,
        "y_min": -0.45, "y_max": 0.45,
        "z_fixed": 0.90,
    }
    MAX_LINEAR_VEL = 0.15
    MAX_ANGULAR_VEL = 0.0

    def __init__(
        self,
        workspace_bounds: Optional[dict] = None,
        z_fixed: Optional[float] = None,
        max_velocity: float = 0.15,
        velocity_topic: str = "/my_gen3/in/cartesian_velocity",
    ):
        self.workspace = workspace_bounds or self.DEFAULT_WORKSPACE
        self.z_fixed = z_fixed or self.workspace.get("z_fixed", 0.90)
        self.max_velocity = max_velocity
        self.velocity_topic = velocity_topic

        if ROS_AVAILABLE:
            self.vel_pub = rospy.Publisher(
                self.velocity_topic, TwistStamped, queue_size=1
            )
        self._gripper_open = True

    def send_velocity(
        self,
        vx: float,
        vy: float,
        vz: float = 0.0,
    ) -> None:
        """Publish a 3D Cartesian velocity command to the Kinova arm.

        Args:
            vx: X velocity (forward/back).
            vy: Y velocity (left/right).
            vz: Z velocity (normally 0 for planar motion).
        """
        vx = np.clip(vx, -self.max_velocity, self.max_velocity)
        vy = np.clip(vy, -self.max_velocity, self.max_velocity)
        vz = np.clip(vz, -self.max_velocity, self.max_velocity)

        if not ROS_AVAILABLE:
            return

        cmd = TwistStamped()
        cmd.header.stamp = rospy.Time.now()
        cmd.header.frame_id = "base_link"
        cmd.twist.linear.x = float(vx)
        cmd.twist.linear.y = float(vy)
        cmd.twist.linear.z = float(vz)
        cmd.twist.angular.x = 0.0
        cmd.twist.angular.y = 0.0
        cmd.twist.angular.z = 0.0

        self.vel_pub.publish(cmd)

    def send_2d_velocity(self, vx: float, vy: float) -> None:
        """Send a 2D planar velocity command (z=0)."""
        self.send_velocity(vx, vy, 0.0)

    def send_grasp_descent(self, vz: float = -0.05) -> None:
        """Send a downward velocity for final grasp descent."""
        self.send_velocity(0.0, 0.0, vz)

    def open_gripper(self) -> None:
        """Open the Kinova gripper via ros_kortex action server."""
        self._gripper_open = True
        self._send_gripper_command(0.0)

    def close_gripper(self) -> None:
        """Close the Kinova gripper via ros_kortex action server."""
        self._gripper_open = False
        self._send_gripper_command(1.0)

    def _send_gripper_command(self, position: float) -> None:
        """Send gripper command (0=open, 1=closed).

        Uses ros_kortex gripper action server.
        """
        if not ROS_AVAILABLE:
            return

        try:
            from kortex_driver.msg import (
                GripperCommand as KortexGripperCommand,
                Finger,
                GripperMode,
            )

            finger = Finger()
            finger.finger_identifier = 0
            finger.value = position

            gripper_cmd = KortexGripperCommand()
            gripper_cmd.mode = GripperMode()
            gripper_cmd.mode.gripper_mode = 2
            gripper_cmd.gripper.finger.append(finger)

            pub = rospy.Publisher(
                "/my_gen3/robotiq_2f_85_gripper_controller/gripper_cmd/goal",
                KortexGripperCommand,
                queue_size=1,
            )
            pub.publish(gripper_cmd)
        except (ImportError, Exception) as e:
            rospy.logwarn(f"[KinovaBridge] Gripper command failed: {e}")

    def stop(self) -> None:
        """Send zero velocity (emergency stop)."""
        self.send_velocity(0.0, 0.0, 0.0)

    def check_workspace_bounds(self, x: float, y: float) -> tuple[bool, str]:
        """Check if a position is within workspace bounds."""
        if x < self.workspace["x_min"] or x > self.workspace["x_max"]:
            return False, f"X={x:.3f} out of [{self.workspace['x_min']}, {self.workspace['x_max']}]"
        if y < self.workspace["y_min"] or y > self.workspace["y_max"]:
            return False, f"Y={y:.3f} out of [{self.workspace['y_min']}, {self.workspace['y_max']}]"
        return True, "OK"

    def clamp_to_workspace(self, vx: float, vy: float, current_x: float, current_y: float, dt: float = 0.05) -> tuple[float, float]:
        """Clamp velocity so the next position stays within workspace bounds."""
        next_x = current_x + vx * dt
        next_y = current_y + vy * dt

        if next_x < self.workspace["x_min"]:
            vx = max(vx, 0.0)
        elif next_x > self.workspace["x_max"]:
            vx = min(vx, 0.0)

        if next_y < self.workspace["y_min"]:
            vy = max(vy, 0.0)
        elif next_y > self.workspace["y_max"]:
            vy = min(vy, 0.0)

        return vx, vy
