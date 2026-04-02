from dataclasses import dataclass
from typing import Tuple

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets import KINOVA_JACO2_N6S300_CFG
import numpy as np


BASE_ROBOT_CFG = KINOVA_JACO2_N6S300_CFG


@dataclass
class SceneConfig:
    """High-level scene configuration for the JACO2 reach-to-grasp demo."""

    num_origins: int = 1
    origin_spacing: float = 2.0

    table_usd_path: str = (
        f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd"
    )
    table_scale: Tuple[float, float, float] = (1.5, 2.0, 1.0)
    table_translation: Tuple[float, float, float] = (0.0, 0.0, 0.8)

    robot_prim_path: str = "/World/Origin1/Robot"
    robot_base_height: float = 0.8

    # Optional: initial joint configuration (by joint name or regex)
    # Example: {"j2n6s300_joint_[1-6]": 0.0} or per-joint mapping
    robot_default_joint_pos: dict[str, float] | None = None


@dataclass
class CameraConfig:
    """Camera placement for the demo viewport."""

    eye: Tuple[float, float, float] = (3.5, 0.0, 3.2)
    target: Tuple[float, float, float] = (0.0, 0.0, 0.5)


@dataclass
class TopDownCameraConfig:
    """Top-down camera configuration for VLA environment.
    
    To adjust the camera view:
    - position: (x, y, z) - Camera location. Higher z = more overhead view.
      Default: (0.0, 0.0, 2.0) = directly above robot base at 2.0m height
    - target: (x, y, z) - Point the camera looks at. 
      Default: (0.4, 0.0, 0.8) = workspace center at table surface height
    - fov: Field of view in degrees. Higher = wider view, lower = zoomed in.
      Default: 70.0 degrees for good workspace coverage
    """

    prim_path: str = "/World/Origin1/TopDownCamera"
    position: Tuple[float, float, float] = (0.4, 0.0, 4.0)  # Directly above table center, ~1.7m above table surface for wider overview
    target: Tuple[float, float, float] = (0.4, 0.0, 0.8)  # Looking at workspace center at table surface (table top at z=0.8)
    resolution: Tuple[int, int] = (640, 640)
    fov: float = 65.0  # degrees - wider FOV to see more of the workspace and robot


DEFAULT_SCENE = SceneConfig(
    robot_default_joint_pos={
        "j2n6s300_joint_1": 0*np.pi, # j2n6s300_joint_1: [-inf, inf]
        "j2n6s300_joint_2": np.pi, # j2n6s300_joint_2: [0.820, 5.463]
        "j2n6s300_joint_3": 1.8*np.pi, # j2n6s300_joint_3: [0.332, 5.952]
        "j2n6s300_joint_4": 0*np.pi, # j2n6s300_joint_4: [-inf, inf]
        "j2n6s300_joint_5": 1.75*np.pi, # j2n6s300_joint_5: [-inf, inf]
        "j2n6s300_joint_6": 0.5*np.pi, # j2n6s300_joint_6: [-inf, inf]
        "j2n6s300_joint_finger_1": 0.0, # j2n6s300_joint_finger_1: [0.000, 1.510]
        "j2n6s300_joint_finger_2": 0.0, # j2n6s300_joint_finger_2: [0.000, 1.510]
        "j2n6s300_joint_finger_3": 0.0, # j2n6s300_joint_finger_3: [0.000, 1.510]
        "j2n6s300_joint_finger_tip_1": 0.0, # j2n6s300_joint_finger_tip_1: [0.000, 2.000]
        "j2n6s300_joint_finger_tip_2": 0.0, # j2n6s300_joint_finger_tip_2: [0.000, 2.000]
        "j2n6s300_joint_finger_tip_3": 0.0, # j2n6s300_joint_finger_tip_3: [0.000, 2.000]
        # You can also use regex like "j2n6s300_joint_[1-6]": 0.0
    }
)

DEFAULT_CAMERA = CameraConfig()
DEFAULT_TOP_DOWN_CAMERA = TopDownCameraConfig()


