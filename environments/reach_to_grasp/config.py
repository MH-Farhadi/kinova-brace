from dataclasses import dataclass
from typing import Tuple

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets import KINOVA_JACO2_N6S300_CFG


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


@dataclass
class CameraConfig:
    """Camera placement for the demo viewport."""

    eye: Tuple[float, float, float] = (3.5, 0.0, 3.2)
    target: Tuple[float, float, float] = (0.0, 0.0, 0.5)


DEFAULT_SCENE = SceneConfig()
DEFAULT_CAMERA = CameraConfig()


