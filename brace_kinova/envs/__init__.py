from brace_kinova.envs.reach_grasp_env import ReachGraspEnv
from brace_kinova.envs.scenarios import SCENARIOS, ScenarioConfig
from brace_kinova.envs.wrappers import NormalizeObservation, GoalMaskedObservation
from brace_kinova.envs.isaac_config import IsaacBraceEnvConfig, IsaacBraceSceneConfig

__all__ = [
    "ReachGraspEnv",
    "SCENARIOS",
    "ScenarioConfig",
    "NormalizeObservation",
    "GoalMaskedObservation",
    "IsaacBraceEnvConfig",
    "IsaacBraceSceneConfig",
]


def get_isaac_env_classes():
    """Lazy import of Isaac Sim environment classes.

    These require ``isaaclab.app.AppLauncher`` to have been called first.
    Import this function *after* launching Isaac Sim::

        from brace_kinova.envs import get_isaac_env_classes
        IsaacReachGraspEnv, IsaacExpertReachGraspEnv = get_isaac_env_classes()
    """
    from brace_kinova.envs.isaac_env import (
        IsaacReachGraspEnv,
        IsaacExpertReachGraspEnv,
    )
    return IsaacReachGraspEnv, IsaacExpertReachGraspEnv
