from __future__ import annotations

__all__ = [
    "grasp_estimation",
    "planners",
    "MotionGenerationAgent",
]


def __getattr__(name: str):
    # Lazy import to avoid pulling heavy deps (and to reduce chance of import-time collisions).
    if name == "MotionGenerationAgent":
        from .mogen import MotionGenerationAgent as _MotionGenerationAgent

        return _MotionGenerationAgent
    raise AttributeError(name)


