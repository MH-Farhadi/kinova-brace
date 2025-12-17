from __future__ import annotations

from typing import Dict

from .spec import TaskSpec


def get_tasks() -> Dict[str, TaskSpec]:
    """Registry of available data-collection tasks.

    Import lazily to avoid importing Isaac/Omni deps at module import time.
    """

    # Local imports keep dependency surfaces minimal.
    from .reach_to_grasp.task import TASK as reach_to_grasp_task

    return {
        reach_to_grasp_task.name: reach_to_grasp_task,
    }


