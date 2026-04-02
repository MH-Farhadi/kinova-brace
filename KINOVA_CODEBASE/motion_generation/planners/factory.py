from __future__ import annotations

from .base import PlannerContext, BasePlanner
from .scripted import ScriptedPlanner
from .rmpflow import RmpFlowPlanner
from .lula import LulaPlanner
from .curobo_v2 import CuroboV2Planner


def create_planner(kind: str, *, ctx: PlannerContext) -> BasePlanner:
    kind_l = (kind or "").lower()
    if kind_l in ("curobo_vla", "curobo-vla", "curobo_v2", "curobo-v2"):
        return CuroboV2Planner(ctx)
    if kind_l == "rmpflow":
        return RmpFlowPlanner(ctx)
    if kind_l == "curobo":
        # Deprecated: alias 'curobo' to 'curobo_v2' to avoid breaking existing configs.
        # Previously this pointed to a stub, now we point to the real implementation.
        return CuroboV2Planner(ctx)
    if kind_l in ("lula", "lula_cspace", "lula-cspace"):
        return LulaPlanner(ctx)
    if kind_l == "scripted" or kind_l == "":
        return ScriptedPlanner(ctx)
    print(f"[MG][PLANNER][WARN] Unknown planner type '{kind}'. Falling back to 'scripted'.")
    return ScriptedPlanner(ctx)


