from __future__ import annotations

from .base import PlannerContext, BasePlanner
from .scripted import ScriptedPlanner
from .rmpflow import RmpFlowPlanner
from .curobo import CuroboPlanner
from .lula import LulaPlanner


def create_planner(kind: str, *, ctx: PlannerContext) -> BasePlanner:
    kind_l = (kind or "").lower()
    if kind_l == "rmpflow":
        return RmpFlowPlanner(ctx)
    if kind_l == "curobo":
        return CuroboPlanner(ctx)
    if kind_l in ("lula", "lula_cspace", "lula-cspace"):
        return LulaPlanner(ctx)
    if kind_l == "scripted" or kind_l == "":
        return ScriptedPlanner(ctx)
    print(f"[MG][PLANNER][WARN] Unknown planner type '{kind}'. Falling back to 'scripted'.")
    return ScriptedPlanner(ctx)


