from __future__ import annotations

from .base import PlannerContext, BasePlanner
from .scripted import ScriptedPlanner
from .rmpflow import RmpFlowPlanner
from .lula import LulaPlanner
from .curobo_vla import CuroboVLAPlanner


def create_planner(kind: str, *, ctx: PlannerContext) -> BasePlanner:
    kind_l = (kind or "").lower()
    if kind_l in ("curobo_vla", "curobo-vla", "curobo_vla_vla", "curobo-vla-vla"):
        return CuroboVLAPlanner(ctx)
    if kind_l == "rmpflow":
        return RmpFlowPlanner(ctx)
    if kind_l == "curobo":
        # Import lazily so missing file / missing deps don't break unrelated planners.
        try:
            from .curobo import CuroboPlanner  # type: ignore
        except Exception as e:
            print(f"[MG][PLANNER][WARN] Failed to create curobo planner ({e}); falling back to scripted.")
            return ScriptedPlanner(ctx)
        return CuroboPlanner(ctx)
    if kind_l in ("lula", "lula_cspace", "lula-cspace"):
        return LulaPlanner(ctx)
    if kind_l == "scripted" or kind_l == "":
        return ScriptedPlanner(ctx)
    print(f"[MG][PLANNER][WARN] Unknown planner type '{kind}'. Falling back to 'scripted'.")
    return ScriptedPlanner(ctx)


