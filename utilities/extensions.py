"""Omniverse extension management utilities."""

from __future__ import annotations


def enable_optional_planner_extensions() -> None:
    """Enable optional planner-related extensions if available.
    
    This attempts to enable Isaac Sim motion planning extensions (LULA, etc.)
    that may be required by certain planners. Failures are silently ignored
    as not all environments have these extensions available.
    """
    try:
        import omni.kit.app  # type: ignore
        app = omni.kit.app.get_app()
        ext = app.get_extension_manager()
        for ext_name in [
            "omni.isaac.motion_generation",
            "omni.isaac.motion_planning.lula",
            "omni.isaac.motion_generation.lula",
        ]:
            try:
                if not ext.is_extension_enabled(ext_name):
                    ext.set_extension_enabled_immediate(ext_name, True)
                    print(f"[MG][EXT] Enabled extension: {ext_name}")
            except Exception:
                pass
    except Exception as e:
        print(f"[MG][EXT][WARN] Could not enable LULA extensions automatically: {e}")

