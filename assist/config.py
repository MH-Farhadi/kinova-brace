from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class AssistConfig:
    """Top-level hyperparameters for the assist loop (v0).

    Change these to tune behavior without touching other modules.
    """

    # Timing
    assist_rate_hz: int = 10                    # policy/orchestrator tick rate
    window_len_s: float = 2.0                    # rolling window length

    # Suggestion policy thresholds
    conf_suggest: float = 0.70                   # >= suggest
    conf_clarify_min: float = 0.40               # [min, max) clarify range
    conf_clarify_max: float = 0.70

    # Cooldowns (seconds)
    cooldown_suggest_s: float = 5.0              # after any suggestion
    cooldown_decline_extra_s: float = 3.0        # extra after decline (total 8s default)
    cooldown_cancel_extra_s: float = 7.0         # extra after cancel/safety stop (total 12s default)

    # Intent heuristics
    min_stable_time_s: float = 0.7               # stable approach before suggesting
    max_object_ray_angle_deg: float = 25.0       # projection cone for intent
    max_object_distance_m: float = 0.6           # ignore far objects

    # Action generation
    pregrasp_offset_m: float = 0.05              # back-off distance along approach vector
    align_speed_mps: float = 0.06                # translational alignment speed
    align_rot_speed_rps: float = 0.5             # rotational speed
    grasp_timeout_s: float = 6.0
    retreat_distance_m: float = 0.10

    # Logging
    logs_root: Path = Path("logs/assist")
    log_only_on_suggestion: bool = True          # write only when suggesting/acting

    # Misc
    auto_accept: bool = False                    # testing only; auto-accept suggestions
    verbose: bool = True                         # console prints


def from_cli_args(args) -> AssistConfig:
    """Build config from CLI args if present, falling back to defaults."""
    cfg = AssistConfig()
    # Use hasattr checks to remain robust if flags are missing
    if hasattr(args, "assist_rate_hz") and args.assist_rate_hz is not None:
        cfg.assist_rate_hz = int(args.assist_rate_hz)
    if hasattr(args, "assist_window_s") and args.assist_window_s is not None:
        cfg.window_len_s = float(args.assist_window_s)
    if hasattr(args, "assist_cooldown_s") and args.assist_cooldown_s is not None:
        cfg.cooldown_suggest_s = float(args.assist_cooldown_s)
    if hasattr(args, "assist_auto_accept") and args.assist_auto_accept:
        cfg.auto_accept = True
    if hasattr(args, "assist_verbose") and args.assist_verbose is not None:
        cfg.verbose = bool(args.assist_verbose)
    return cfg


