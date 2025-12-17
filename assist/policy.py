from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .config import AssistConfig
from .schemas import DetectedObject, RollingWindow


@dataclass
class PolicyOutput:
    intent: Optional[str]
    confidence: float
    text: Optional[str]
    kind: str  # "suggest" | "clarify" | "defer"
    top_object_id: Optional[str]
    alt_object_id: Optional[str]


def _unit(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n < 1e-6:
        return (0.0, 0.0, 0.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def _dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _vec_to(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (b[0] - a[0], b[1] - a[1], b[2] - a[2])


def _distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    d = _vec_to(a, b)
    return math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])


def _stable_approach_score(win: RollingWindow, obj: DetectedObject, horizon_s: float) -> float:
    """Return fraction of samples in the last horizon where distance decreased and direction aligned."""
    if len(win.robot_states) < 2 or len(win.user_inputs) < 1:
        return 0.0
    # Pick subset within horizon
    t_cut = win.robot_states[-1].t_ms - int(horizon_s * 1000)
    pairs = [(rs, ui) for rs, ui in zip(win.robot_states, win.user_inputs) if rs.t_ms >= t_cut]
    if len(pairs) < 2:
        return 0.0
    agree = 0
    total = 0
    last_dist = None
    for (rs, ui) in pairs:
        ee = rs.ee_pose.position_m
        dist = _distance(ee, obj.pose.position_m)
        v_cmd = (ui.cartesian_vel_cmd[0], ui.cartesian_vel_cmd[1], ui.cartesian_vel_cmd[2])
        v_dir = _unit(v_cmd)
        dir_to_obj = _unit(_vec_to(ee, obj.pose.position_m))
        cosang = _dot(v_dir, dir_to_obj)
        if last_dist is not None:
            if dist <= last_dist and cosang > math.cos(math.radians(35.0)):
                agree += 1
            total += 1
        last_dist = dist
    return float(agree) / float(total) if total > 0 else 0.0


def infer_policy(cfg: AssistConfig, win: RollingWindow) -> PolicyOutput:
    """Heuristic suggestion policy.

    - Estimate best object by ray alignment and distance
    - Build confidence from angle, distance, and stability
    - Return suggest/clarify/defer with short text
    """
    if len(win.robot_states) == 0 or len(win.user_inputs) == 0 or len(win.objects) == 0:
        return PolicyOutput(intent=None, confidence=0.0, text=None, kind="defer", top_object_id=None, alt_object_id=None)

    rs = win.robot_states[-1]
    ui = win.user_inputs[-1]
    ee = rs.ee_pose.position_m
    v_cmd = (ui.cartesian_vel_cmd[0], ui.cartesian_vel_cmd[1], ui.cartesian_vel_cmd[2])
    v_dir = _unit(v_cmd)

    # Score objects by angle and distance
    scored: List[Tuple[str, float, float, float]] = []  # (id, cosang, dist, stability)
    for obj in win.objects:
        dist = _distance(ee, obj.pose.position_m)
        if dist > cfg.max_object_distance_m:
            continue
        dir_to_obj = _unit(_vec_to(ee, obj.pose.position_m))
        cosang = max(-1.0, min(1.0, _dot(v_dir, dir_to_obj)))
        stability = _stable_approach_score(win, obj, cfg.min_stable_time_s)
        scored.append((obj.id, cosang, dist, stability))

    if not scored:
        return PolicyOutput(intent=None, confidence=0.0, text=None, kind="defer", top_object_id=None, alt_object_id=None)

    scored.sort(key=lambda x: (x[1], -x[2], x[3]), reverse=True)  # prefer high cosang, short dist, stable
    top = scored[0]
    alt = scored[1] if len(scored) > 1 else None

    # Confidence heuristic
    cosang = top[1]
    dist = top[2]
    stability = top[3]
    angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
    angle_score = max(0.0, 1.0 - (angle_deg / cfg.max_object_ray_angle_deg))
    dist_score = max(0.0, 1.0 - (dist / cfg.max_object_distance_m))
    conf = max(0.0, min(1.0, 0.5 * angle_score + 0.3 * dist_score + 0.2 * stability))

    # Decide kind
    if conf >= cfg.conf_suggest:
        intent = f"align_{top[0]}"
        text = f"Looks like you're aligning with {top[0]}. Want help? Yes/No."
        return PolicyOutput(intent=intent, confidence=conf, text=text, kind="suggest", top_object_id=top[0], alt_object_id=None)

    if cfg.conf_clarify_min <= conf < cfg.conf_clarify_max and alt is not None:
        text = f"Do you mean {top[0]} or {alt[0]}? (A/B or 1/2)"
        return PolicyOutput(intent=None, confidence=conf, text=text, kind="clarify", top_object_id=top[0], alt_object_id=alt[0])

    return PolicyOutput(intent=None, confidence=conf, text=None, kind="defer", top_object_id=None, alt_object_id=None)


