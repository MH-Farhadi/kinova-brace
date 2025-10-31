from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .config import AssistConfig
from .schemas import RollingWindow


class JsonlLogger:
    """Simple JSONL session logger for suggestions and actions."""

    def __init__(self, cfg: AssistConfig, session_name: str | None = None) -> None:
        self.cfg = cfg
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_name = session_name or f"session_{ts}"
        self.root = Path(cfg.logs_root) / self.session_name
        self.root.mkdir(parents=True, exist_ok=True)
        self.file = (self.root / "events.jsonl").open("a", buffering=1)

    def close(self) -> None:
        try:
            self.file.close()
        except Exception:
            pass

    def log(self, event_type: str, payload: Dict[str, Any]) -> None:
        record = {
            "t_ms": int(time.time() * 1000),
            "type": event_type,
            "data": payload,
        }
        try:
            self.file.write(json.dumps(record) + "\n")
        except Exception:
            # Fallback to print if disk issue
            print(f"[ASSIST-LOG] {record}")

    @staticmethod
    def summarize_window(win: RollingWindow) -> Dict[str, Any]:
        # Compact snapshot with last robot/user and current objects
        robot_last = None
        if len(win.robot_states) > 0:
            r = win.robot_states[-1]
            robot_last = {
                "t_ms": r.t_ms,
                "ee_pose": {
                    "position_m": list(r.ee_pose.position_m),
                    "orientation_wxyz": list(r.ee_pose.orientation_wxyz),
                },
                "ee_linear_vel_mps": list(r.ee_linear_vel_mps),
                "ee_angular_vel_rps": list(r.ee_angular_vel_rps),
                "mode": r.mode,
            }
        user_last = None
        if len(win.user_inputs) > 0:
            u = win.user_inputs[-1]
            user_last = {
                "t_ms": u.t_ms,
                "cartesian_vel_cmd": list(u.cartesian_vel_cmd),
                "mode": u.mode,
            }
        objs = [
            {
                "id": o.id,
                "label": o.label,
                "pose": {
                    "position_m": list(o.pose.position_m),
                    "orientation_wxyz": list(o.pose.orientation_wxyz),
                },
                "confidence": o.confidence,
            }
            for o in win.objects
        ]
        return {
            "window_ms": win.window_ms,
            "robot_last": robot_last,
            "user_last": user_last,
            "objects": objs,
        }


