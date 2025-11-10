from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, Tuple

from isaaclab.app import AppLauncher

# Allow running as a script: ensure package root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--target-label", type=str, default=None)
    parser.add_argument("--objects-dataset", type=str, nargs="*", default=[])
    parser.add_argument("--num-objects", type=int, default=1)
    parser.add_argument("--spawn-min", type=float, nargs=3, default=[0.30, -0.20, 0.02])
    parser.add_argument("--spawn-max", type=float, nargs=3, default=[0.55, 0.20, 0.05])
    parser.add_argument("--pregrasp", type=float, default=0.10)
    parser.add_argument("--lift", type=float, default=0.15)
    parser.add_argument("--speed", type=float, default=0.20)
    parser.add_argument("--tolerance", type=float, default=0.005)
    parser.add_argument("--logs-root", type=str, default="logs/assist")
    parser.add_argument("--planner", type=str, default="scripted", choices=["scripted", "rmpflow", "curobo", "lula"], help="Planner to demo")
    # Grasp provider options
    parser.add_argument("--grasp", type=str, default="aabb", choices=["aabb", "replicator"], help="Grasp pose provider")
    parser.add_argument("--rep-gripper-prim-path", type=str, default=None, help="Gripper prim path for Replicator grasping")
    parser.add_argument("--rep-config-yaml", type=str, default=None, help="Replicator grasping YAML config path")
    AppLauncher.add_app_launcher_args(parser)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Scripted reach-to-grasp data collection")
    add_cli_args(parser)
    args = parser.parse_args(argv)

    # Delegate to planner demo, which sets up sim and runs episodes using selected planner
    try:
        # If invoked as a module
        from .demo import run_motion_planner_demo  # type: ignore[no-redef]
    except Exception:
        # If invoked as a script
        from motion_generation.demo import run_motion_planner_demo  # type: ignore[no-redef]
    return run_motion_planner_demo(args)


if __name__ == "__main__":
    raise SystemExit(main())


