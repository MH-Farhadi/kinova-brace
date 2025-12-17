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


def add_motion_gen_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add all CLI arguments for motion generation demos.
    
    Consolidates arguments from demo.py and provides sensible defaults
    for motion planning, grasping, object spawning, and robot control.
    
    Args:
        parser: ArgumentParser to add arguments to
        
    Returns:
        The parser with all arguments added
    """
    # Episode and dataset configuration
    parser.add_argument("--num-episodes", type=int, default=3, help="Number of grasp episodes to run")
    parser.add_argument("--target-label", type=str, default=None, help="Optional target object label filter")
    parser.add_argument("--objects-dataset", type=str, nargs="*", default=[], help="Object dataset directories")
    parser.add_argument("--num-objects", type=int, default=1, help="Number of objects to spawn per episode")
    
    # Object spawning bounds
    parser.add_argument("--spawn-min", type=float, nargs=3, default=[0.30, -0.20, 0.02], 
                        help="Spawn AABB min xyz (m) relative to robot origin")
    parser.add_argument("--spawn-max", type=float, nargs=3, default=[0.55, 0.20, 0.05],
                        help="Spawn AABB max xyz (m) relative to robot origin")
    parser.add_argument("--min-distance", type=float, default=0.15, help="Min distance between spawned objects (m)")
    parser.add_argument("--scale-min", type=float, default=None, help="Optional uniform scale min for objects")
    parser.add_argument("--scale-max", type=float, default=None, help="Optional uniform scale max for objects")
    parser.add_argument("--no-objects", action="store_true", help="Skip spawning objects")
    
    # Robot and controller configuration
    parser.add_argument("--ee-link", type=str, default="j2n6s300_end_effector", help="End-effector link name")
    parser.add_argument("--speed", type=float, default=0.4, help="Linear speed (m/s)")
    parser.add_argument("--rot-speed", type=float, default=2.0, help="Angular speed (rad/s)")
    parser.add_argument("--tolerance", type=float, default=0.005, help="Position tolerance for waypoint reaching (m)")
    
    # Motion planning configuration
    parser.add_argument("--pregrasp", type=float, default=0.10, help="Pre-grasp offset distance (m)")
    parser.add_argument("--lift", type=float, default=0.15, help="Lift height after grasp (m)")
    parser.add_argument("--planner", type=str, default="scripted", 
                        choices=["scripted", "rmpflow", "curobo", "lula"], 
                        help="Motion planner to use")
    
    # Grasp pose estimation
    parser.add_argument("--grasp", type=str, default="obb", 
                        choices=["obb", "replicator"], 
                        help="Grasp pose provider algorithm")
    parser.add_argument("--rep-gripper-prim-path", type=str, default=None, 
                        help="Gripper prim path for Replicator grasping")
    parser.add_argument("--rep-config-yaml", type=str, default=None, 
                        help="Replicator grasping YAML config path")
    parser.add_argument("--rep-max-candidates", type=int, default=16,
                        help="Max grasp candidates for Replicator")
    
    # Physics stabilization
    parser.add_argument("--stabilize-steps", type=int, default=500, 
                        help="Simulation steps to wait after spawning for physics stabilization")
    
    # Logging and debugging
    parser.add_argument("--logs-root", type=str, default="logs/assist", help="Root directory for logs")
    parser.add_argument("--print-ee", action="store_true", help="Print end-effector position each step")
    parser.add_argument("--ee-frame", type=str, default="world", 
                        choices=["world", "base"], 
                        help="Frame for EE logging")
    parser.add_argument("--print-interval", type=int, default=1, help="Print every N steps")
    
    # Isaac Sim / AppLauncher arguments
    AppLauncher.add_app_launcher_args(parser)
    
    return parser


# Backward compatibility alias
add_cli_args = add_motion_gen_cli_args


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for motion generation demo CLI."""
    parser = argparse.ArgumentParser(description="Motion generation grasp demo with configurable planners")
    add_motion_gen_cli_args(parser)
    args = parser.parse_args(argv)

    # Delegate to demo, which sets up sim and runs episodes using selected planner
    try:
        # If invoked as a module
        from .demo import run_grasp_loop_demo  # type: ignore[no-redef]
    except Exception:
        # If invoked as a script
        from motion_generation.demo import run_grasp_loop_demo  # type: ignore[no-redef]
    return run_grasp_loop_demo(args)


if __name__ == "__main__":
    raise SystemExit(main())


