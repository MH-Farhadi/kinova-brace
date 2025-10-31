from __future__ import annotations

import argparse


def add_demo_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register all CLI arguments for the Kinova demo and return the parser.

    Centralizes defaults for object spawning and controller behavior.
    """
    # Robot/controller
    parser.add_argument("--ee_link", type=str, default="j2n6s300_end_effector", help="End-effector link name")
    parser.add_argument("--speed", type=float, default=0.20, help="Linear speed (m/s)")
    parser.add_argument("--rot-speed", type=float, default=0.5, help="Angular speed (rad/s)")

    # Object spawning (Nucleus-only)
    parser.add_argument("--num-objects", type=int, default=4, help="Number of random objects to spawn")
    parser.add_argument(
        "--spawn-min",
        type=float,
        nargs=3,
        default=[0.2, -0.3, 0.9],
        help="Spawn AABB min xyz (m) relative to /World/Origin1",
    )
    parser.add_argument(
        "--spawn-max",
        type=float,
        nargs=3,
        default=[0.60, 0.45, 1.05],
        help="Spawn AABB max xyz (m) relative to /World/Origin1",
    )
    parser.add_argument("--min-distance", type=float, default=0.1, help="Min distance between objects (m)")
    parser.add_argument("--scale-min", type=float, default=None, help="Optional uniform scale min for objects")
    parser.add_argument("--scale-max", type=float, default=None, help="Optional uniform scale max for objects")
    # Physics-related controls are now centralized in environments/physix.py
    parser.add_argument("--no-objects", action="store_true", help="Skip spawning objects")

    # Logging
    parser.add_argument("--print-ee", action="store_true", help="Print EE XYZ each step")
    parser.add_argument("--ee-frame", type=str, default="world", choices=["world", "base"], help="EE logging frame")
    parser.add_argument("--print-interval", type=int, default=1, help="Print every N steps")

    # Assist layer (v0)
    parser.add_argument("--assist", action="store_true", help="Enable assist suggestion and actions")
    parser.add_argument("--assist-rate-hz", type=int, default=10, help="Assist tick rate in Hz (default 10)")
    parser.add_argument("--assist-window-s", type=float, default=2.0, help="Rolling window length in seconds (default 2.0)")
    parser.add_argument("--assist-cooldown-s", type=float, default=5.0, help="Base cooldown between suggestions (default 5s)")
    parser.add_argument("--assist-auto-accept", action="store_true", help="Auto-accept suggestions (testing)")
    parser.add_argument("--assist-verbose", action="store_true", help="Verbose console logging for assist")

    return parser


