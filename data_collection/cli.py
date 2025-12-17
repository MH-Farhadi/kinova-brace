from __future__ import annotations

import argparse
from typing import Optional

from isaaclab.app import AppLauncher


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
    parser.add_argument("--logs-root", type=str, default="logs/data_collection")
    parser.add_argument(
        "--planner",
        type=str,
        default="scripted",
        choices=["scripted", "rmpflow", "curobo"],
        help="Planner backend to use during data collection.",
    )
    AppLauncher.add_app_launcher_args(parser)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Data collection using motion_generation planners")
    add_cli_args(parser)
    args = parser.parse_args(argv)

    # Delegate to data collection demo
    from .demo import run_data_collection

    return run_data_collection(args)


if __name__ == "__main__":
    raise SystemExit(main())


