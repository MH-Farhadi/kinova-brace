from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional


# Ensure kinova-isaac root is first on sys.path (avoid collisions with Isaac/Omni modules).
ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str in sys.path:
    sys.path.remove(root_str)
sys.path.insert(0, root_str)
_env_mod = sys.modules.get("environments")
if _env_mod is not None and not hasattr(_env_mod, "__path__"):
    del sys.modules["environments"]


def main(argv: Optional[list[str]] = None) -> int:
    from data_collection.profiles.registry import get_profiles

    profiles = get_profiles()
    parser = argparse.ArgumentParser(description="Modular data collection entrypoint")
    parser.add_argument("--profile", type=str, default="ticks_v0", choices=sorted(profiles.keys()))

    # Add common args (object spawning + controller knobs)
    from scripts.cli import add_demo_cli_args

    add_demo_cli_args(parser)

    # Add profile args (includes --env etc)
    profiles["ticks_v0"].add_cli_args(parser)

    # Isaac AppLauncher args (device, headless, etc)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args(argv)
    return profiles[str(args.profile)].run(args)


if __name__ == "__main__":
    raise SystemExit(main())


