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
    
    # First pass: parse only the profile argument
    parser_pre = argparse.ArgumentParser(description="Modular data collection entrypoint", add_help=False)
    parser_pre.add_argument("--profile", type=str, default="ticks_v0", choices=sorted(profiles.keys()))
    pre_args, remaining_argv = parser_pre.parse_known_args(argv)
    
    # Get the selected profile
    selected_profile = profiles[str(pre_args.profile)]
    
    # Second pass: add all args from the selected profile
    parser = argparse.ArgumentParser(description="Modular data collection entrypoint")
    parser.add_argument("--profile", type=str, default=pre_args.profile, choices=sorted(profiles.keys()))

    # Add common args (object spawning + controller knobs)
    from scripts.cli import add_demo_cli_args

    add_demo_cli_args(parser)

    # Add args only from the selected profile
    selected_profile.add_cli_args(parser)

    # Isaac AppLauncher args (device, headless, etc)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args(remaining_argv)
    return selected_profile.run(args)


if __name__ == "__main__":
    raise SystemExit(main())


