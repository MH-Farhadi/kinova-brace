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

# Isaac's bundled packages may import `cv2.utils` early, which collides with this repo's `utils/` package.
# If `utils` is already loaded from a non-repo location, purge it so imports resolve to our package.
_utils_mod = sys.modules.get("utils")
if _utils_mod is not None:
    _utils_file = str(getattr(_utils_mod, "__file__", "") or "")
    if _utils_file and root_str not in _utils_file:
        for _k in list(sys.modules.keys()):
            if _k == "utils" or _k.startswith("utils."):
                del sys.modules[_k]


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

    parser.add_argument(
        "--suppress-spam",
        action="store_true",
        help=(
            "Deprecated/temporary no-op. Previous attempts to change Kit log/notification settings here can break Kit startup. "
            "Kept for CLI compatibility."
        ),
    )
    args = parser.parse_args(remaining_argv)

    if getattr(args, "suppress_spam", False):
        # NOTE: Do not attempt to touch carb/omni settings here.
        # At this point Kit may not be initialized; writing settings can corrupt Kit's internal dictionaries and crash startup.
        print("[collect_data][WARN] --suppress-spam is currently a no-op (kept for compatibility).")
    return selected_profile.run(args)


if __name__ == "__main__":
    raise SystemExit(main())


