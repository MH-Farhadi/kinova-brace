from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

# Ensure kinova-isaac root is first on sys.path so our top-level packages (e.g. `environments/`)
# win over any Isaac/Omni modules with the same name.
ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str in sys.path:
    sys.path.remove(root_str)
sys.path.insert(0, root_str)

# If a non-package `environments` module gets preloaded, it will break `environments.*` imports.
_env_mod = sys.modules.get("environments")
if _env_mod is not None and not hasattr(_env_mod, "__path__"):
    del sys.modules["environments"]

from data_collection.tasks.registry import get_tasks


def main(argv: Optional[list[str]] = None) -> int:
    tasks = get_tasks()
    parser = argparse.ArgumentParser(description="Data collection task runner")
    subparsers = parser.add_subparsers(dest="task", required=False)

    # Default task if user doesn't provide a subcommand
    default_task = "reach_to_grasp"

    for name, spec in tasks.items():
        sp = subparsers.add_parser(name, help=f"Run task: {name}")
        spec.add_cli_args(sp)
        sp.set_defaults(_run_fn=spec.run)

    # Parse
    args = parser.parse_args(argv)
    if getattr(args, "_run_fn", None) is None:
        # No subcommand provided → run default task
        spec = tasks[default_task]
        sp = argparse.ArgumentParser(description=f"Data collection task runner ({default_task})")
        spec.add_cli_args(sp)
        args2 = sp.parse_args(argv)
        return spec.run(args2)

    return args._run_fn(args)  # type: ignore[attr-defined]


if __name__ == "__main__":
    raise SystemExit(main())


