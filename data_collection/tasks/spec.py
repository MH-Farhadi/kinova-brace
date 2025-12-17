from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


AddArgsFn = Callable[["argparse.ArgumentParser"], None]
RunFn = Callable[["argparse.Namespace"], int]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    add_cli_args: AddArgsFn
    run: RunFn


