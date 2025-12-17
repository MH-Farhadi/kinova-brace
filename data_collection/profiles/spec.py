from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


AddArgsFn = Callable[["argparse.ArgumentParser"], None]
RunFn = Callable[["argparse.Namespace"], int]


@dataclass(frozen=True)
class ProfileSpec:
    """A collection profile: defines recorded structure + collection loop implementation."""

    name: str
    add_cli_args: AddArgsFn
    run: RunFn


