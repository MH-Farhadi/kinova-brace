from __future__ import annotations

from typing import List, Optional

import torch

from controllers.base import InputProvider


class CommandMuxInputProvider(InputProvider):
    """Multiplex base input with an injected action command stream.

    - When idle, returns base_provider.advance().
    - When an action is active, returns the next precomputed command tensor.
    """

    def __init__(self, base_provider: Optional[InputProvider] = None) -> None:
        self._base = base_provider
        self._action_stream: List[torch.Tensor] = []
        self._cursor: int = 0
        self._last_cmd: Optional[torch.Tensor] = None

    def set_base(self, base_provider: Optional[InputProvider]) -> None:
        self._base = base_provider

    def reset(self) -> None:
        if self._base is not None:
            self._base.reset()
        self._action_stream = []
        self._cursor = 0
        self._last_cmd = None

    def run_action(self, stream: List[torch.Tensor]) -> None:
        self._action_stream = stream
        self._cursor = 0

    def cancel_action(self) -> None:
        self._action_stream = []
        self._cursor = 0

    def is_action_active(self) -> bool:
        return self._cursor < len(self._action_stream)

    def advance(self) -> torch.Tensor:
        if self.is_action_active():
            cmd = self._action_stream[self._cursor]
            self._cursor += 1
            self._last_cmd = cmd
            return cmd
        if self._base is None:
            cmd = torch.zeros(1, 6)
            self._last_cmd = cmd
            return cmd
        cmd = self._base.advance()
        self._last_cmd = cmd
        return cmd

    @property
    def last_cmd(self) -> Optional[torch.Tensor]:
        return self._last_cmd


