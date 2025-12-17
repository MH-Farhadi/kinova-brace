from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from .config import AssistConfig


@dataclass
class SuggestionState:
    last_suggest_ms: int = 0
    pending_kind: Optional[str] = None  # "suggest" | "clarify"
    pending_text: Optional[str] = None
    pending_top_id: Optional[str] = None
    pending_alt_id: Optional[str] = None


class DialogueManager:
    """Manages cooldowns and user responses for suggestions."""

    def __init__(self, cfg: AssistConfig):
        self.cfg = cfg
        self.state = SuggestionState()
        self._last_declined_ms: int = 0
        self._last_cancel_ms: int = 0

    def can_suggest(self, now_ms: int) -> bool:
        cool = self.cfg.cooldown_suggest_s
        # Extend cooldown after decline/cancel
        if self._last_declined_ms > self.state.last_suggest_ms:
            cool = max(cool, self.cfg.cooldown_suggest_s + self.cfg.cooldown_decline_extra_s)
        if self._last_cancel_ms > self.state.last_suggest_ms:
            cool = max(cool, self.cfg.cooldown_suggest_s + self.cfg.cooldown_cancel_extra_s)
        return (now_ms - self.state.last_suggest_ms) >= int(cool * 1000.0)

    def set_pending(self, kind: str, text: str, top_id: Optional[str], alt_id: Optional[str], now_ms: int) -> None:
        self.state.pending_kind = kind
        self.state.pending_text = text
        self.state.pending_top_id = top_id
        self.state.pending_alt_id = alt_id
        self.state.last_suggest_ms = now_ms

    def clear_pending(self) -> None:
        self.state = SuggestionState(last_suggest_ms=self.state.last_suggest_ms)

    # User responses
    def on_yes(self) -> Optional[str]:
        if self.state.pending_kind == "suggest" and self.state.pending_top_id:
            obj_id = self.state.pending_top_id
            self.clear_pending()
            return obj_id
        return None

    def on_no(self) -> None:
        self._last_declined_ms = int(time.time() * 1000)
        self.clear_pending()

    def on_choice(self, choice: int) -> Optional[str]:
        # choice 1 -> top, 2 -> alt
        if self.state.pending_kind == "clarify":
            obj_id = self.state.pending_top_id if choice == 1 else self.state.pending_alt_id
            self.clear_pending()
            return obj_id
        return None

    def on_cancel(self) -> None:
        self._last_cancel_ms = int(time.time() * 1000)
        self.clear_pending()


