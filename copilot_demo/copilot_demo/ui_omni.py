"""Lightweight omni.ui panel for grasp-copilot assistance."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence


class AssistUI:
    """Small omni.ui helper that mirrors gui_assist_demo semantics."""

    def __init__(
        self,
        *,
        on_ask: Callable[[], None],
        on_reset: Callable[[], None],
        on_choice: Callable[[str], None],
        on_mode_change: Callable[[str], None],
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self._on_ask = on_ask
        self._on_reset = on_reset
        self._on_choice = on_choice
        self._on_mode_change = on_mode_change
        self._mode = "translation"
        self._choices: Sequence[str] = []
        # Track created omni.ui widgets explicitly since some omni.ui container
        # types (e.g. VStack) don't expose a stable `.children` API across versions.
        self._choice_widgets: List[object] = []
        self._log: List[str] = []
        self._ui = None
        if enabled:
            self._build_ui()

    def _build_ui(self) -> None:
        import omni.ui as ui  # type: ignore

        self._ui = ui
        self.window = ui.Window("Grasp Copilot", width=380, height=520)
        with self.window.frame:
            with ui.VStack(spacing=6, style={"margin": 10}):
                self.status_label = ui.Label("Status: idle", word_wrap=True)

                with ui.HStack(height=28):
                    ui.Label("Mode:", width=60)
                    self._mode_buttons = {}
                    for m in ("translation", "rotation", "gripper"):
                        btn = ui.Button(m, width=90, clicked_fn=lambda mm=m: self._set_mode(mm))
                        self._mode_buttons[m] = btn

                with ui.HStack(spacing=8):
                    ui.Button("Ask assistance", height=32, clicked_fn=self._on_ask)
                    ui.Button("Reset", height=32, clicked_fn=self._on_reset)

                ui.Label("Log")
                self._log_frame = ui.ScrollingFrame(height=180, style={"background_color": 0x151515ff})
                with self._log_frame:
                    self._log_stack = ui.VStack(spacing=2, style={"margin": 4})

                ui.Label("Choices")
                self._choice_stack = ui.VStack(spacing=4)

                ui.Label("Workspace grid (A1..C3)")
                with ui.VStack(spacing=2, style={"margin": 4}):
                    for row in ("A", "B", "C"):
                        with ui.HStack(spacing=4):
                            for col in ("1", "2", "3"):
                                ui.Label(f"{row}{col}", width=40, alignment=ui.Alignment.CENTER)
                self._bounds_label = ui.Label("", word_wrap=True)

    def set_status(self, text: str) -> None:
        self._log.append(text)
        if len(self._log) > 200:
            self._log = self._log[-200:]
        if not self.enabled or self._ui is None:
            return
        self.status_label.text = f"Status: {text}"
        # Append to log area
        with self._log_stack:
            self._ui.Label(text, word_wrap=True)

    def set_bounds_hint(self, workspace_min, workspace_max) -> None:
        if not self.enabled or self._ui is None:
            return
        self._bounds_label.text = f"Workspace X[{workspace_min[0]:.2f},{workspace_max[0]:.2f}] Y[{workspace_min[1]:.2f},{workspace_max[1]:.2f}]"

    def _set_mode(self, mode: str) -> None:
        self._mode = mode
        try:
            self._on_mode_change(mode)
        except Exception:
            pass

    def set_choices(self, choices: Sequence[str]) -> None:
        self._choices = choices
        if not self.enabled or self._ui is None:
            return
        # Rebuild choice buttons
        for w in list(self._choice_widgets):
            try:
                w.destroy()  # type: ignore[attr-defined]
            except Exception:
                pass
        self._choice_widgets = []
        with self._choice_stack:
            for ch in choices:
                btn = self._ui.Button(ch, clicked_fn=lambda cc=ch: self._on_choice(cc))
                self._choice_widgets.append(btn)
