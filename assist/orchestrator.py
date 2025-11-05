from __future__ import annotations

import time
from typing import Optional

import torch

from .actions import build_align_and_grasp
from .config import AssistConfig, from_cli_args
from .dialogue import DialogueManager
from .logger import JsonlLogger
from .objects import ObjectsTracker
from .policy import infer_policy
from .schemas import Event
from .window import RollingWindowBuilder


class AssistOrchestrator:
    """10 Hz assist loop coordinating window, policy, dialogue, and actions."""

    def __init__(
        self,
        cfg: AssistConfig,
        sim,
        robot,
        controller,
        objects: ObjectsTracker,
        mux_input,
        physics_dt: float,
    ) -> None:
        self.cfg = cfg
        self.sim = sim
        self.robot = robot
        self.controller = controller
        self.objects = objects
        self.mux = mux_input
        self.physics_dt = float(physics_dt)

        self.win = RollingWindowBuilder(cfg)
        self.dialogue = DialogueManager(cfg)
        self.logger = JsonlLogger(cfg)

        self._accum_s = 0.0
        self._last_user_cmd: Optional[torch.Tensor] = None
        self._action_active: bool = False

    def close(self) -> None:
        self.logger.close()

    def set_last_user_cmd(self, cmd: Optional[torch.Tensor]) -> None:
        self._last_user_cmd = cmd

    def on_yes(self) -> None:
        obj_id = self.dialogue.on_yes()
        if obj_id is not None:
            self._start_action(obj_id)

    def on_no(self) -> None:
        self.dialogue.on_no()
        self.win.add_event(Event(t_ms=int(time.time() * 1000), type="assist_decline"))
        self.logger.log("assist_decline", {})

    def on_choice(self, which: int) -> None:
        obj_id = self.dialogue.on_choice(which)
        if obj_id is not None:
            self.logger.log("assist_choice", {"choice": which, "object_id": obj_id})
            self._start_action(obj_id)

    def _start_action(self, object_id: str) -> None:
        # Resolve object pose from tracker
        objs = {o.id: o for o in self.objects.snapshot()}
        if object_id not in objs:
            return
        ee_pos = self.robot.data.body_pose_w[0, 0, 0:3].tolist() if hasattr(self.robot.data, "body_pose_w") else [0.0, 0.0, 0.0]
        plan = build_align_and_grasp(self.cfg, (float(ee_pos[0]), float(ee_pos[1]), float(ee_pos[2])), objs[object_id].pose, self.physics_dt)
        self.mux.run_action(plan.cmd_stream)
        self._action_active = True
        self.logger.log("action_start", {"object_id": object_id, "steps": len(plan.cmd_stream)})

    def _finish_action(self) -> None:
        # Close gripper (best-effort)
        try:
            self.controller.set_mode("gripper")
            self.controller.gripper.command_close(self.robot)
            # Restore translate mode
            self.controller.set_mode("translate")
        except Exception:
            pass
        self._action_active = False
        self.logger.log("action_done", {})

    def tick(self, dt: float) -> None:
        # Called every physics step; run policy at assist rate
        self._accum_s += float(dt)
        if self._action_active:
            # Monitor completion
            if not self.mux.is_action_active():
                self._finish_action()
            return

        period = 1.0 / float(self.cfg.assist_rate_hz)
        if self._accum_s + 1e-9 < period:
            return
        self._accum_s = 0.0

        now_ms = int(time.time() * 1000)
        objs = self.objects.snapshot()
        win = self.win.tick(now_ms, self.robot, self._last_user_cmd, objs, robot_mode="manual")

        # If a suggestion is already pending, don't repeat
        if self.dialogue.state.pending_kind is not None:
            return

        # Suggest only if cooldown allows
        if not self.dialogue.can_suggest(now_ms):
            return

        pol = infer_policy(self.cfg, win)
        if pol.kind in ("suggest", "clarify") and pol.text:
            # Log and render (include compact context summary)
            context = JsonlLogger.summarize_window(win)
            self.logger.log(
                "suggest",
                {"text": pol.text, "confidence": pol.confidence, "top": pol.top_object_id, "alt": pol.alt_object_id, "context": context},
            )
            if self.cfg.verbose:
                print(f"[ASSIST] {pol.text} (conf={pol.confidence:.2f})")
            self.dialogue.set_pending(pol.kind, pol.text, pol.top_object_id, pol.alt_object_id, now_ms)
            if self.cfg.auto_accept and pol.kind == "suggest" and pol.top_object_id:
                self._start_action(pol.top_object_id)


