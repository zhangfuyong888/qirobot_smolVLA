"""Optional observation points for hardware teleop. Collection hooks in; teleop stays unchanged."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from teleoperation.protocol import ControllerFrame


@dataclass
class TickRequest:
    allow_teleop: bool = True
    command_override: np.ndarray | None = None
    hand_triggers: tuple[float, float] | None = None
    publish_arms: bool = True
    stop: bool = False
    hud: dict[str, str] | None = None


@dataclass
class TeleopTick:
    monotonic_s: float
    timestamp_ns: int
    frame: ControllerFrame | None
    actual_action: np.ndarray
    command_action: np.ndarray
    published_arm: dict[str, float]
    left_trigger: float
    right_trigger: float
    left_active: bool
    right_active: bool
    fault_active: bool
    output_relinquished: bool
    state_feed_stale: bool
    state_age_s: float
    command_transport_ok: bool
    command_published: bool
    input_stale: bool
    fault_reason: str
    mapping_dt: float


class TeleopHooks:
    def begin_tick(self, frame: ControllerFrame | None, now_s: float) -> TickRequest:
        del frame, now_s
        return TickRequest()

    def end_tick(self, tick: TeleopTick) -> None:
        del tick
