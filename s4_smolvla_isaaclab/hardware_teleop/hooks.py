"""Optional observation points for hardware teleop. Collection hooks in; teleop stays unchanged."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from teleoperation.protocol import ControllerFrame


@dataclass
class TickRequest:
    allow_teleop: bool = True
    enabled_arms: str | None = None
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


@dataclass(frozen=True)
class TeleopStatus:
    monotonic_s: float
    loop_hz: float
    target_hz: float
    quest_clients: int
    quest_frame_age_s: float
    input_stale: bool
    state_feed_stale: bool
    arm_graph_conflict: bool
    left_active: bool
    right_active: bool
    left_grip: float
    right_grip: float
    left_trigger: float
    right_trigger: float
    left_tracking: bool
    right_tracking: bool
    left_requires_release: bool
    right_requires_release: bool
    calibrated: bool
    boundary_safe: bool
    boundary_distance_m: float | None
    fault_reason: str
    proximal_tracking_error_rad: float
    left_tcp_error_m: float
    right_tcp_error_m: float
    state_age_s: float
    command_output_enabled: bool
    output_relinquished: bool
    joint_limit_active_joints: int
    minimum_joint_limit_margin_rad: float


class TeleopHooks:
    def begin_tick(self, frame: ControllerFrame | None, now_s: float) -> TickRequest:
        del frame, now_s
        return TickRequest()

    def end_tick(self, tick: TeleopTick) -> None:
        del tick

    def on_status(self, status: TeleopStatus) -> bool:
        """Return true when the hook rendered the periodic status itself."""
        del status
        return False

    def on_client_log(self, level: str, message: str) -> bool:
        """Return true when the hook consumed a WebXR client log message."""
        del level, message
        return False
