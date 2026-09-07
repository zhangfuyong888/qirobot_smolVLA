from __future__ import annotations

from types import SimpleNamespace

import pytest

from real_vla_stack.common.errors import (
    CommandOutputRelinquishedError,
    CommandRouteConflictError,
    RobotStateStaleError,
    RobotTrackingError,
)
from real_vla_stack.robot.rollout.main import (
    _ensure_command_tracking,
    _ensure_robot_health,
    _wait_for_fresh_initial_state,
)


class _Bridge:
    def __init__(self) -> None:
        self.last_state_age_s = 0.01
        self.output_relinquished = False
        self.release_reason = ""
        self.command_publisher_conflicts = ()
        self.stale = False
        self.conflicted = False

    def is_state_feed_stale(self, _max_age_s: float) -> bool:
        return self.stale

    def is_arm_command_graph_conflicted(self) -> bool:
        return self.conflicted

    @property
    def state_ready(self) -> bool:
        return True

    def spin_once(self, timeout_sec: float = 0.0) -> None:
        del timeout_sec


def _hardware():
    return SimpleNamespace(hardware=SimpleNamespace(max_state_age_s=0.2))


def test_health_rejects_stale_feedback_even_in_shadow() -> None:
    bridge = _Bridge()
    bridge.stale = True
    with pytest.raises(RobotStateStaleError):
        _ensure_robot_health(bridge, _hardware(), live=False)


def test_initial_state_wait_refreshes_historically_ready_state() -> None:
    class RefreshingBridge(_Bridge):
        def __init__(self) -> None:
            super().__init__()
            self.stale = True
            self.spin_count = 0

        def spin_once(self, timeout_sec: float = 0.0) -> None:
            del timeout_sec
            self.spin_count += 1
            self.stale = False

    bridge = RefreshingBridge()
    _wait_for_fresh_initial_state(bridge, _hardware(), timeout_s=0.1)
    assert bridge.spin_count == 1


def test_live_health_rejects_dynamic_command_conflict() -> None:
    bridge = _Bridge()
    bridge.conflicted = True
    bridge.command_publisher_conflicts = ("/legacy/controller",)
    with pytest.raises(CommandRouteConflictError, match="legacy/controller"):
        _ensure_robot_health(bridge, _hardware(), live=True)


def test_live_health_rejects_relinquished_output() -> None:
    bridge = _Bridge()
    bridge.output_relinquished = True
    bridge.release_reason = "watchdog"
    with pytest.raises(CommandOutputRelinquishedError, match="watchdog"):
        _ensure_robot_health(bridge, _hardware(), live=True)


def test_live_tracking_rejects_accumulated_command_error() -> None:
    published = SimpleNamespace(
        motion_allowed=True,
        arm_target_q=[0.2] * 7,
    )
    adapter = SimpleNamespace(last_published=lambda: published)
    with pytest.raises(RobotTrackingError, match="0.200rad"):
        _ensure_command_tracking(adapter, [0.0] * 7, 0.18, live=True)
