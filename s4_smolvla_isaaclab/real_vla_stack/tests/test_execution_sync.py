from __future__ import annotations

from real_vla_stack.robot.rollout.execution_sync import ExecutionSyncGuard


def _guard() -> ExecutionSyncGuard:
    return ExecutionSyncGuard(
        policy_lag_rad=0.12,
        contact_tracking_error_rad=0.12,
        trigger_cycles=3,
    )


def test_policy_lag_triggers_hold_and_rtc_reset_after_sustained_drift() -> None:
    guard = _guard()
    assert not guard.observe(
        policy_execution_lag_rad=0.13,
        command_tracking_error_rad=0.01,
        gripper_closed=False,
    )
    assert not guard.observe(
        policy_execution_lag_rad=0.14,
        command_tracking_error_rad=0.01,
        gripper_closed=False,
    )
    assert guard.observe(
        policy_execution_lag_rad=0.15,
        command_tracking_error_rad=0.01,
        gripper_closed=False,
    )
    assert guard.hold
    assert guard.reset_pending
    assert guard.reason == "policy_execution_lag"


def test_contact_tracking_only_triggers_while_gripper_is_closed() -> None:
    guard = _guard()
    for _ in range(4):
        assert not guard.observe(
            policy_execution_lag_rad=0.01,
            command_tracking_error_rad=0.14,
            gripper_closed=False,
        )
    for index in range(3):
        triggered = guard.observe(
            policy_execution_lag_rad=0.01,
            command_tracking_error_rad=0.14,
            gripper_closed=True,
        )
        assert triggered == (index == 2)
    assert guard.reason == "contact_tracking_lag"


def test_acknowledged_resync_clears_hold() -> None:
    guard = _guard()
    for _ in range(3):
        guard.observe(
            policy_execution_lag_rad=0.2,
            command_tracking_error_rad=0.0,
            gripper_closed=False,
        )
    guard.acknowledge_resync()
    assert not guard.hold
    assert not guard.reset_pending
    assert guard.reason == ""


def test_unsafe_chunk_forces_immediate_resync() -> None:
    guard = _guard()
    assert guard.force_resync("unsafe_policy_chunk")
    assert guard.hold
    assert guard.reset_pending
    assert guard.reason == "unsafe_policy_chunk"
    assert not guard.force_resync("unsafe_policy_chunk")
