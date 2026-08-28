from __future__ import annotations

from types import SimpleNamespace

import pytest

from hardware_teleop.vendored.joint_layout import DEFAULT_REVERSED_JOINT_NAMES
from hardware_teleop.vendored.joint_state_safety import JointStateFrameGuard
from hardware_teleop.vendored.lowstate_decode import decode_lowstate_arm_positions
from hardware_teleop.joint_mapping import ARM_JOINT_NAMES, build_sign_map


def test_decode_lowstate_applies_sign_and_returns_arms() -> None:
    sign_map = build_sign_map(DEFAULT_REVERSED_JOINT_NAMES)
    guard = JointStateFrameGuard(ARM_JOINT_NAMES, reject_zero_glitches=False)
    motors = []
    for index in range(26):
        motors.append(SimpleNamespace(q=float(index) * 0.01, dq=0.0, tau_est=0.0))

    arms, validation = decode_lowstate_arm_positions(
        motors,
        body_dof=26,
        sign_by_name=sign_map,
        guard=guard,
        arm_names=ARM_JOINT_NAMES,
    )
    assert validation.accepted
    assert arms is not None
    assert set(arms) == set(ARM_JOINT_NAMES)


def test_decode_lowstate_rejects_empty() -> None:
    sign_map = build_sign_map(DEFAULT_REVERSED_JOINT_NAMES)
    guard = JointStateFrameGuard(ARM_JOINT_NAMES)
    arms, validation = decode_lowstate_arm_positions(
        [],
        body_dof=26,
        sign_by_name=sign_map,
        guard=guard,
        arm_names=ARM_JOINT_NAMES,
    )
    assert arms is None
    assert not validation.accepted
