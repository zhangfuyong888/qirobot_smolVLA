from __future__ import annotations

import numpy as np
import pytest

from real_vla_stack.robot.rollout.command_filter import JointCommandFilter


def test_joint_filter_limits_velocity_and_acceleration() -> None:
    filt = JointCommandFilter(
        control_hz=10,
        max_velocity_rad_s=0.5,
        max_acceleration_rad_s2=1.0,
    )
    filt.reset(np.zeros(7))
    first = filt.step(np.ones(7))
    second = filt.step(np.ones(7))
    assert first == pytest.approx(np.full(7, 0.01))
    assert second == pytest.approx(np.full(7, 0.03))
    assert filt.velocity == pytest.approx(np.full(7, 0.2))


def test_joint_filter_brakes_before_reversing() -> None:
    filt = JointCommandFilter(
        control_hz=10,
        max_velocity_rad_s=0.5,
        max_acceleration_rad_s2=1.0,
    )
    filt.reset(np.zeros(7))
    for _ in range(5):
        filt.step(np.ones(7))
    before = filt.step(np.ones(7))
    after_reversal = filt.step(-np.ones(7))
    assert np.all(after_reversal > before)
    assert filt.velocity == pytest.approx(np.full(7, 0.4))


def test_joint_filter_applies_per_joint_limits_and_reports_them() -> None:
    filt = JointCommandFilter(
        control_hz=10,
        max_velocity_rad_s=np.arange(1, 8, dtype=float) / 10,
        max_acceleration_rad_s2=np.arange(1, 8, dtype=float),
    )
    filt.reset(np.zeros(7))
    result = filt.step(np.ones(7))
    assert result == pytest.approx(np.arange(1, 8, dtype=float) / 100)
    assert filt.limited_joints.tolist() == [True] * 7
