from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hardware_teleop.joint_mapping import (
    ARM_JOINT_NAMES,
    apply_arm_q14,
    arms_to_bimanual_state,
    bimanual_to_arm_q14,
    build_sign_map,
    limit_arm_step,
    teleop_to_robot_sign,
)
from hardware_teleop.vendored.joint_layout import DEFAULT_REVERSED_JOINT_NAMES
from s4_robot.s4_robot_cfg import LEFT_ARM_JOINTS


def test_sign_inversion_matches_qiling_defaults() -> None:
    sign_map = build_sign_map(DEFAULT_REVERSED_JOINT_NAMES)
    assert teleop_to_robot_sign("left_wrist_roll_joint", sign_map) == -1.0
    assert teleop_to_robot_sign("left_elbow_joint", sign_map) == 1.0


def test_limit_arm_step_clamps_large_delta() -> None:
    previous = {name: 0.0 for name in ARM_JOINT_NAMES}
    desired = dict(previous)
    desired[LEFT_ARM_JOINTS[0]] = 1.0
    limited = limit_arm_step(desired, previous, max_step_rad=0.065)
    assert limited[LEFT_ARM_JOINTS[0]] == pytest.approx(0.065)


def test_arms_to_bimanual_state_shape() -> None:
    arms = {name: float(index) * 0.01 for index, name in enumerate(ARM_JOINT_NAMES)}
    left_hand = np.zeros(6, dtype=np.float32)
    right_hand = np.ones(6, dtype=np.float32)
    state = arms_to_bimanual_state(arms, left_hand=left_hand, right_hand=right_hand)
    assert state.shape == (26,)


def test_pink_q14_round_trip_preserves_hands_and_arm_order() -> None:
    action = np.arange(26, dtype=np.float32) * 0.01
    q14 = bimanual_to_arm_q14(action)
    assert q14 == pytest.approx(np.r_[action[0:7], action[13:20]])

    replacement = np.linspace(-0.7, 0.6, 14, dtype=np.float32)
    result = apply_arm_q14(action, replacement)
    assert bimanual_to_arm_q14(result) == pytest.approx(replacement)
    assert result[7:13] == pytest.approx(action[7:13])
    assert result[20:26] == pytest.approx(action[20:26])
