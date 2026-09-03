from __future__ import annotations

import numpy as np
import pytest

from hardware_teleop.config_loader import HardwareStartupConfig
from hardware_teleop.startup import (
    arm_home_error,
    build_home_action,
    interpolate_home_quintic,
    interpolate_toward_home,
    merge_startup_home_poses,
    quintic_unit,
    run_startup_homing,
)
from s4_robot.control_mapping import ACTION_SLICES, BIMANUAL_ACTION_DIM


def _zero_action() -> np.ndarray:
    return np.zeros(BIMANUAL_ACTION_DIM, dtype=np.float32)


def test_build_home_action_uses_task_home_poses() -> None:
    actual = _zero_action()
    actual[ACTION_SLICES.left_arm] = np.ones(7, dtype=np.float32)
    profiles = {
        "left_open": np.full(6, 0.1, dtype=np.float32),
        "right_open": np.full(6, 0.2, dtype=np.float32),
    }
    home_poses = {
        "left_arm": np.zeros(7, dtype=np.float32),
        "right_arm": np.full(7, 0.5, dtype=np.float32),
    }
    target = build_home_action(actual, profiles, home_poses)
    np.testing.assert_allclose(target[ACTION_SLICES.left_arm], 0.0)
    np.testing.assert_allclose(target[ACTION_SLICES.right_arm], 0.5)
    np.testing.assert_allclose(target[ACTION_SLICES.left_hand], 0.1)
    np.testing.assert_allclose(target[ACTION_SLICES.right_hand], 0.2)


def test_interpolate_toward_home_limits_joint_step() -> None:
    current = _zero_action()
    home = _zero_action()
    home[ACTION_SLICES.left_arm] = np.full(7, 1.0, dtype=np.float32)
    next_action = interpolate_toward_home(current, home, max_step_rad=0.2)
    np.testing.assert_allclose(next_action[ACTION_SLICES.left_arm], 0.2)


def test_quintic_unit_is_rest_to_rest() -> None:
    assert quintic_unit(0.0) == pytest.approx(0.0)
    assert quintic_unit(1.0) == pytest.approx(1.0)
    assert quintic_unit(0.5) == pytest.approx(0.5)
    dt = 1.0e-4
    assert (quintic_unit(dt) - quintic_unit(0.0)) / dt == pytest.approx(0.0, abs=1.0e-3)
    assert (quintic_unit(1.0) - quintic_unit(1.0 - dt)) / dt == pytest.approx(0.0, abs=1.0e-3)


def test_interpolate_home_quintic_only_moves_arms() -> None:
    start = _zero_action()
    start[ACTION_SLICES.left_arm] = 1.0
    start[ACTION_SLICES.left_hand] = 0.4
    home = _zero_action()
    home[ACTION_SLICES.left_hand] = 0.9
    mid = interpolate_home_quintic(start, home, 0.5)
    np.testing.assert_allclose(mid[ACTION_SLICES.left_arm], 0.5)
    np.testing.assert_allclose(mid[ACTION_SLICES.left_hand], 0.4)


def test_merge_startup_home_poses_overrides_task_home() -> None:
    merged = merge_startup_home_poses(
        HardwareStartupConfig(
            move_to_home=True,
            duration_s=5.0,
            max_joint_step_rad=0.05,
            position_tolerance_rad=0.05,
            check_arm_command_publishers=True,
            require_sdk_arm_replay=True,
            home_left_arm=(-0.25, 0.45, -0.34, -0.52, -0.65, -0.33, 0.19),
            home_right_arm=(-0.25, -0.45, -0.34, -0.52, 0.65, -0.33, 0.19),
        ),
        {"left_arm": np.ones(7, dtype=np.float32)},
    )
    np.testing.assert_allclose(merged["left_arm"], [-0.25, 0.45, -0.34, -0.52, -0.65, -0.33, 0.19])
    np.testing.assert_allclose(merged["right_arm"], [-0.25, -0.45, -0.34, -0.52, 0.65, -0.33, 0.19])


def test_run_startup_homing_reaches_home() -> None:
    actual = _zero_action()
    actual[ACTION_SLICES.left_arm] = np.full(7, 1.0, dtype=np.float32)
    profiles = {
        "left_open": np.zeros(6, dtype=np.float32),
        "right_open": np.zeros(6, dtype=np.float32),
    }
    home_poses = {"left_arm": np.zeros(7, dtype=np.float32)}
    published: list[np.ndarray] = []

    def read_state() -> np.ndarray:
        return actual.copy()

    def publish_step(action: np.ndarray) -> None:
        published.append(action.copy())
        actual[ACTION_SLICES.left_arm] = action[ACTION_SLICES.left_arm]

    result = run_startup_homing(
        startup_cfg=HardwareStartupConfig(
            move_to_home=True,
            duration_s=0.05,
            max_joint_step_rad=0.25,
            position_tolerance_rad=0.05,
            check_arm_command_publishers=True,
            require_sdk_arm_replay=True,
        ),
        control_dt=0.01,
        read_state=read_state,
        publish_step=publish_step,
        spin_once=lambda: None,
        home_poses=home_poses,
        profiles=profiles,
    )
    assert arm_home_error(result, build_home_action(actual, profiles, home_poses)) <= 0.05
    assert len(published) >= 4


def test_run_startup_homing_skips_when_disabled() -> None:
    actual = _zero_action()
    actual[ACTION_SLICES.left_arm] = np.full(7, 1.0, dtype=np.float32)
    profiles = {
        "left_open": np.zeros(6, dtype=np.float32),
        "right_open": np.zeros(6, dtype=np.float32),
    }
    published: list[np.ndarray] = []

    result = run_startup_homing(
        startup_cfg=HardwareStartupConfig(
            move_to_home=False,
            duration_s=1.0,
            max_joint_step_rad=0.1,
            position_tolerance_rad=0.01,
            check_arm_command_publishers=True,
            require_sdk_arm_replay=True,
        ),
        control_dt=0.01,
        read_state=lambda: actual.copy(),
        publish_step=lambda action: published.append(action.copy()),
        spin_once=lambda: None,
        home_poses={"left_arm": np.zeros(7, dtype=np.float32)},
        profiles=profiles,
    )
    assert published == []
    np.testing.assert_allclose(result[ACTION_SLICES.left_arm], 1.0)
