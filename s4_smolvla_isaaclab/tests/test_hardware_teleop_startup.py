from __future__ import annotations

import numpy as np
import pytest

from hardware_teleop.config_loader import HardwareStartupConfig
from hardware_teleop.startup import (
    arm_home_error,
    build_home_action,
    interpolate_toward_home,
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
            duration_s=2.0,
            max_joint_step_rad=0.25,
            position_tolerance_rad=0.05,
            check_lowcmd_publishers=True,
            require_policy_lowcmd=True,
            policy_initial_timeout_s=5.0,
            policy_min_valid_frames=3,
            max_policy_age_s=0.5,
            require_sdk_mode5_merge=True,
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
            check_lowcmd_publishers=True,
            require_policy_lowcmd=True,
            policy_initial_timeout_s=5.0,
            policy_min_valid_frames=3,
            max_policy_age_s=0.5,
            require_sdk_mode5_merge=True,
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
