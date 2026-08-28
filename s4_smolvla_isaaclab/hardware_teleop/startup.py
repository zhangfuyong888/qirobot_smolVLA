"""Startup helpers: homing to task home pose before Quest teleoperation."""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from hardware_teleop.config_loader import HardwareStartupConfig
from s4_robot.control_mapping import ACTION_SLICES


def build_home_action(
    actual_action: np.ndarray,
    profiles: dict[str, np.ndarray],
    home_poses: dict[str, np.ndarray],
) -> np.ndarray:
    target = np.asarray(actual_action, dtype=np.float32).copy()
    if "left_arm" in home_poses:
        target[ACTION_SLICES.left_arm] = np.asarray(home_poses["left_arm"], dtype=np.float32)
    if "right_arm" in home_poses:
        target[ACTION_SLICES.right_arm] = np.asarray(home_poses["right_arm"], dtype=np.float32)
    target[ACTION_SLICES.left_hand] = np.asarray(profiles["left_open"], dtype=np.float32)
    target[ACTION_SLICES.right_hand] = np.asarray(profiles["right_open"], dtype=np.float32)
    return target


def interpolate_toward_home(
    current: np.ndarray,
    home: np.ndarray,
    *,
    max_step_rad: float,
) -> np.ndarray:
    """Move each arm joint toward home by at most max_step_rad."""
    next_action = np.asarray(current, dtype=np.float32).copy()
    max_step = max(float(max_step_rad), 0.0)
    for sl in (ACTION_SLICES.left_arm, ACTION_SLICES.right_arm):
        delta = home[sl] - next_action[sl]
        if max_step > 0.0:
            delta = np.clip(delta, -max_step, max_step)
        next_action[sl] = next_action[sl] + delta
    next_action[ACTION_SLICES.left_hand] = home[ACTION_SLICES.left_hand]
    next_action[ACTION_SLICES.right_hand] = home[ACTION_SLICES.right_hand]
    return next_action


def arm_home_error(current: np.ndarray, home: np.ndarray) -> float:
    left = float(np.max(np.abs(current[ACTION_SLICES.left_arm] - home[ACTION_SLICES.left_arm])))
    right = float(np.max(np.abs(current[ACTION_SLICES.right_arm] - home[ACTION_SLICES.right_arm])))
    return max(left, right)


def run_startup_homing(
    *,
    startup_cfg: HardwareStartupConfig,
    control_dt: float,
    read_state: Callable[[], np.ndarray],
    publish_step: Callable[[np.ndarray], None],
    spin_once: Callable[[], None],
    home_poses: dict[str, np.ndarray],
    profiles: dict[str, np.ndarray],
) -> np.ndarray:
    actual = read_state()
    if not startup_cfg.move_to_home or not home_poses:
        print("[HW-TELEOP] startup homing skipped (disabled or no home poses)", flush=True)
        return actual

    home_action = build_home_action(actual, profiles, home_poses)
    error = arm_home_error(actual, home_action)
    if error <= startup_cfg.position_tolerance_rad:
        print(f"[HW-TELEOP] already at home pose (err={error:.4f} rad)", flush=True)
        return home_action

    print(
        f"[HW-TELEOP] startup homing: moving arms to task home over up to "
        f"{startup_cfg.duration_s:.1f}s (step={startup_cfg.max_joint_step_rad:.3f} rad)",
        flush=True,
    )

    command = np.asarray(actual, dtype=np.float32).copy()
    deadline = time.monotonic() + max(startup_cfg.duration_s, control_dt)
    while time.monotonic() < deadline:
        spin_once()
        actual = read_state()
        command = interpolate_toward_home(
            actual,
            home_action,
            max_step_rad=startup_cfg.max_joint_step_rad,
        )
        publish_step(command)
        if arm_home_error(actual, home_action) <= startup_cfg.position_tolerance_rad:
            print("[HW-TELEOP] startup homing reached home pose", flush=True)
            return home_action
        time.sleep(control_dt)

    print("[HW-TELEOP][WARN] startup homing timed out before reaching home tolerance", flush=True)
    return command
