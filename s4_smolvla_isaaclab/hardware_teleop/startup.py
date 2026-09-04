"""Startup helpers: homing to a rest pose before Quest teleoperation."""

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


def merge_startup_home_poses(
    startup_cfg: HardwareStartupConfig,
    home_poses: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    merged = dict(home_poses)
    if startup_cfg.home_left_arm:
        merged["left_arm"] = np.asarray(startup_cfg.home_left_arm, dtype=np.float32)
    if startup_cfg.home_right_arm:
        merged["right_arm"] = np.asarray(startup_cfg.home_right_arm, dtype=np.float32)
    return merged


def quintic_unit(tau: float) -> float:
    """Rest-to-rest quintic: 10τ³ − 15τ⁴ + 6τ⁵, with zero vel/acc at 0 and 1."""
    tau = float(np.clip(tau, 0.0, 1.0))
    tau3 = tau * tau * tau
    return float(tau3 * (10.0 + tau * (-15.0 + 6.0 * tau)))


def interpolate_home_quintic(
    start: np.ndarray,
    home: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Blend arm joints from start to home. Hands stay at the start pose."""
    start_np = np.asarray(start, dtype=np.float32)
    home_np = np.asarray(home, dtype=np.float32)
    result = start_np.copy()
    blend = np.float32(quintic_unit(tau))
    for sl in (ACTION_SLICES.left_arm, ACTION_SLICES.right_arm):
        result[sl] = start_np[sl] + blend * (home_np[sl] - start_np[sl])
    return result


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
    resolved_home = merge_startup_home_poses(startup_cfg, home_poses)
    if not startup_cfg.move_to_home or not resolved_home:
        print("[HW-TELEOP] startup homing skipped (disabled or no home poses)", flush=True)
        return actual

    home_action = build_home_action(actual, profiles, resolved_home)
    # Do not command the hands during arm homing; they stay at the measured pose.
    home_action[ACTION_SLICES.left_hand] = actual[ACTION_SLICES.left_hand]
    home_action[ACTION_SLICES.right_hand] = actual[ACTION_SLICES.right_hand]
    error = arm_home_error(actual, home_action)
    if error <= startup_cfg.position_tolerance_rad:
        print(f"[HW-TELEOP] already at home pose (err={error:.4f} rad)", flush=True)
        return home_action

    requested_duration_s = max(float(startup_cfg.duration_s), control_dt)
    start_action = np.asarray(actual, dtype=np.float32).copy()
    max_step = max(float(startup_cfg.max_joint_step_rad), 0.0)
    requested_steps = max(1, int(np.ceil(requested_duration_s / control_dt)))
    # A quintic's peak slope is 1.875. Increase duration when necessary so
    # the safety step limiter does not flatten the curve around its midpoint.
    step_limited_steps = (
        int(np.ceil(1.875 * error / max_step)) if max_step > 0.0 else 1
    )
    total_steps = max(requested_steps, step_limited_steps)
    duration_s = total_steps * control_dt
    print(
        f"[HW-TELEOP] startup homing: quintic rest-to-rest over {duration_s:.1f}s "
        f"start_err={error:.3f} rad "
        f"left={np.round(home_action[ACTION_SLICES.left_arm], 3).tolist()} "
        f"right={np.round(home_action[ACTION_SLICES.right_arm], 3).tolist()}",
        flush=True,
    )
    print(
        "[HW-TELEOP] keep clear; Quest clutch is ignored until homing finishes",
        flush=True,
    )

    command = start_action.copy()
    next_deadline = time.monotonic()
    for step_index in range(total_steps + 1):
        spin_once()
        tau = step_index / total_steps
        desired = interpolate_home_quintic(start_action, home_action, tau)
        if max_step > 0.0:
            command = interpolate_toward_home(command, desired, max_step_rad=max_step)
            command[ACTION_SLICES.left_hand] = start_action[ACTION_SLICES.left_hand]
            command[ACTION_SLICES.right_hand] = start_action[ACTION_SLICES.right_hand]
        else:
            command = desired
        publish_step(command)
        if step_index == total_steps:
            actual = read_state()
            final_err = arm_home_error(actual, home_action)
            print(
                f"[HW-TELEOP] startup homing finished command_err="
                f"{arm_home_error(command, home_action):.4f} rad "
                f"measured_err={final_err:.4f} rad",
                flush=True,
            )
            return command
        next_deadline += control_dt
        sleep_s = next_deadline - time.monotonic()
        if sleep_s > 0.0:
            time.sleep(sleep_s)
        else:
            next_deadline = time.monotonic()
