"""Joint mapping between 26D teleop commands and qiling lowcmd layout."""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from hardware_teleop.vendored.joint_layout import REAL_ROBOT_BODY_JOINT_ORDER
from s4_robot.control_mapping import ACTION_SLICES, BIMANUAL_ACTION_DIM, clip_joint_targets
from s4_robot.s4_robot_cfg import LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS


ARM_JOINT_NAMES = tuple(LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS)


def build_sign_map(reversed_joint_names: tuple[str, ...]) -> dict[str, float]:
    reversed_set = set(reversed_joint_names)
    return {name: (-1.0 if name in reversed_set else 1.0) for name in REAL_ROBOT_BODY_JOINT_ORDER}


def robot_to_teleop_sign(joint_name: str, sign_by_name: Mapping[str, float]) -> float:
    """Convert qiling /joint_states (already sign-corrected) to teleop URDF convention."""
    return float(sign_by_name.get(joint_name, 1.0))


def teleop_to_robot_sign(joint_name: str, sign_by_name: Mapping[str, float]) -> float:
    """Convert teleop URDF joint values to lowcmd motor convention."""
    return float(sign_by_name.get(joint_name, 1.0))


def extract_arm_positions(joint_positions: Mapping[str, float]) -> dict[str, float]:
    missing = [name for name in ARM_JOINT_NAMES if name not in joint_positions]
    if missing:
        raise KeyError(f"missing arm joints in /joint_states: {missing}")
    return {name: float(joint_positions[name]) for name in ARM_JOINT_NAMES}


def arms_to_bimanual_state(
    arm_positions: Mapping[str, float],
    *,
    left_hand: np.ndarray,
    right_hand: np.ndarray,
) -> np.ndarray:
    state = np.zeros(BIMANUAL_ACTION_DIM, dtype=np.float32)
    for index, name in enumerate(LEFT_ARM_JOINTS):
        state[ACTION_SLICES.left_arm][index] = float(arm_positions[name])
    for index, name in enumerate(RIGHT_ARM_JOINTS):
        state[ACTION_SLICES.right_arm][index] = float(arm_positions[name])
    state[ACTION_SLICES.left_hand] = np.asarray(left_hand, dtype=np.float32)
    state[ACTION_SLICES.right_hand] = np.asarray(right_hand, dtype=np.float32)
    if state.shape != (BIMANUAL_ACTION_DIM,) or not np.isfinite(state).all():
        raise ValueError("invalid bimanual state assembled from hardware feedback")
    return state


def bimanual_to_arm_q14(action: np.ndarray) -> np.ndarray:
    """Extract Pink's LA7+RA7 state from the shared 26D action layout."""
    action_np = np.asarray(action, dtype=np.float64)
    if action_np.shape != (BIMANUAL_ACTION_DIM,) or not np.isfinite(action_np).all():
        raise ValueError(
            f"expected finite action shape ({BIMANUAL_ACTION_DIM},), got {action_np.shape}"
        )
    return np.concatenate(
        (action_np[ACTION_SLICES.left_arm], action_np[ACTION_SLICES.right_arm])
    )


def apply_arm_q14(action: np.ndarray, arm_q14: np.ndarray) -> np.ndarray:
    """Return a 26D action with Pink's LA7+RA7 arm target inserted."""
    result = np.asarray(action, dtype=np.float32).copy()
    arm = np.asarray(arm_q14, dtype=np.float32)
    if result.shape != (BIMANUAL_ACTION_DIM,) or not np.isfinite(result).all():
        raise ValueError(
            f"expected finite action shape ({BIMANUAL_ACTION_DIM},), got {result.shape}"
        )
    if arm.shape != (14,) or not np.isfinite(arm).all():
        raise ValueError(f"expected finite Pink arm shape (14,), got {arm.shape}")
    result[ACTION_SLICES.left_arm] = arm[:7]
    result[ACTION_SLICES.right_arm] = arm[7:]
    return result


def limit_arm_step(
    desired_by_name: Mapping[str, float],
    previous_by_name: Mapping[str, float],
    *,
    max_step_rad: float,
) -> dict[str, float]:
    limited: dict[str, float] = {}
    max_step = max(float(max_step_rad), 0.0)
    for name in ARM_JOINT_NAMES:
        target = float(desired_by_name[name])
        previous = float(previous_by_name.get(name, target))
        delta = target - previous
        if max_step > 0.0 and abs(delta) > max_step:
            target = previous + math.copysign(max_step, delta)
        limited[name] = target
    return clip_joint_targets(limited)


def bimanual_arm_targets(action: np.ndarray) -> dict[str, float]:
    action_np = np.asarray(action, dtype=np.float64)
    if action_np.shape != (BIMANUAL_ACTION_DIM,):
        raise ValueError(f"expected action shape ({BIMANUAL_ACTION_DIM},), got {action_np.shape}")
    targets = {}
    targets.update({name: float(value) for name, value in zip(LEFT_ARM_JOINTS, action_np[ACTION_SLICES.left_arm], strict=True)})
    targets.update({name: float(value) for name, value in zip(RIGHT_ARM_JOINTS, action_np[ACTION_SLICES.right_arm], strict=True)})
    return clip_joint_targets(targets)
