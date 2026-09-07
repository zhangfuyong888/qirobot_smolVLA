from __future__ import annotations

import numpy as np

from ...common.errors import ContractError


def validate_policy_chunk(
    chunk: np.ndarray,
    *,
    measured_q7: np.ndarray,
    max_target_jump_rad: float,
    max_tracking_error_rad: float,
    enforce_initial_tracking: bool = True,
) -> np.ndarray:
    value = np.asarray(chunk, dtype=np.float32)
    measured = np.asarray(measured_q7, dtype=np.float32).reshape(7)
    if value.ndim != 2 or value.shape[1] != 8 or not np.isfinite(value).all():
        raise ContractError(f"policy chunk must be finite [N,8], got {value.shape}")
    if value.shape[0] > 1 and float(np.max(np.abs(np.diff(value[:, :7], axis=0)))) > max_target_jump_rad:
        raise ContractError("policy chunk contains an excessive adjacent joint-target jump")
    if enforce_initial_tracking:
        delta = np.abs(value[0, :7] - measured)
        joint = int(np.argmax(delta))
        if float(delta[joint]) > max_tracking_error_rad:
            raise ContractError(
                "first policy target is too far from observation state: "
                f"joint={joint} delta={float(delta[joint]):.3f}rad "
                f"limit={float(max_tracking_error_rad):.3f}rad"
            )
    # The deployed hand is binary and BinaryGripper applies hysteresis. Finite
    # generative-policy overshoot is therefore safely saturated instead of
    # discarding an otherwise valid seven-joint trajectory.
    value = value.copy()
    value[:, 7] = np.clip(value[:, 7], 0.0, 1.0)
    return value


def validate_execution_target(
    target_q7: np.ndarray,
    *,
    measured_q7: np.ndarray,
    max_tracking_error_rad: float,
) -> None:
    """Check the delay-aligned target that would actually enter the controller."""
    target = np.asarray(target_q7, dtype=np.float32).reshape(7)
    measured = np.asarray(measured_q7, dtype=np.float32).reshape(7)
    delta = np.abs(target - measured)
    joint = int(np.argmax(delta))
    if float(delta[joint]) > float(max_tracking_error_rad):
        raise ContractError(
            "delay-aligned policy target is too far from current state: "
            f"joint={joint} delta={float(delta[joint]):.3f}rad "
            f"limit={float(max_tracking_error_rad):.3f}rad"
        )
