from __future__ import annotations

import numpy as np

from ...common.errors import ContractError


def validate_policy_chunk(
    chunk: np.ndarray,
    *,
    measured_q7: np.ndarray,
    max_target_jump_rad: float,
    max_tracking_error_rad: float,
) -> np.ndarray:
    value = np.asarray(chunk, dtype=np.float32)
    measured = np.asarray(measured_q7, dtype=np.float32).reshape(7)
    if value.ndim != 2 or value.shape[1] != 8 or not np.isfinite(value).all():
        raise ContractError(f"policy chunk must be finite [N,8], got {value.shape}")
    if value.shape[0] > 1 and float(np.max(np.abs(np.diff(value[:, :7], axis=0)))) > max_target_jump_rad:
        raise ContractError("policy chunk contains an excessive adjacent joint-target jump")
    if float(np.max(np.abs(value[0, :7] - measured))) > max_tracking_error_rad:
        raise ContractError("first policy target is too far from measured joint state")
    if np.any((value[:, 7] < -0.05) | (value[:, 7] > 1.05)):
        raise ContractError("policy gripper values are outside the logical [0,1] range")
    return value
