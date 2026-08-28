"""Map teleop trigger blend to qiling HandsCmd uint16 positions."""

from __future__ import annotations

import numpy as np

from hardware_teleop.config_loader import HardwareHandsConfig


def blend_uint16(open_values: tuple[int, ...], close_values: tuple[int, ...], trigger: float) -> list[int]:
    if len(open_values) != 6 or len(close_values) != 6:
        raise ValueError("hand profiles must contain six uint16 values")
    t = float(np.clip(trigger, 0.0, 1.0))
    blended = []
    for open_value, close_value in zip(open_values, close_values, strict=True):
        value = int(round(float(open_value) + t * (float(close_value) - float(open_value))))
        blended.append(int(np.clip(value, 0, 255)))
    return blended


def left_hand_positions(config: HardwareHandsConfig, trigger: float) -> list[int]:
    return blend_uint16(config.left_open_uint16, config.left_close_uint16, trigger)


def right_hand_positions(config: HardwareHandsConfig, trigger: float) -> list[int]:
    return blend_uint16(config.right_open_uint16, config.right_close_uint16, trigger)


def trigger_from_hand6(open_profile: np.ndarray, close_profile: np.ndarray, hand6: np.ndarray) -> float:
    """Recover an approximate 0..1 trigger from smoothed hand6 radians."""
    open_values = np.asarray(open_profile, dtype=np.float64)
    close_values = np.asarray(close_profile, dtype=np.float64)
    hand_values = np.asarray(hand6, dtype=np.float64)
    span = close_values - open_values
    weights = np.abs(span)
    if float(weights.sum()) <= 1.0e-8:
        return 0.0
    normalized = (hand_values - open_values) / np.where(np.abs(span) < 1.0e-8, 1.0, span)
    return float(np.clip(np.average(normalized, weights=weights), 0.0, 1.0))
