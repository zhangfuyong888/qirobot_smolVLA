from __future__ import annotations

import numpy as np
import pytest

from hardware_teleop.hand_mapping import blend_uint16, trigger_from_hand6


def test_blend_uint16_open_close() -> None:
    open_values = (255, 255, 255, 255, 255, 255)
    close_values = (60, 128, 200, 200, 150, 90)
    assert blend_uint16(open_values, close_values, 0.0) == list(open_values)
    assert blend_uint16(open_values, close_values, 1.0) == list(close_values)
    mid = blend_uint16(open_values, close_values, 0.5)
    assert all(0 <= value <= 255 for value in mid)


def test_trigger_from_hand6_recovers_blend() -> None:
    open_profile = np.array([0.9, 0.0, 0.05, 0.05, 0.05, 0.05], dtype=np.float32)
    close_profile = np.array([1.0, 0.22, 0.85, 0.85, 0.85, 0.85], dtype=np.float32)
    hand6 = open_profile + 0.4 * (close_profile - open_profile)
    trigger = trigger_from_hand6(open_profile, close_profile, hand6)
    assert trigger == pytest.approx(0.4, abs=0.05)
