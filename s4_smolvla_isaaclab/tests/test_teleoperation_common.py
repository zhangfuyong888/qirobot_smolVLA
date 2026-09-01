from __future__ import annotations

import numpy as np
import pytest

from teleoperation.common import load_task_control_profiles, smooth_command


def test_common_smoothing_is_finite_and_step_limited() -> None:
    result = smooth_command(
        np.array([0.0, 1.0]),
        np.array([1.0, -1.0]),
        alpha=0.5,
        max_joint_step=0.1,
    )
    assert result == pytest.approx([0.1, 0.9])
    with pytest.raises(ValueError, match="finite"):
        smooth_command(np.array([0.0]), np.array([np.nan]), 1.0, 0.1)


def test_common_loads_task_hand_and_home_profiles() -> None:
    profiles, home = load_task_control_profiles("drawer_insert_close")
    assert set(profiles) == {"left_open", "left_close", "right_open", "right_close"}
    assert profiles["left_open"].shape == (6,)
    assert home["left_arm"].shape == (7,)
