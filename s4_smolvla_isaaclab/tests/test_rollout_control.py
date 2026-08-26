from __future__ import annotations

import numpy as np
import pytest

from s4_pipeline.rollout_control import apply_phase_action_mask, rollout_hard_failure_reason


def test_phase_action_mask_changes_only_declared_groups():
    hold = np.arange(26, dtype=np.float32)
    candidate = hold + 100.0
    masked = apply_phase_action_mask(candidate, hold, ["right_arm", "right_hand"])

    np.testing.assert_allclose(masked[:13], hold[:13])
    np.testing.assert_allclose(masked[13:], candidate[13:])


def test_phase_action_mask_rejects_unknown_group_and_wrong_shape():
    with pytest.raises(ValueError, match="unknown active action groups"):
        apply_phase_action_mask(np.zeros(26), np.zeros(26), ["torso"])
    with pytest.raises(ValueError, match=r"shape=\(26,\)"):
        apply_phase_action_mask(np.zeros(25), np.zeros(26), ["left_arm"])


def test_only_drawer_opening_condition_produces_a_physical_hard_failure():
    gate = {"drawer_open_min": 0.08}
    assert (
        rollout_hard_failure_reason(
            "drawer_open_min", drawer_open_m=0.079, gate_config=gate
        )
        == "drawer=0.079<0.080"
    )
    assert (
        rollout_hard_failure_reason(
            "drawer_open_min", drawer_open_m=0.081, gate_config=gate
        )
        is None
    )
    assert rollout_hard_failure_reason("none", drawer_open_m=0.0, gate_config={}) is None


def test_rollout_hard_failure_condition_rejects_invalid_contract():
    with pytest.raises(ValueError, match="unknown rollout failure condition"):
        rollout_hard_failure_reason("object_speed", drawer_open_m=0.1, gate_config={})
    with pytest.raises(ValueError, match="requires gate_config"):
        rollout_hard_failure_reason("drawer_open_min", drawer_open_m=0.1, gate_config={})
