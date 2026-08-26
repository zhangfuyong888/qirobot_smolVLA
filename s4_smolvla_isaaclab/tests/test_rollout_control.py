from __future__ import annotations

import numpy as np
import pytest

from s4_pipeline.rollout_control import apply_phase_action_mask


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
