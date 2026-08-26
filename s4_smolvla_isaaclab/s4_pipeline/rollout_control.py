"""Pure phase-aware action helpers for the 26D bimanual rollout contract."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


ACTION_GROUP_SLICES = {
    "left_arm": slice(0, 7),
    "left_hand": slice(7, 13),
    "right_arm": slice(13, 20),
    "right_hand": slice(20, 26),
}


def apply_phase_action_mask(
    candidate_action: np.ndarray,
    hold_action: np.ndarray,
    active_action_groups: Iterable[str],
) -> np.ndarray:
    """Keep inactive action groups at their phase-entry command.

    SmolVLA always predicts all 26 action dimensions.  The drawer task is
    intentionally serialized, so only groups declared active by the language
    phase are allowed to change; all other position targets remain controlled
    at the command captured when the phase began.
    """
    candidate = np.asarray(candidate_action, dtype=np.float32)
    hold = np.asarray(hold_action, dtype=np.float32)
    if candidate.shape != (26,) or hold.shape != (26,):
        raise ValueError(
            f"phase action mask requires candidate/hold shape=(26,), got "
            f"{candidate.shape}/{hold.shape}"
        )
    active = tuple(str(group) for group in active_action_groups)
    unknown = sorted(set(active) - set(ACTION_GROUP_SLICES))
    if unknown:
        raise ValueError(f"unknown active action groups: {unknown}")
    masked = hold.copy()
    for group in active:
        group_slice = ACTION_GROUP_SLICES[group]
        masked[group_slice] = candidate[group_slice]
    return masked

