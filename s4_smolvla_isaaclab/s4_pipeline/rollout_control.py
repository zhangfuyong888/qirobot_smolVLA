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


def rollout_hard_failure_reason(
    failure_condition: str,
    *,
    drawer_open_m: float,
    gate_config: dict,
) -> str | None:
    """Return a physical hard-failure reason independent of soft phase gates."""
    condition = str(failure_condition)
    if condition == "none":
        return None
    if condition != "drawer_open_min":
        raise ValueError(f"unknown rollout failure condition: {condition!r}")
    if gate_config.get("drawer_open_min") is None:
        raise ValueError("drawer_open_min failure condition requires gate_config.drawer_open_min")
    minimum = float(gate_config["drawer_open_min"])
    opening = float(drawer_open_m)
    if not np.isfinite(opening) or opening < minimum:
        return f"drawer={opening:.3f}<{minimum:.3f}"
    return None


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
