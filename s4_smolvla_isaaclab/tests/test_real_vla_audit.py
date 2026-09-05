from __future__ import annotations

from pathlib import Path

import numpy as np

from real_vla.scripts.audit_dataset import _quarantine_target, _review_notes


def test_review_notes_accept_drawer_like_action() -> None:
    trajectory = {
        "action_gripper_target": np.asarray([[0.0], [1.0], [1.0], [0.0]]),
        "action_arm_target_q": np.asarray(
            [[0.0] * 7, [0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]]
        ),
    }
    assert _review_notes(trajectory) == []


def test_review_notes_flag_missing_release_and_motion() -> None:
    trajectory = {
        "action_gripper_target": np.ones((4, 1)),
        "action_arm_target_q": np.zeros((4, 7)),
    }
    notes = _review_notes(trajectory)
    assert "gripper did not reopen after closing" in notes
    assert any("very little arm motion" in note for note in notes)


def test_quarantine_preserves_session_relative_path(tmp_path: Path) -> None:
    root = tmp_path / "data"
    episode = root / "session_a" / "episodes" / "episode_000001"
    episode.mkdir(parents=True)
    quarantine = root / "quarantine" / "stamp"
    assert _quarantine_target(root, episode, quarantine) == (
        quarantine / "session_a" / "episodes" / "episode_000001"
    )
