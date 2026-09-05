from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from real_vla.scripts import audit_dataset
from real_vla.scripts.audit_dataset import (
    _quarantine_target,
    _require_video_decoder,
    _review_notes,
)


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


def test_missing_cv2_is_reported_as_a_dependency_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_cv2(name: str):
        raise ModuleNotFoundError("No module named 'cv2'", name=name)

    monkeypatch.setattr(audit_dataset.importlib, "import_module", missing_cv2)
    with pytest.raises(RuntimeError, match=r"OpenCV.*cv2.*required"):
        _require_video_decoder()


def test_quarantine_moves_pending_data_and_reports_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "data"
    pending = root / "session_a" / "pending" / "episode_000001.tmp"
    pending.mkdir(parents=True)
    report = tmp_path / "audit.json"
    monkeypatch.setattr(audit_dataset, "_require_video_decoder", lambda: None)

    assert audit_dataset.main(
        [str(root), "--quarantine-invalid", "--yes", "--report", str(report)]
    ) == 0

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["counts"]["PENDING"] == 1
    record = payload["pending_records"][0]
    assert record["path"] == str(pending)
    assert record["action"].startswith("quarantined:")
    target = Path(record["action"].split(":", 1)[1])
    assert target.is_dir()
    assert not pending.exists()
