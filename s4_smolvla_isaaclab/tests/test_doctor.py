from __future__ import annotations

import json
from pathlib import Path

from scripts.doctor import resolve_latest_checkpoint


def _checkpoint(root: Path, name: str) -> Path:
    model = root / "checkpoints" / name / "pretrained_model"
    model.mkdir(parents=True)
    (model / "config.json").write_text(json.dumps({"step": name}), encoding="utf-8")
    return model


def test_doctor_prefers_complete_last_checkpoint(tmp_path: Path):
    _checkpoint(tmp_path, "50000")
    last = _checkpoint(tmp_path, "last")
    assert resolve_latest_checkpoint(tmp_path) == last


def test_doctor_uses_latest_complete_numeric_checkpoint(tmp_path: Path):
    _checkpoint(tmp_path, "50000")
    latest = _checkpoint(tmp_path, "500000")
    (tmp_path / "checkpoints/600000").mkdir(parents=True)
    assert resolve_latest_checkpoint(tmp_path) == latest


def test_doctor_returns_expected_last_path_before_training(tmp_path: Path):
    assert resolve_latest_checkpoint(tmp_path) == tmp_path / "checkpoints/last/pretrained_model"
