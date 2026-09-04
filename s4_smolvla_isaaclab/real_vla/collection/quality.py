"""Lightweight episode quality checks. Never blocks robot control."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from real_vla.config_loader import QualityConfig


@dataclass
class QualityResult:
    valid: bool
    warning: bool
    notes: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        if not self.valid:
            return "QUALITY INVALID"
        if self.warning:
            return "QUALITY WARN"
        return "QUALITY PASS"


def _strictly_increasing(values: np.ndarray) -> bool:
    if values.size <= 1:
        return True
    return bool(np.all(np.diff(values.astype(np.int64)) > 0))


def _max_gap_ms(values: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    return float(np.max(np.diff(values.astype(np.int64))) / 1.0e6)


def evaluate_episode(
    *,
    quality: QualityConfig,
    duration_s: float,
    arm_q: np.ndarray,
    action_q: np.ndarray,
    state_ts: np.ndarray,
    action_ts: np.ndarray,
    camera_ts: dict[str, np.ndarray],
    camera_seq: dict[str, np.ndarray],
    writer_drops: dict[str, int],
    video_ok: dict[str, bool],
) -> QualityResult:
    notes: list[str] = []
    warning = False
    valid = True

    if duration_s < quality.min_duration_s:
        valid = False
        notes.append(f"duration {duration_s:.2f}s below {quality.min_duration_s:.2f}s")

    if arm_q.size and not np.isfinite(arm_q).all():
        valid = False
        notes.append("arm_q contains non-finite values")
    if action_q.size and not np.isfinite(action_q).all():
        valid = False
        notes.append("action contains non-finite values")

    for label, stamps in (("state", state_ts), ("action", action_ts), *camera_ts.items()):
        array = np.asarray(stamps, dtype=np.int64)
        if array.size == 0:
            valid = False
            notes.append(f"{label} has no samples")
            continue
        if not _strictly_increasing(array):
            valid = False
            notes.append(f"{label} timestamps are not strictly increasing")
        gap = _max_gap_ms(array)
        if label in camera_ts:
            if gap > quality.camera_gap_invalid_ms:
                valid = False
                notes.append(f"{label} gap {gap:.0f}ms")
            elif gap > quality.camera_gap_warning_ms:
                warning = True
                notes.append(f"{label} gap {gap:.0f}ms")
            if array.size < quality.min_camera_frames:
                valid = False
                notes.append(f"{label} frames {array.size} below {quality.min_camera_frames}")
        elif label == "state" and gap > quality.robot_state_invalid_ms:
            valid = False
            notes.append(f"robot state gap {gap:.0f}ms")

    if len(camera_ts) < 2:
        valid = False
        notes.append("need at least two cameras")

    for name, seq in camera_seq.items():
        array = np.asarray(seq, dtype=np.int64)
        if array.size >= 2:
            skipped = int(np.sum(np.diff(array) > 1))
            if skipped:
                warning = True
                notes.append(f"{name} capture_seq gaps={skipped}")

    for name, drops in writer_drops.items():
        if int(drops) > 0:
            warning = True
            notes.append(f"{name} writer dropped {drops} frames")

    for name, ok in video_ok.items():
        if not ok:
            valid = False
            notes.append(f"{name} video cannot be opened")

    return QualityResult(valid=valid, warning=warning, notes=notes)
