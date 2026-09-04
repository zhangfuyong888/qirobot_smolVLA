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
    decoded_video_frames: dict[str, int] | None = None,
    state_valid: np.ndarray | None = None,
    action_published: np.ndarray | None = None,
    fault_active: np.ndarray | None = None,
    input_valid: np.ndarray | None = None,
    lowdim_drops: int = 0,
    t_start_ns: int | None = None,
    t_end_ns: int | None = None,
    forced_invalid_notes: list[str] | None = None,
) -> QualityResult:
    notes: list[str] = []
    warning = False
    valid = True
    decoded_video_frames = decoded_video_frames or {}

    for note in forced_invalid_notes or []:
        valid = False
        notes.append(str(note))

    if duration_s < quality.min_duration_s:
        valid = False
        notes.append(f"duration {duration_s:.2f}s below {quality.min_duration_s:.2f}s")

    if arm_q.size and not np.isfinite(arm_q).all():
        valid = False
        notes.append("arm_q contains non-finite values")
    if action_q.size and not np.isfinite(action_q).all():
        valid = False
        notes.append("action contains non-finite values")

    if state_ts.size != action_ts.size:
        valid = False
        notes.append(f"state/action count mismatch {state_ts.size}/{action_ts.size}")
    elif state_ts.size and not np.array_equal(
        np.asarray(state_ts, dtype=np.int64),
        np.asarray(action_ts, dtype=np.int64),
    ):
        valid = False
        notes.append("state/action timestamps do not match")
    if state_valid is not None and (
        np.asarray(state_valid).size != state_ts.size
        or not np.all(np.asarray(state_valid, dtype=np.float64) > 0.5)
    ):
        valid = False
        notes.append("robot state contains stale or invalid samples")
    if action_published is not None and (
        np.asarray(action_published).size != action_ts.size
        or not np.all(np.asarray(action_published, dtype=np.float64) > 0.5)
    ):
        valid = False
        notes.append("action contains commands that were not published")
    if fault_active is not None and np.any(
        np.asarray(fault_active, dtype=np.float64) > 0.5
    ):
        valid = False
        notes.append("controller fault occurred during recording")
    if input_valid is not None and (
        np.asarray(input_valid).size != action_ts.size
        or not np.all(np.asarray(input_valid, dtype=np.float64) > 0.5)
    ):
        valid = False
        notes.append("Quest input contains stale samples during teleoperation")
    if int(lowdim_drops) > 0:
        valid = False
        notes.append(f"low-dimensional writer dropped {int(lowdim_drops)} samples")

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
        elif label in {"state", "action"} and gap > quality.robot_state_invalid_ms:
            valid = False
            notes.append(f"{label} gap {gap:.0f}ms")

        coverage_limit_ms = (
            quality.camera_gap_invalid_ms
            if label in camera_ts
            else quality.robot_state_invalid_ms
        )
        if t_start_ns is not None and array.size:
            start_gap_ms = (int(array[0]) - int(t_start_ns)) / 1.0e6
            if start_gap_ms > coverage_limit_ms:
                valid = False
                notes.append(f"{label} starts {start_gap_ms:.0f}ms after episode")
        if t_end_ns is not None and array.size:
            end_gap_ms = (int(t_end_ns) - int(array[-1])) / 1.0e6
            if end_gap_ms > coverage_limit_ms:
                valid = False
                notes.append(f"{label} ends {end_gap_ms:.0f}ms before episode")

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
        decoded = decoded_video_frames.get(name)
        expected = np.asarray(camera_ts.get(name, ())).size
        if decoded is not None and int(decoded) != int(expected):
            valid = False
            notes.append(
                f"{name} decoded frames {int(decoded)} do not match timestamps {int(expected)}"
            )

    return QualityResult(valid=valid, warning=warning, notes=notes)
