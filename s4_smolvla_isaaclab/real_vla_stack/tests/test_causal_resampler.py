from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from real_vla_stack.common.config import load_pipeline_config
from real_vla_stack.common.errors import DataValidationError
from real_vla_stack.host.dataset.causal_resampler import build_sample_map
from real_vla_stack.host.dataset.raw_reader import RawEpisode


def episode() -> RawEpisode:
    timestamps = np.arange(0, 301_000_000, 25_000_000, dtype=np.int64)
    count = len(timestamps)
    data = {
        "robot_state_timestamp_ns": timestamps,
        "action_timestamp_ns": timestamps.copy(),
        "collection_phase": np.zeros((count, 1)),
        "camera_head_timestamp_ns": np.arange(10_000_000, 301_000_000, 30_000_000, dtype=np.int64),
        "camera_wrist_right_timestamp_ns": np.arange(12_000_000, 301_000_000, 30_000_000, dtype=np.int64),
        "robot_state_valid": np.ones((count, 1)),
        "action_published": np.ones((count, 1)),
        "action_motion_allowed": np.ones((count, 1)),
        "action_fault_active": np.zeros((count, 1)),
        "action_input_valid": np.ones((count, 1)),
    }
    return RawEpisode(Path("episode"), "session", {}, data)


def test_resampler_is_strictly_causal_and_20hz() -> None:
    contract = load_pipeline_config().contract
    raw = episode()
    mapping = build_sample_map(raw, contract)
    assert np.all(np.diff(mapping.policy_timestamp_ns) == 50_000_000)
    action_ts = raw.trajectory["action_timestamp_ns"]
    assert np.all(action_ts[mapping.control_index] <= mapping.policy_timestamp_ns)
    for source, indices in mapping.camera_indices.items():
        camera_ts = raw.trajectory[f"camera_{source}_timestamp_ns"]
        assert np.all(camera_ts[indices] <= action_ts[mapping.control_index])


def test_resampler_rejects_future_or_stale_camera_contract() -> None:
    raw = episode()
    contract = replace(load_pipeline_config().contract, max_camera_age_ms=1.0)
    with pytest.raises(DataValidationError, match="camera age"):
        build_sample_map(raw, contract)


def test_unclutched_hold_frames_are_not_rejected() -> None:
    raw = episode()
    raw.trajectory["action_motion_allowed"][3:6] = 0
    assert build_sample_map(raw, load_pipeline_config().contract).frame_count > 0
