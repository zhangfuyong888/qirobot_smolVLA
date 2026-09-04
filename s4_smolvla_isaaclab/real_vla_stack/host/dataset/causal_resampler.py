from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...common.contract import PolicyContract
from ...common.errors import DataValidationError
from .raw_reader import RawEpisode


PHASE_IDS = {"teleop": 0, "return_home": 1}


@dataclass(frozen=True)
class SampleMap:
    policy_timestamp_ns: np.ndarray
    control_index: np.ndarray
    camera_indices: dict[str, np.ndarray]
    camera_age_ms: dict[str, np.ndarray]
    cross_camera_skew_ms: np.ndarray
    dropped_prefix_policy_frames: int = 0

    @property
    def frame_count(self) -> int:
        return int(self.control_index.size)


def _latest_before(timestamps: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.asarray(timestamps, dtype=np.int64), targets, side="right") - 1


def build_sample_map(episode: RawEpisode, contract: PolicyContract) -> SampleMap:
    data = episode.trajectory
    state_ts = np.asarray(data["robot_state_timestamp_ns"], dtype=np.int64)
    action_ts = np.asarray(data["action_timestamp_ns"], dtype=np.int64)
    if state_ts.shape != action_ts.shape or not np.array_equal(state_ts, action_ts):
        raise DataValidationError(f"{episode.path}: state/action timestamps differ")
    phase = np.asarray(data["collection_phase"]).reshape(-1).astype(np.int64)
    phase_ids = {PHASE_IDS[name] for name in contract.phase_filter}
    eligible = np.flatnonzero(np.isin(phase, list(phase_ids)))
    if eligible.size < 2:
        raise DataValidationError(f"{episode.path}: fewer than two eligible control samples")
    camera_timestamps = {
        source: np.asarray(data[f"camera_{source}_timestamp_ns"], dtype=np.int64)
        for source in contract.camera_sources
    }
    if any(values.size == 0 for values in camera_timestamps.values()):
        raise DataValidationError(f"{episode.path}: one or more cameras have no timestamps")
    step_ns = int(round(1.0e9 / contract.dataset_fps))
    nominal_start = int(action_ts[eligible[0]])
    causal_start = max(nominal_start, *(int(values[0]) for values in camera_timestamps.values()))
    valid_starts = eligible[action_ts[eligible] >= causal_start]
    if valid_starts.size == 0:
        raise DataValidationError(f"{episode.path}: cameras start after all eligible actions")
    grid_start = int(action_ts[valid_starts[0]])
    dropped_prefix = max(0, int(np.ceil((grid_start - nominal_start) / step_ns)))
    grid = np.arange(grid_start, action_ts[eligible[-1]] + 1, step_ns, dtype=np.int64)
    control = _latest_before(action_ts, grid)
    keep = np.isin(control, eligible)
    grid = grid[keep]
    control = control[keep]
    if grid.size < 2:
        raise DataValidationError(f"{episode.path}: causal resampling produced fewer than two frames")

    camera_indices: dict[str, np.ndarray] = {}
    camera_age: dict[str, np.ndarray] = {}
    selected_camera_ts: list[np.ndarray] = []
    control_time = action_ts[control]
    for source in contract.camera_sources:
        timestamps = camera_timestamps[source]
        indices = _latest_before(timestamps, control_time)
        if np.any(indices < 0):
            raise DataValidationError(f"{episode.path}: {source} has no causal frame at episode start")
        selected = timestamps[indices]
        age_ms = (control_time - selected) / 1.0e6
        if np.any(age_ms < 0) or float(np.max(age_ms)) > contract.max_camera_age_ms:
            raise DataValidationError(
                f"{episode.path}: {source} camera age max={float(np.max(age_ms)):.1f}ms "
                f"exceeds {contract.max_camera_age_ms:.1f}ms"
            )
        camera_indices[source] = indices.astype(np.int64)
        camera_age[source] = age_ms
        selected_camera_ts.append(selected)
    skew_ms = np.abs(selected_camera_ts[0] - selected_camera_ts[1]) / 1.0e6
    if float(np.max(skew_ms)) > contract.max_cross_camera_skew_ms:
        raise DataValidationError(
            f"{episode.path}: cross-camera skew max={float(np.max(skew_ms)):.1f}ms "
            f"exceeds {contract.max_cross_camera_skew_ms:.1f}ms"
        )

    hard_flags = {
        "robot_state_valid": True,
        "action_published": True,
        "action_fault_active": False,
        "action_input_valid": True,
    }
    for key, expected in hard_flags.items():
        values = np.asarray(data[key]).reshape(-1)[control] > 0.5
        ok = values if expected else ~values
        if not np.all(ok):
            raise DataValidationError(f"{episode.path}: selected policy frames violate {key}")
    motion_allowed = np.asarray(data["action_motion_allowed"]).reshape(-1)[control] > 0.5
    if not np.any(motion_allowed):
        raise DataValidationError(f"{episode.path}: episode contains no motion-enabled policy frames")
    return SampleMap(grid, control, camera_indices, camera_age, skew_ms, dropped_prefix)


def distribution(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }
