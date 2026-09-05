"""Reusable validation for one saved real-robot episode."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from real_vla import SCHEMA_VERSION
from real_vla.collection.episode_writer import _decoded_video_frames, _video_openable
from real_vla.collection.quality import QualityResult, evaluate_episode
from real_vla.config_loader import CollectionConfig
from real_vla.data.episode_reader import load_meta, load_trajectory


def _shape_note(
    trajectory: dict[str, np.ndarray],
    key: str,
    expected: tuple[int, ...],
) -> str | None:
    value = trajectory.get(key)
    if value is None:
        return f"missing trajectory field {key}"
    if tuple(np.asarray(value).shape) != expected:
        return f"{key} shape {tuple(np.asarray(value).shape)} expected {expected}"
    return None


def validate_saved_episode(
    episode_dir: Path,
    config: CollectionConfig,
) -> tuple[QualityResult, dict, dict[str, np.ndarray]]:
    """Load and fully validate one episode, including decoding both videos."""
    episode_dir = Path(episode_dir)
    meta = load_meta(episode_dir)
    trajectory = load_trajectory(episode_dir)
    forced_invalid: list[str] = []

    if meta.get("schema_version") != SCHEMA_VERSION:
        forced_invalid.append(
            f"schema_version {meta.get('schema_version')!r} expected {SCHEMA_VERSION!r}"
        )
    if meta.get("result") != "saved":
        forced_invalid.append(f"episode result is {meta.get('result')!r}, not 'saved'")
    if meta.get("quality_valid") is False:
        notes = meta.get("quality_notes") or ["stored metadata marks episode invalid"]
        forced_invalid.extend(f"stored invalid: {note}" for note in notes)

    state_ts = np.asarray(
        trajectory.get("robot_state_timestamp_ns", np.zeros((0,), dtype=np.int64))
    )
    action_ts = np.asarray(
        trajectory.get("action_timestamp_ns", np.zeros((0,), dtype=np.int64))
    )
    state_count = int(state_ts.size)
    action_count = int(action_ts.size)
    required_shapes = (
        ("robot_state_arm_q", (state_count, 7)),
        ("robot_state_gripper", (state_count, 1)),
        ("robot_state_valid", (state_count, 1)),
        ("action_arm_target_q", (action_count, 7)),
        ("action_gripper_target", (action_count, 1)),
        ("action_published", (action_count, 1)),
        ("action_fault_active", (action_count, 1)),
        ("action_input_valid", (action_count, 1)),
        ("collection_phase", (state_count, 1)),
    )
    for key, expected in required_shapes:
        note = _shape_note(trajectory, key, expected)
        if note is not None:
            forced_invalid.append(note)

    phase = np.asarray(trajectory.get("collection_phase", np.zeros((0,)))).reshape(-1)
    if phase.size and not ({0.0, 1.0} <= set(phase.astype(float).tolist())):
        forced_invalid.append("collection_phase does not contain teleop and return_home")

    camera_ts: dict[str, np.ndarray] = {}
    camera_seq: dict[str, np.ndarray] = {}
    for key, value in trajectory.items():
        if key.startswith("camera_") and key.endswith("_timestamp_ns"):
            name = key[len("camera_") : -len("_timestamp_ns")]
            camera_ts[name] = value
        elif key.startswith("camera_") and key.endswith("_capture_seq"):
            name = key[len("camera_") : -len("_capture_seq")]
            camera_seq[name] = value

    stored_cameras = set(meta.get("cameras") or [])
    if (
        len(stored_cameras) != 2
        or "head" not in stored_cameras
        or not any(name in {"wrist_left", "wrist_right"} for name in stored_cameras)
    ):
        forced_invalid.append(f"invalid stored camera contract {sorted(stored_cameras)}")
    if set(camera_ts) != stored_cameras:
        forced_invalid.append(
            f"camera streams {sorted(camera_ts)} expected {sorted(stored_cameras)}"
        )
    video_paths = {
        name: episode_dir / ("head.mkv" if name == "head" else f"{name}.mkv")
        for name in camera_ts
    }
    video_ok = {name: _video_openable(path) for name, path in video_paths.items()}
    decoded_frames = {
        name: _decoded_video_frames(path) for name, path in video_paths.items()
    }

    result = evaluate_episode(
        quality=config.quality,
        duration_s=float(meta.get("duration_s", 0.0)),
        arm_q=np.asarray(trajectory.get("robot_state_arm_q", np.zeros((0, 7)))),
        action_q=np.asarray(trajectory.get("action_arm_target_q", np.zeros((0, 7)))),
        state_ts=state_ts,
        action_ts=action_ts,
        camera_ts=camera_ts,
        camera_seq=camera_seq,
        writer_drops={k: int(v) for k, v in dict(meta.get("writer_drops") or {}).items()},
        video_ok=video_ok,
        decoded_video_frames=decoded_frames,
        state_valid=np.asarray(trajectory.get("robot_state_valid", np.zeros((0, 1)))),
        action_published=np.asarray(trajectory.get("action_published", np.zeros((0, 1)))),
        fault_active=np.asarray(trajectory.get("action_fault_active", np.zeros((0, 1)))),
        input_valid=np.asarray(trajectory.get("action_input_valid", np.zeros((0, 1)))),
        lowdim_drops=int(meta.get("lowdim_drops", 0)),
        t_start_ns=int(meta.get("t_start_ns", 0)),
        t_end_ns=int(meta.get("t_end_ns", 0)),
        forced_invalid_notes=forced_invalid,
    )
    return result, meta, trajectory
