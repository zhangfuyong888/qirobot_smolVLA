from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ...common.contract import PolicyContract
from ...common.errors import DataValidationError


def validate_lerobot_dataset(
    path: Path,
    contract: PolicyContract,
    *,
    smoke_load: bool = True,
    raw_root: Path | None = None,
) -> dict[str, Any]:
    import av
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(path).expanduser().resolve()
    actual_contract = PolicyContract.read(root / "meta" / "s4_contract.json")
    if actual_contract.sha256 != contract.sha256:
        raise DataValidationError("LeRobot dataset contract differs from active pipeline contract")
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    if int(info["fps"]) != contract.dataset_fps:
        raise DataValidationError(f"dataset fps={info['fps']}, expected={contract.dataset_fps}")
    features = info["features"]
    expected_keys = {"observation.state", "action", *contract.camera_keys}
    if not expected_keys.issubset(features):
        raise DataValidationError(f"dataset feature mismatch: missing={sorted(expected_keys - set(features))}")
    if features["observation.state"]["shape"] != [8] or features["action"]["shape"] != [8]:
        raise DataValidationError("dataset state/action must be 8D")
    data_files = sorted((root / "data").rglob("*.parquet"))
    if not data_files:
        raise DataValidationError("dataset has no parquet data")
    table = pa.concat_tables(
        [pq.read_table(p, columns=["episode_index", "frame_index", "observation.state", "action"]) for p in data_files]
    )
    states = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
    actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != 8 or not np.isfinite(states).all():
        raise DataValidationError("invalid observation.state values")
    if actions.ndim != 2 or actions.shape[1] != 8 or not np.isfinite(actions).all():
        raise DataValidationError("invalid action values")
    episodes = [int(v) for v in table.column("episode_index").to_pylist()]
    frames = [int(v) for v in table.column("frame_index").to_pylist()]
    previous: dict[int, int] = {}
    for episode, frame in zip(episodes, frames, strict=True):
        if frame != previous.get(episode, -1) + 1:
            raise DataValidationError(f"non-contiguous frame index episode={episode} frame={frame}")
        previous[episode] = frame
    tasks = pq.read_table(root / "meta" / "tasks.parquet").column("task").to_pylist()
    if tasks != [contract.task]:
        raise DataValidationError(f"dataset tasks={tasks}, expected exactly {[contract.task]}")
    video_frames: dict[str, int] = {}
    for key in contract.camera_keys:
        paths = sorted((root / "videos" / key).rglob("*.mp4"))
        if not paths:
            raise DataValidationError(f"missing encoded video for {key}")
        count = 0
        for video in paths:
            with av.open(str(video)) as container:
                decoded = 0
                for frame in container.decode(video=0):
                    if frame.format is None:
                        raise DataValidationError(f"undecodable video frame in {video}")
                    decoded += 1
                count += decoded
        if count != table.num_rows:
            raise DataValidationError(f"{key} decoded frames={count}, expected={table.num_rows}")
        video_frames[key] = count
    source_index = pq.read_table(root / "meta" / "s4_source_index.parquet")
    if source_index.num_rows != table.num_rows:
        raise DataValidationError("source index does not cover every converted frame")
    if smoke_load:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(repo_id=root.name, root=str(root), video_backend="pyav")
        if len(dataset) != table.num_rows:
            raise DataValidationError(f"LeRobotDataset len={len(dataset)}, expected={table.num_rows}")
        sample = dataset[min(len(dataset) - 1, 1)]
        if tuple(sample["observation.state"].shape) != (8,) or tuple(sample["action"].shape) != (8,):
            raise DataValidationError("LeRobotDataset smoke sample has wrong state/action shape")
        if raw_root is not None:
            from real_vla.data.episode_reader import load_trajectory

            rows = source_index.to_pylist()
            for index in sorted({0, len(rows) // 2, len(rows) - 1}):
                row = rows[index]
                episode_path = (
                    Path(raw_root) / row["raw_session"] / "episodes" / row["raw_episode"]
                )
                trajectory = load_trajectory(episode_path)
                raw_index = int(row["raw_control_index"])
                expected_state = np.concatenate(
                    (
                        trajectory["robot_state_arm_q"][raw_index],
                        np.asarray(trajectory["robot_state_gripper"][raw_index]).reshape(1),
                    )
                ).astype(np.float32)
                expected_action = np.concatenate(
                    (
                        trajectory["action_arm_target_q"][raw_index],
                        np.asarray(trajectory["action_gripper_target"][raw_index]).reshape(1),
                    )
                ).astype(np.float32)
                actual = dataset[index]
                if not np.allclose(np.asarray(actual["observation.state"]), expected_state):
                    raise DataValidationError(f"source audit state mismatch at converted index {index}")
                if not np.allclose(np.asarray(actual["action"]), expected_action):
                    raise DataValidationError(f"source audit action mismatch at converted index {index}")
    codecs = {
        key: str(features[key].get("info", {}).get("video.codec", "unknown"))
        for key in contract.camera_keys
    }
    return {
        "contract_sha256": contract.sha256,
        "episodes": len(previous),
        "frames": table.num_rows,
        "video_frames": video_frames,
        "fps": int(info["fps"]),
        "video_codecs": codecs,
        "source_audit_samples": 3 if raw_root is not None and smoke_load else 0,
    }
