from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ...common.config import PipelineConfig
from ..inference.policy_runner import PolicyRunner


OFFSETS = (-10, -5, -2, 0, 2, 5)


def _as_rgb_uint8(image: Any) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim == 3 and value.shape[0] == 3:
        value = np.transpose(value, (1, 2, 0))
    if np.issubdtype(value.dtype, np.floating) and float(np.max(value)) <= 1.0:
        value = value * 255.0
    return np.clip(value, 0, 255).astype(np.uint8)


def _event_frames(actions: np.ndarray) -> dict[str, list[int]]:
    closed = actions[:, 7] >= 0.5
    grasps = (np.flatnonzero(~closed[:-1] & closed[1:]) + 1).tolist()
    releases = (np.flatnonzero(closed[:-1] & ~closed[1:]) + 1).tolist()
    pull: list[int] = []
    for grasp in grasps:
        deltas = np.linalg.norm(np.diff(actions[:, :7], axis=0), axis=1)
        candidates = np.flatnonzero(deltas[max(grasp, 0) :] >= 0.01)
        if len(candidates):
            pull.append(int(grasp + candidates[0] + 1))
    return {"grasp": grasps, "pull": pull, "release": releases}


def probe_checkpoint_behavior(
    config: PipelineConfig,
    checkpoint: Path,
    *,
    samples_per_observation: int = 10,
    max_episodes: int | None = None,
) -> dict[str, Any]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset_root = config.host_path_value("lerobot_root") / str(config.host["dataset"]["repo_id"])
    dataset = LeRobotDataset(repo_id=dataset_root.name, root=str(dataset_root), video_backend="pyav")
    runner = PolicyRunner(checkpoint, config.contract, device=str(config.host["server"]["device"]))
    episode_indices = np.asarray(dataset.hf_dataset["episode_index"], dtype=np.int64)
    actions = np.asarray(dataset.hf_dataset["action"], dtype=np.float32)
    records: list[dict[str, Any]] = []
    for episode in np.unique(episode_indices)[:max_episodes]:
        global_indices = np.flatnonzero(episode_indices == episode)
        episode_actions = actions[global_indices]
        events = _event_frames(episode_actions)
        for event_name, event_values in events.items():
            for onset in event_values[:1]:
                for offset in OFFSETS:
                    local_index = int(np.clip(onset + offset, 0, len(global_indices) - 1))
                    sample = dataset[int(global_indices[local_index])]
                    images = {key: _as_rgb_uint8(sample[key]) for key in config.contract.camera_keys}
                    predictions = np.stack(
                        [
                            runner.predict_chunk(
                                np.asarray(sample["observation.state"], dtype=np.float32),
                                images,
                                config.contract.task,
                            )
                            for _ in range(int(samples_per_observation))
                        ]
                    )
                    pred_first = predictions[:, :5, :7]
                    gt_first = episode_actions[local_index : local_index + 5, :7]
                    usable = min(pred_first.shape[1], gt_first.shape[0])
                    gt_direction = np.sign(gt_first[usable - 1] - gt_first[0]) if usable > 1 else np.zeros(7)
                    pred_direction = (
                        np.sign(pred_first[:, usable - 1] - pred_first[:, 0])
                        if usable > 1
                        else np.zeros((len(predictions), 7))
                    )
                    records.append(
                        {
                            "episode_index": int(episode),
                            "event": event_name,
                            "offset": offset,
                            "frame_index": local_index,
                            "gt_gripper": float(episode_actions[local_index, 7]),
                            "pred_gripper_first10_mean": float(predictions[:, :10, 7].mean()),
                            "pred_gripper_first10_max": float(predictions[:, :10, 7].max()),
                            "p_gripper_gt_0_5": float(np.mean(predictions[:, :10, 7] > 0.5)),
                            "p_gripper_gt_0_6": float(np.mean(predictions[:, :10, 7] > 0.6)),
                            "p_gripper_gt_0_65": float(np.mean(predictions[:, :10, 7] > 0.65)),
                            "joint_first_step_mae": float(
                                np.mean(
                                    np.abs(
                                        predictions[:, 0, :7]
                                        - episode_actions[local_index, :7]
                                    )
                                )
                            ),
                            "joint_first5_mae": float(
                                np.mean(np.abs(pred_first[:, :usable] - gt_first[None, :usable]))
                            ),
                            "joint_direction_agreement": float(
                                np.mean(pred_direction == gt_direction[None, :])
                            ),
                        }
                    )
    return {
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "samples_per_observation": int(samples_per_observation),
        "records": records,
    }
