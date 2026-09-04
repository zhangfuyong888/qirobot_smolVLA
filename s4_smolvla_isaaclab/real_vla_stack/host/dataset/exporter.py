from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from data.lerobot_conversion import publish_converted_dataset, safe_dataset_root

from ...common.config import PipelineConfig
from .raw_validator import validate_raw_dataset
from .video_decoder import decode_selected_rgb


def _git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _features(config: PipelineConfig, image_shape: tuple[int, int, int]) -> dict[str, Any]:
    contract = config.contract
    features: dict[str, Any] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (contract.state_dim,),
            "names": list(contract.state_names),
        },
        "action": {
            "dtype": "float32",
            "shape": (contract.action_dim,),
            "names": list(contract.action_names),
        },
    }
    for key in contract.camera_keys:
        features[key] = {
            "dtype": "video",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
            "video_info": {
                "video.fps": float(contract.dataset_fps),
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    return features


def convert_raw_to_lerobot(config: PipelineConfig, *, overwrite: bool = False) -> Path:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    raw_root = config.host_path_value("raw_root")
    output_root = config.host_path_value("lerobot_root")
    repo_id = str(config.host["dataset"]["repo_id"])
    episodes, mappings, report = validate_raw_dataset(raw_root, config.contract)
    shape_by_source = {
        source: (
            int(episodes[0].meta["camera_specs"][source]["height"]),
            int(episodes[0].meta["camera_specs"][source]["width"]),
            3,
        )
        for source in config.contract.camera_sources
    }
    if len(set(shape_by_source.values())) != 1:
        raise ValueError(f"LeRobot v1 exporter requires equal camera shapes: {shape_by_source}")
    dataset_root = safe_dataset_root(output_root, repo_id)
    staging = dataset_root.parent / f".{dataset_root.name}.converting.{uuid.uuid4().hex}"
    if dataset_root.exists() and not overwrite:
        raise FileExistsError(f"dataset exists: {dataset_root}; pass --overwrite to replace it")
    dataset = None
    published = False
    source_rows: list[dict[str, Any]] = []
    try:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=str(staging),
            fps=config.contract.dataset_fps,
            robot_type=config.contract.robot_type,
            features=_features(config, next(iter(shape_by_source.values()))),
            video_backend="pyav",
        )
        for converted_episode, (episode, mapping) in enumerate(zip(episodes, mappings, strict=True)):
            images = {
                source: decode_selected_rgb(episode.camera_video(source), mapping.camera_indices[source])
                for source in config.contract.camera_sources
            }
            state = np.column_stack(
                (episode.trajectory["robot_state_arm_q"], episode.trajectory["robot_state_gripper"])
            )[mapping.control_index].astype(np.float32)
            action = np.column_stack(
                (episode.trajectory["action_arm_target_q"], episode.trajectory["action_gripper_target"])
            )[mapping.control_index].astype(np.float32)
            for converted_frame, raw_index in enumerate(mapping.control_index.tolist()):
                frame: dict[str, Any] = {
                    "observation.state": state[converted_frame],
                    "action": action[converted_frame],
                    "task": config.contract.task,
                }
                for key, source in zip(config.contract.camera_keys, config.contract.camera_sources, strict=True):
                    frame[key] = images[source][converted_frame]
                dataset.add_frame(frame)
                row: dict[str, Any] = {
                    "episode_index": converted_episode,
                    "frame_index": converted_frame,
                    "raw_session": episode.session,
                    "raw_episode": episode.path.name,
                    "raw_control_index": int(raw_index),
                    "raw_control_timestamp_ns": int(episode.trajectory["action_timestamp_ns"][raw_index]),
                }
                for source in config.contract.camera_sources:
                    camera_index = int(mapping.camera_indices[source][converted_frame])
                    row[f"raw_{source}_frame_index"] = camera_index
                    row[f"raw_{source}_timestamp_ns"] = int(
                        episode.trajectory[f"camera_{source}_timestamp_ns"][camera_index]
                    )
                    row[f"{source}_age_ms"] = float(mapping.camera_age_ms[source][converted_frame])
                source_rows.append(row)
            dataset.save_episode()
        dataset.finalize()
        meta = staging / "meta"
        config.contract.write(meta / "s4_contract.json")
        report.update(
            {
                "repo_id": repo_id,
                "source_index_rows": len(source_rows),
                "project_git_commit": _git_head(Path(__file__).resolve().parents[4]),
                "lerobot_commit": _git_head(Path(__file__).resolve().parents[4] / "lerobot"),
            }
        )
        (meta / "s4_conversion_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.Table.from_pylist(source_rows), meta / "s4_source_index.parquet")
        publish_converted_dataset(staging, dataset_root, overwrite=overwrite)
        published = True
    finally:
        dataset = None
        if not published and staging.exists():
            shutil.rmtree(staging)
    return dataset_root
