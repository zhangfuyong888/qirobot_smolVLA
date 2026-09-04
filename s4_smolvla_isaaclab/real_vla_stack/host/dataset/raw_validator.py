from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ...common.contract import PolicyContract
from ...common.errors import DataValidationError
from .causal_resampler import SampleMap, build_sample_map, distribution
from .raw_reader import RawEpisode, discover_raw_episodes


def validate_raw_dataset(raw_root: Path, contract: PolicyContract) -> tuple[list[RawEpisode], list[SampleMap], dict[str, Any]]:
    episodes = discover_raw_episodes(raw_root)
    maps: list[SampleMap] = []
    expected_meta: dict[str, Any] | None = None
    for episode in episodes:
        meta = episode.meta
        actual = {
            "schema": meta.get("schema_version"),
            "active_arm": meta.get("active_arm"),
            "cameras": tuple(meta.get("cameras", [])),
            "state_dim": meta.get("state_spec", {}).get("dim"),
            "action_dim": meta.get("action_spec", {}).get("dim"),
            "action_semantics": meta.get("action_spec", {}).get("semantics"),
        }
        required = {
            "schema": contract.raw_schema_version,
            "active_arm": contract.active_arm,
            "cameras": contract.camera_sources,
            "state_dim": contract.state_dim,
            "action_dim": contract.action_dim,
            "action_semantics": contract.action_semantics,
        }
        if actual != required:
            raise DataValidationError(f"{episode.path}: raw contract mismatch\nactual={actual}\nexpected={required}")
        if expected_meta is None:
            expected_meta = actual
        elif actual != expected_meta:
            raise DataValidationError(f"{episode.path}: mixed raw contracts are forbidden")
        state = np.column_stack(
            (episode.trajectory["robot_state_arm_q"], episode.trajectory["robot_state_gripper"])
        )
        action = np.column_stack(
            (episode.trajectory["action_arm_target_q"], episode.trajectory["action_gripper_target"])
        )
        if state.shape[1:] != (8,) or action.shape[1:] != (8,):
            raise DataValidationError(f"{episode.path}: expected state/action 8D")
        if not np.isfinite(state).all() or not np.isfinite(action).all():
            raise DataValidationError(f"{episode.path}: state/action contains NaN or Inf")
        for source in contract.camera_sources:
            if not episode.camera_video(source).is_file():
                raise DataValidationError(f"{episode.path}: missing {source} video")
        maps.append(build_sample_map(episode, contract))
    ages = {
        source: np.concatenate([mapping.camera_age_ms[source] for mapping in maps])
        for source in contract.camera_sources
    }
    report = {
        "contract_sha256": contract.sha256,
        "episodes": len(episodes),
        "raw_frames": int(sum(len(ep.trajectory["action_timestamp_ns"]) for ep in episodes)),
        "converted_frames": int(sum(mapping.frame_count for mapping in maps)),
        "dropped_prefix_policy_frames": int(
            sum(mapping.dropped_prefix_policy_frames for mapping in maps)
        ),
        "camera_age_ms": {source: distribution(values) for source, values in ages.items()},
        "cross_camera_skew_ms": distribution(
            np.concatenate([mapping.cross_camera_skew_ms for mapping in maps])
        ),
        "action_motion_allowed_ratio": float(
            np.mean(
                np.concatenate(
                    [
                        np.asarray(ep.trajectory["action_motion_allowed"]).reshape(-1)[mapping.control_index]
                        for ep, mapping in zip(episodes, maps, strict=True)
                    ]
                )
            )
        ),
        "action_limited_ratio": float(
            np.mean(
                np.concatenate(
                    [
                        np.asarray(ep.trajectory["action_limited"]).reshape(-1)[mapping.control_index]
                        for ep, mapping in zip(episodes, maps, strict=True)
                    ]
                )
            )
        ),
        "episode_paths": [str(ep.path) for ep in episodes],
    }
    return episodes, maps, report


def write_raw_report(path: Path, report: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
