"""Small HDF5 writer for staged S4 demonstrations.

This module intentionally has no IsaacLab dependency. The simulator side should
collect numpy arrays and call this writer at episode boundaries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from . import hdf5_schema as schema


def _create_nested_dataset(group: h5py.Group, path: str, data: np.ndarray, **dataset_kwargs: Any) -> None:
    parent = group
    parts = path.split("/")
    for part in parts[:-1]:
        parent = parent.require_group(part)
    parent.create_dataset(parts[-1], data=data, **dataset_kwargs)


def _image_dataset_kwargs(data: np.ndarray) -> dict[str, Any]:
    if data.ndim != 4:
        return {}
    return {
        "compression": "gzip",
        "compression_opts": 4,
        "shuffle": True,
        "chunks": (1, *data.shape[1:]),
    }


@dataclass
class EpisodeBuffer:
    metadata: dict[str, Any] = field(default_factory=dict)
    actions: list[np.ndarray] = field(default_factory=list)
    full_joint_pos: list[np.ndarray] = field(default_factory=list)
    active_joint_pos: list[np.ndarray] = field(default_factory=list)
    chest_front_rgb: list[np.ndarray] = field(default_factory=list)
    task_descriptions: list[str] = field(default_factory=list)
    language_phase_ids: list[str] = field(default_factory=list)
    expert_phase_names: list[str] = field(default_factory=list)
    left_wrist_rgb: list[np.ndarray] = field(default_factory=list)
    right_wrist_rgb: list[np.ndarray] = field(default_factory=list)
    left_eef_pose: list[np.ndarray] = field(default_factory=list)
    right_eef_pose: list[np.ndarray] = field(default_factory=list)
    red_block_pose: list[np.ndarray] = field(default_factory=list)
    blue_block_pose: list[np.ndarray] = field(default_factory=list)
    plate_pose: list[np.ndarray] = field(default_factory=list)
    drawer_task_object_pose: list[np.ndarray] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.actions)

    def validate(self) -> None:
        lengths: dict[str, int] = {
            "actions": len(self.actions),
            "full_joint_pos": len(self.full_joint_pos),
            "chest_front_rgb": len(self.chest_front_rgb),
        }
        optional_sequences = {
            "active_joint_pos": self.active_joint_pos,
            "task_descriptions": self.task_descriptions,
            "language_phase_ids": self.language_phase_ids,
            "expert_phase_names": self.expert_phase_names,
            "left_wrist_rgb": self.left_wrist_rgb,
            "right_wrist_rgb": self.right_wrist_rgb,
            "left_eef_pose": self.left_eef_pose,
            "right_eef_pose": self.right_eef_pose,
            "red_block_pose": self.red_block_pose,
            "blue_block_pose": self.blue_block_pose,
            "plate_pose": self.plate_pose,
            "drawer_task_object_pose": self.drawer_task_object_pose,
        }
        lengths.update({name: len(values) for name, values in optional_sequences.items() if values})
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Episode arrays have mismatched lengths: {lengths}")
        if not self.actions:
            raise ValueError("EpisodeBuffer is empty")


class Hdf5DemoWriter:
    def __init__(
        self,
        path: Path,
        env_args: dict[str, Any],
        *,
        resume: bool = False,
        overwrite: bool = True,
    ):
        self.path = Path(path)
        self.env_args = env_args
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if resume and not self.path.exists():
            raise FileNotFoundError(f"Cannot resume missing HDF5 file: {self.path}")
        if self.path.exists() and not resume and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing HDF5: {self.path}. "
                "Use --resume to continue it, or choose a new --output path."
            )
        append = bool(resume and self.path.exists())
        self._file = h5py.File(self.path, "r+" if append else "w")
        if append:
            if "data" not in self._file:
                self._file.close()
                raise ValueError(f"Cannot resume HDF5 without /data group: {self.path}")
            self._data = self._file["data"]
            try:
                self._validate_resume_contract(env_args)
            except Exception:
                self._file.close()
                raise
            for name in list(self._data):
                if name.startswith("_pending_demo_"):
                    del self._data[name]
            self._file.flush()
            indices = [
                int(name.removeprefix("demo_"))
                for name in self._data
                if name.startswith("demo_") and name.removeprefix("demo_").isdigit()
            ]
            self._episode_index = max(indices, default=-1) + 1
        else:
            self._data = self._file.create_group("data")
            self._data.attrs["env_args"] = json.dumps(env_args, ensure_ascii=False)
            self._episode_index = 0

    @property
    def episode_count(self) -> int:
        return sum(name.startswith("demo_") for name in self._data)

    def _validate_resume_contract(self, env_args: dict[str, Any]) -> None:
        raw = self._data.attrs.get("env_args")
        if raw is None:
            raise ValueError(f"Cannot resume HDF5 without env_args: {self.path}")
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        previous = json.loads(raw) if isinstance(raw, str) else dict(raw)
        keys = (
            "task",
            "sim_dt",
            "record_every_n",
            "randomization",
            "distractor_cans_enabled",
            "distractor_assets",
            "grasp_can_nominal_position",
            "grasp_can_scale",
            "camera",
            "data_contract",
            "language_contract",
        )
        def comparable(key: str, value: Any) -> Any:
            if key != "randomization" or not isinstance(value, dict):
                return value
            value = json.loads(json.dumps(value))
            can_xy = value.get("can_xy", {})
            if isinstance(can_xy, dict):
                can_xy.pop("max_points_per_cell", None)
            return value

        mismatches = [
            key
            for key in keys
            if comparable(key, previous.get(key)) != comparable(key, env_args.get(key))
        ]
        if mismatches:
            raise ValueError(
                f"Cannot resume {self.path}: collection contract changed for {mismatches}. "
                "Use a new HDF5 file instead of mixing incompatible episodes."
            )

    def read_collection_state(self) -> dict[str, Any] | None:
        raw = self._data.attrs.get("collection_state")
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return json.loads(raw) if isinstance(raw, str) else dict(raw)

    def write_collection_state(self, state: dict[str, Any]) -> None:
        self._data.attrs["collection_state"] = json.dumps(state, ensure_ascii=False)
        self._file.flush()

    def write_episode(
        self,
        episode: EpisodeBuffer,
        *,
        collection_state: dict[str, Any] | None = None,
    ) -> str:
        episode.validate()
        name = f"demo_{self._episode_index}"
        pending_name = f"_pending_{name}"
        if pending_name in self._data:
            del self._data[pending_name]
        group = self._data.create_group(pending_name)
        if episode.metadata:
            group.attrs["episode_metadata"] = json.dumps(episode.metadata, ensure_ascii=False)
        _create_nested_dataset(group, schema.PROCESSED_ACTIONS, np.asarray(episode.actions, dtype=np.float32))
        _create_nested_dataset(group, schema.FULL_JOINT_POS, np.asarray(episode.full_joint_pos, dtype=np.float32))
        if episode.active_joint_pos:
            _create_nested_dataset(group, schema.ACTIVE_JOINT_POS, np.asarray(episode.active_joint_pos, dtype=np.float32))
        if episode.task_descriptions:
            string_dtype = h5py.string_dtype(encoding="utf-8")
            _create_nested_dataset(
                group,
                schema.TASK_DESCRIPTION,
                np.asarray(episode.task_descriptions, dtype=object),
                dtype=string_dtype,
            )
        if episode.language_phase_ids:
            string_dtype = h5py.string_dtype(encoding="utf-8")
            _create_nested_dataset(
                group,
                schema.LANGUAGE_PHASE_ID,
                np.asarray(episode.language_phase_ids, dtype=object),
                dtype=string_dtype,
            )
        if episode.expert_phase_names:
            string_dtype = h5py.string_dtype(encoding="utf-8")
            _create_nested_dataset(
                group,
                schema.EXPERT_PHASE_NAME,
                np.asarray(episode.expert_phase_names, dtype=object),
                dtype=string_dtype,
            )
        rgb = np.asarray(episode.chest_front_rgb, dtype=np.uint8)
        _create_nested_dataset(group, schema.CHEST_FRONT_RGB, rgb, **_image_dataset_kwargs(rgb))
        if episode.left_wrist_rgb:
            left_rgb = np.asarray(episode.left_wrist_rgb, dtype=np.uint8)
            _create_nested_dataset(group, schema.LEFT_WRIST_RGB, left_rgb, **_image_dataset_kwargs(left_rgb))
        if episode.right_wrist_rgb:
            right_rgb = np.asarray(episode.right_wrist_rgb, dtype=np.uint8)
            _create_nested_dataset(group, schema.RIGHT_WRIST_RGB, right_rgb, **_image_dataset_kwargs(right_rgb))
        if episode.left_eef_pose:
            _create_nested_dataset(group, schema.LEFT_EEF_POSE, np.asarray(episode.left_eef_pose, dtype=np.float32))
        if episode.right_eef_pose:
            _create_nested_dataset(group, schema.RIGHT_EEF_POSE, np.asarray(episode.right_eef_pose, dtype=np.float32))
        if episode.red_block_pose:
            _create_nested_dataset(group, schema.RED_BLOCK_POSE, np.asarray(episode.red_block_pose, dtype=np.float32))
        if episode.blue_block_pose:
            _create_nested_dataset(group, schema.BLUE_BLOCK_POSE, np.asarray(episode.blue_block_pose, dtype=np.float32))
        if episode.plate_pose:
            _create_nested_dataset(group, schema.PLATE_POSE, np.asarray(episode.plate_pose, dtype=np.float32))
        if episode.drawer_task_object_pose:
            _create_nested_dataset(
                group,
                schema.DRAWER_TASK_OBJECT_POSE,
                np.asarray(episode.drawer_task_object_pose, dtype=np.float32),
            )
        self._data.move(pending_name, name)
        if collection_state is not None:
            self._data.attrs["collection_state"] = json.dumps(collection_state, ensure_ascii=False)
        self._file.flush()
        self._episode_index += 1
        return name

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "Hdf5DemoWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
