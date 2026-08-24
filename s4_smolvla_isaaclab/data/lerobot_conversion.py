"""Local HDF5 to LeRobotDataset conversion utilities.

The implementation mirrors the relevant BenchHub flow but stays in this
project. It imports LeRobot lazily so IsaacLab-side smoke checks can run
without the training environment installed.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import TYPE_CHECKING
import uuid

import h5py
import numpy as np

from . import hdf5_schema as schema

if TYPE_CHECKING:
    from s4_pipeline.language_phases import LanguagePhaseContract

RIGHT_ONLY_ACTION_SLICE = slice(13, 26)
SAFE_DATASET_LEAF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def safe_dataset_root(output_root: Path, repo_id: str) -> Path:
    """Resolve a dataset leaf without permitting traversal or broad targets."""
    leaf = str(repo_id).split("/")[-1]
    if not SAFE_DATASET_LEAF.fullmatch(leaf) or leaf in {".", ".."}:
        raise ValueError(f"unsafe dataset repo/name leaf: {leaf!r}")
    parent = Path(os.path.abspath(Path(output_root).expanduser()))
    project_root = Path(__file__).resolve().parents[1]
    data_root = Path(os.environ.get("S4_DATA_ROOT", project_root / "datasets")).expanduser().resolve(strict=False)
    if parent in {Path("/"), Path.home().resolve(), project_root.resolve(), data_root}:
        raise ValueError(f"unsafe broad dataset output root: {parent}")
    current = Path(parent.anchor)
    for part in parent.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"dataset output root contains symlink component: {current}")
    resolved_parent = parent.resolve(strict=False)
    target = (resolved_parent / leaf).resolve(strict=False)
    if target.parent != resolved_parent or target == resolved_parent:
        raise ValueError(f"dataset target must be a strict child of {resolved_parent}: {target}")
    return target


def validate_overwrite_dataset_target(dataset_root: Path) -> None:
    """Reject recursive deletion unless the target is an actual LeRobotDataset."""
    if dataset_root.is_symlink() or not dataset_root.is_dir():
        raise ValueError(f"refusing to overwrite non-directory dataset target: {dataset_root}")
    required_markers = (dataset_root / "meta" / "info.json", dataset_root / "data", dataset_root / "videos")
    if not all(marker.exists() for marker in required_markers):
        raise ValueError(
            f"refusing to recursively overwrite a directory that is not recognizably a "
            f"LeRobotDataset: {dataset_root}"
        )


def publish_converted_dataset(staging_root: Path, dataset_root: Path, *, overwrite: bool) -> None:
    """Publish a completed sibling dataset while preserving the old target until swap."""
    validate_overwrite_dataset_target(staging_root)
    backup_root: Path | None = None
    if dataset_root.exists():
        if not overwrite:
            raise FileExistsError(f"LeRobotDataset appeared during conversion: {dataset_root}")
        validate_overwrite_dataset_target(dataset_root)
        backup_root = dataset_root.parent / f".{dataset_root.name}.backup.{uuid.uuid4().hex}"
        os.replace(dataset_root, backup_root)
    try:
        os.replace(staging_root, dataset_root)
    except BaseException:
        if backup_root is not None and backup_root.exists() and not dataset_root.exists():
            os.replace(backup_root, dataset_root)
        raise
    if backup_root is not None:
        shutil.rmtree(backup_root)


def _read_utf8_sequence(demo: h5py.Group, path: str, frame_count: int) -> list[str] | None:
    if path not in demo:
        return None
    raw_values = np.asarray(demo[path])
    if len(raw_values) != frame_count:
        raise ValueError(f"{path} length={len(raw_values)}, expected={frame_count}")
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in raw_values]


def resolve_demo_language_tasks(
    demo: h5py.Group,
    *,
    frame_count: int,
    default_task: str,
    language_contract: "LanguagePhaseContract | None",
    source: str,
) -> tuple[list[str], list[str | None]]:
    """Resolve legacy or current HDF5 language fields to canonical macro prompts."""
    try:
        recorded_tasks = _read_utf8_sequence(demo, schema.TASK_DESCRIPTION, frame_count)
        phase_ids = _read_utf8_sequence(demo, schema.LANGUAGE_PHASE_ID, frame_count)
        expert_names = _read_utf8_sequence(demo, schema.EXPERT_PHASE_NAME, frame_count)
    except ValueError as exc:
        raise ValueError(f"{source}: {exc}") from exc

    if language_contract is None:
        return recorded_tasks or [str(default_task)] * frame_count, [None] * frame_count
    if recorded_tasks is None and phase_ids is None and expert_names is None:
        raise ValueError(
            f"{source}: no per-frame language metadata; cannot build the active "
            f"{language_contract.version} contract"
        )

    canonical_tasks: list[str] = []
    canonical_ids: list[str | None] = []
    for frame_index in range(frame_count):
        try:
            if phase_ids is not None:
                phase = language_contract.for_id(phase_ids[frame_index])
            elif expert_names is not None:
                phase = language_contract.for_expert_phase(expert_names[frame_index])
            elif recorded_tasks is not None:
                phase = language_contract.resolve_recorded_task(recorded_tasks[frame_index])
            else:  # Guarded above, retained for type narrowing.
                raise ValueError("missing language metadata")
            if expert_names is not None:
                expert_phase = language_contract.for_expert_phase(expert_names[frame_index])
                if expert_phase.id != phase.id:
                    raise ValueError(
                        f"language_phase_id={phase.id!r} conflicts with "
                        f"expert_phase_name={expert_names[frame_index]!r}"
                    )
            if recorded_tasks is not None:
                recorded_phase = language_contract.resolve_recorded_task(recorded_tasks[frame_index])
                if recorded_phase.id != phase.id:
                    raise ValueError(
                        f"language_phase_id={phase.id!r} conflicts with "
                        f"task_description={recorded_tasks[frame_index]!r}"
                    )
        except ValueError as exc:
            raise ValueError(f"{source}:frame={frame_index}: {exc}") from exc
        canonical_tasks.append(phase.task)
        canonical_ids.append(phase.id)
    return canonical_tasks, canonical_ids


def video_info(fps: int) -> dict:
    return {
        "video.fps": float(fps),
        "video.codec": "h264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "has_audio": False,
    }


def discover_hdf5_files(root_path: Path) -> list[Path]:
    root_path = Path(root_path)
    if root_path.is_file() and root_path.suffix == ".hdf5":
        return [root_path]
    if root_path.is_dir():
        return sorted(root_path.glob("*.hdf5"))
    raise FileNotFoundError(f"HDF5 root does not exist: {root_path}")


def validate_recording_fps(hdf5_files: list[Path], expected_fps: int) -> None:
    """Reject HDF5 inputs whose recorded timebase differs from the dataset timebase."""
    recorded_rates: list[tuple[Path, float]] = []
    for hdf5_path in hdf5_files:
        with h5py.File(hdf5_path, "r") as f:
            raw_env_args = f["data"].attrs.get("env_args")
            if raw_env_args is None:
                continue
            if isinstance(raw_env_args, bytes):
                raw_env_args = raw_env_args.decode("utf-8")
            env_args = json.loads(raw_env_args) if isinstance(raw_env_args, str) else dict(raw_env_args)
            record_fps = env_args.get("record_fps")
            if record_fps is None:
                sim_dt = env_args.get("sim_dt")
                record_every_n = env_args.get("record_every_n")
                if sim_dt and record_every_n:
                    record_fps = 1.0 / (float(sim_dt) * int(record_every_n))
            if record_fps is not None:
                recorded_rates.append((hdf5_path, float(record_fps)))

    mismatches = [
        (path, rate)
        for path, rate in recorded_rates
        if not math.isclose(rate, float(expected_fps), rel_tol=1e-4, abs_tol=1e-4)
    ]
    if mismatches:
        details = "\n".join(f"  {path}: {rate:.3f} Hz" for path, rate in mismatches)
        raise ValueError(
            f"HDF5 recording rate does not match configured LeRobot fps={expected_fps}:\n{details}\n"
            "For the 120 Hz simulator, record with --record-every-n 6 to produce 20 Hz data. "
            "Do not relabel an existing recording with a different fps."
        )


def read_recording_contract(hdf5_path: Path) -> dict:
    """Read portable simulator-side metadata embedded in an HDF5 recording."""
    with h5py.File(hdf5_path, "r") as stream:
        raw = stream["data"].attrs.get("env_args")
    if raw is None:
        return {}
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw) if isinstance(raw, str) else dict(raw)


def validate_scene_contracts(hdf5_files: list[Path]) -> dict:
    """Reject mixing recordings made with different visual scene contracts."""
    first = read_recording_contract(hdf5_files[0])
    expected = {
        "distractor_cans_enabled": bool(first.get("distractor_cans_enabled", False)),
        "distractor_assets": list(first.get("distractor_assets", [])),
        "grasp_can_nominal_position": list(first.get("grasp_can_nominal_position", [])),
        "grasp_can_scale": list(first.get("grasp_can_scale", [])),
    }
    mismatches: list[str] = []
    for path in hdf5_files[1:]:
        contract = read_recording_contract(path)
        actual = {
            "distractor_cans_enabled": bool(contract.get("distractor_cans_enabled", False)),
            "distractor_assets": list(contract.get("distractor_assets", [])),
            "grasp_can_nominal_position": list(contract.get("grasp_can_nominal_position", [])),
            "grasp_can_scale": list(contract.get("grasp_can_scale", [])),
        }
        if actual != expected:
            mismatches.append(f"  {path}: {actual}")
    if mismatches:
        raise ValueError(
            "HDF5 files use different distractor scene contracts; do not merge visually "
            "inconsistent collection runs:\n" + "\n".join(mismatches)
        )
    return first


def inspect_first_demo(hdf5_path: Path, camera_path: str, control_mode: str) -> tuple[int, int, tuple[int, ...]]:
    with h5py.File(hdf5_path, "r") as f:
        demo_names = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))
        for demo_name in demo_names:
            demo = f["data"][demo_name]
            if schema.PROCESSED_ACTIONS in demo and schema.FULL_JOINT_POS in demo and camera_path in demo:
                if control_mode == "right_only":
                    action_dim = 13
                    state_dim = 13
                else:
                    action_dim = int(np.asarray(demo[schema.PROCESSED_ACTIONS]).shape[1])
                    state_path = schema.ACTIVE_JOINT_POS if schema.ACTIVE_JOINT_POS in demo else schema.FULL_JOINT_POS
                    state_dim = int(np.asarray(demo[state_path]).shape[1])
                camera_shape = tuple(np.asarray(demo[camera_path][0]).shape)
                return state_dim, action_dim, camera_shape
    raise ValueError(f"No valid demo found in {hdf5_path}")


def build_lerobot_features(
    camera_paths: list[str],
    state_dim: int,
    action_dim: int,
    camera_shape: tuple[int, ...],
    fps: int,
) -> dict:
    features = {
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": None},
        "action": {"dtype": "float32", "shape": (action_dim,), "names": None},
        "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
        "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
        "index": {"dtype": "int64", "shape": (1,), "names": None},
        "task_index": {"dtype": "int64", "shape": (1,), "names": None},
    }
    for camera_path in camera_paths:
        camera_name = camera_path.split("/")[-1]
        features[f"observation.images.{camera_name}"] = {
            "dtype": "video",
            "shape": camera_shape,
            "names": ["height", "width", "channel"],
            "video_info": video_info(fps),
        }
    return features


def convert_hdf5_to_lerobot(
    root_path: Path,
    output_root: Path,
    repo_id: str,
    task_description: str,
    robot_type: str,
    camera_paths: list[str] | None = None,
    fps: int = 30,
    overwrite: bool = False,
    control_mode: str = "right_only",
    language_contract: "LanguagePhaseContract | None" = None,
) -> Path:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    camera_paths = camera_paths or [schema.CHEST_FRONT_RGB]
    if control_mode not in {"right_only", "bimanual"}:
        raise ValueError(f"control_mode must be right_only or bimanual, got {control_mode!r}")
    hdf5_files = discover_hdf5_files(root_path)
    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 files found under {root_path}")
    validate_recording_fps(hdf5_files, fps)
    recording_contract = validate_scene_contracts(hdf5_files)

    state_dim, action_dim, camera_shape = inspect_first_demo(hdf5_files[0], camera_paths[0], control_mode)
    features = build_lerobot_features(camera_paths, state_dim, action_dim, camera_shape, fps=fps)
    # ``repo_id`` may carry a Hub-style namespace (for example ``local/foo``),
    # while local training and rollout consistently address the leaf dataset
    # directory. Never create an accidental extra namespace directory locally.
    dataset_root = safe_dataset_root(output_root, repo_id)
    if dataset_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"LeRobotDataset already exists: {dataset_root}\n"
                "Use --overwrite to rebuild it, or pass --repo-id/--output-root to write a new dataset."
            )
        validate_overwrite_dataset_target(dataset_root)
    staging_root = dataset_root.parent / f".{dataset_root.name}.converting.{uuid.uuid4().hex}"
    dataset = None
    published = False
    try:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=str(staging_root),
            fps=fps,
            robot_type=robot_type,
            features=features,
            video_backend="pyav",
        )

        for hdf5_path in hdf5_files:
            with h5py.File(hdf5_path, "r") as f:
                demo_names = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1]))
                for demo_name in demo_names:
                    demo = f["data"][demo_name]
                    if schema.PROCESSED_ACTIONS not in demo:
                        continue
                    actions = np.asarray(demo[schema.PROCESSED_ACTIONS])
                    if control_mode == "right_only":
                        actions = actions[:, RIGHT_ONLY_ACTION_SLICE]
                        states = np.asarray(demo[schema.ACTIVE_JOINT_POS])[:, RIGHT_ONLY_ACTION_SLICE]
                    else:
                        state_path = (
                            schema.ACTIVE_JOINT_POS
                            if schema.ACTIVE_JOINT_POS in demo
                            else schema.FULL_JOINT_POS
                        )
                        states = np.asarray(demo[state_path])
                    cameras = {path: np.asarray(demo[path]) for path in camera_paths}
                    lengths = {
                        "actions": len(actions),
                        "states": len(states),
                        **{path: len(values) for path, values in cameras.items()},
                    }
                    if len(set(lengths.values())) != 1:
                        raise ValueError(f"{hdf5_path}:{demo_name} frame lengths mismatch: {lengths}")
                    frame_count = len(actions)
                    frame_tasks, _frame_phase_ids = resolve_demo_language_tasks(
                        demo,
                        frame_count=frame_count,
                        default_task=task_description,
                        language_contract=language_contract,
                        source=f"{hdf5_path}:{demo_name}",
                    )
                    for i in range(frame_count):
                        frame = {
                            "observation.state": states[i].astype(np.float32),
                            "action": actions[i].astype(np.float32),
                            "task": frame_tasks[i],
                        }
                        for camera_path, values in cameras.items():
                            camera_name = camera_path.split("/")[-1]
                            frame[f"observation.images.{camera_name}"] = values[i]
                        dataset.add_frame(frame)
                    dataset.save_episode()
        dataset.finalize()
        portable_contract = {
            "schema_version": "s4_bimanual_v1",
            "action_semantics": "absolute_joint_target",
            "state_dim": state_dim,
            "action_dim": action_dim,
            "fps": int(fps),
            "camera_paths": list(camera_paths),
            "distractor_cans_enabled": bool(recording_contract.get("distractor_cans_enabled", False)),
            "distractor_assets": list(recording_contract.get("distractor_assets", [])),
            "grasp_can_nominal_position": list(recording_contract.get("grasp_can_nominal_position", [])),
            "grasp_can_scale": list(recording_contract.get("grasp_can_scale", [])),
        }
        if language_contract is not None:
            portable_contract["language_contract_version"] = language_contract.version
            portable_contract["language_phases"] = language_contract.as_portable_records()
        contract_path = staging_root / "meta" / "s4_contract.json"
        contract_path.write_text(json.dumps(portable_contract, indent=2) + "\n", encoding="utf-8")
        publish_converted_dataset(staging_root, dataset_root, overwrite=overwrite)
        published = True
    finally:
        dataset = None
        if not published and staging_root.exists():
            shutil.rmtree(staging_root)
    return dataset_root
