#!/usr/bin/env python3
"""Validate HDF5 or LeRobotDataset data against the active task contract."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s4_pipeline.config import load_project_config
from s4_pipeline.language_phases import load_language_phase_contract
from tasks import get_task_spec
from tasks.loading import load_yaml


def _fail(message: str) -> None:
    raise ValueError(message)


def _active_language_contract(cfg):
    task_spec = get_task_spec(cfg.dataset.task_id)
    if task_spec.scripted_config is None:
        _fail(f"Task has no scripted language contract: {cfg.dataset.task_id}")
    return load_language_phase_contract(load_yaml(task_spec.scripted_config))


def _validate_portable_contract(contract: dict, cfg, language_contract) -> None:
    """Validate the conversion artifact consumed by training and rollout."""
    if contract.get("schema_version") != cfg.dataset.schema_version:
        _fail(f"dataset schema={contract.get('schema_version')!r}, expected={cfg.dataset.schema_version!r}")
    if contract.get("action_semantics") != cfg.dataset.action_semantics:
        _fail(f"dataset action semantics mismatch: {contract.get('action_semantics')}")
    if int(contract.get("state_dim", -1)) != cfg.features.state_dim:
        _fail(f"dataset contract state_dim={contract.get('state_dim')}")
    if int(contract.get("action_dim", -1)) != cfg.features.action_dim:
        _fail(f"dataset contract action_dim={contract.get('action_dim')}")
    if int(contract.get("fps", -1)) != cfg.dataset.fps:
        _fail(f"dataset contract fps={contract.get('fps')}, expected={cfg.dataset.fps}")
    expected_camera_paths = list(cfg.raw.get("dataset", {}).get("camera_paths", []))
    if contract.get("camera_paths") != expected_camera_paths:
        _fail(
            f"dataset contract camera_paths={contract.get('camera_paths')!r}, "
            f"expected={expected_camera_paths!r}"
        )
    if contract.get("language_contract_version") != language_contract.version:
        _fail(f"dataset language contract={contract.get('language_contract_version')!r}")
    if contract.get("language_phases") != language_contract.as_portable_records():
        _fail("dataset language phase definitions do not match the active scripted config")
    from s4_pipeline.drawer_distractors import (
        GRASP_CAN_NOMINAL_POSITION,
        GRASP_CAN_SCALE,
        asset_contract,
        distractor_cans_enabled_from_scripted,
    )

    scripted_cfg = load_yaml(get_task_spec(cfg.dataset.task_id).scripted_config)
    distractors_enabled = distractor_cans_enabled_from_scripted(scripted_cfg)
    expected_scene = {
        "distractor_cans_enabled": distractors_enabled,
        "distractor_assets": asset_contract() if distractors_enabled else [],
        "grasp_can_nominal_position": list(GRASP_CAN_NOMINAL_POSITION),
        "grasp_can_scale": list(GRASP_CAN_SCALE),
    }
    for key, expected in expected_scene.items():
        if contract.get(key) != expected:
            _fail(f"dataset scene contract {key}={contract.get(key)!r}, expected={expected!r}")


def _validate_task_sequences(
    task_pairs: list[tuple[int, str]],
    episode_indices: list[int],
    task_indices: list[int],
    expected_tasks: list[str],
) -> None:
    """Validate temporal phase order without assuming categorical IDs are ordered."""
    index_to_task = {int(index): str(task) for index, task in task_pairs}
    if len(index_to_task) != len(task_pairs):
        _fail(f"tasks.parquet contains duplicate task_index values: {task_pairs}")
    actual_tasks = list(index_to_task.values())
    if len(actual_tasks) != len(set(actual_tasks)) or set(actual_tasks) != set(expected_tasks):
        _fail(f"dataset language task set mismatch: actual={actual_tasks}, expected={expected_tasks}")

    transitions_by_episode: dict[int, list[str]] = {}
    for episode, task_index in zip(episode_indices, task_indices, strict=True):
        episode = int(episode)
        task_index = int(task_index)
        if task_index not in index_to_task:
            _fail(f"episode={episode} references unknown task_index={task_index}")
        task = index_to_task[task_index]
        transitions = transitions_by_episode.setdefault(episode, [])
        if not transitions or transitions[-1] != task:
            transitions.append(task)
    for episode, transitions in sorted(transitions_by_episode.items()):
        if transitions != expected_tasks:
            _fail(
                f"episode={episode} language phase sequence mismatch: "
                f"actual={transitions}, expected={expected_tasks}"
            )


def _check_hdf5(path: Path, cfg) -> tuple[int, int]:
    import h5py
    import numpy as np
    from data.hdf5_schema import (
        ACTIVE_JOINT_POS,
        CHEST_FRONT_RGB,
        LEFT_WRIST_RGB,
        PROCESSED_ACTIONS,
        RIGHT_WRIST_RGB,
        DRAWER_TASK_OBJECT_POSE,
    )
    from data.lerobot_conversion import resolve_demo_language_tasks

    files = [path] if path.is_file() else sorted(path.rglob("*.hdf5"))
    if not files:
        _fail(f"No HDF5 files under {path}")
    episodes = frames = 0
    grid_samples: list[tuple[int, int, int]] = []
    expected_grid_shape: tuple[int, int] | None = None
    language_contract = _active_language_contract(cfg)
    for file in files:
        with h5py.File(file, "r") as stream:
            raw_env_args = stream["data"].attrs.get("env_args")
            env_args = {}
            if raw_env_args is not None:
                if isinstance(raw_env_args, bytes):
                    raw_env_args = raw_env_args.decode("utf-8")
                env_args = json.loads(raw_env_args) if isinstance(raw_env_args, str) else dict(raw_env_args)
                grid_cells = env_args.get("randomization", {}).get("can_xy", {}).get("grid_cells")
                if isinstance(grid_cells, list) and len(grid_cells) == 2:
                    shape = (int(grid_cells[0]), int(grid_cells[1]))
                    if expected_grid_shape is not None and shape != expected_grid_shape:
                        _fail(f"HDF5 files use different grid shapes: {expected_grid_shape} vs {shape}")
                    expected_grid_shape = shape
            for name, group in stream["data"].items():
                if not name.startswith("demo_"):
                    continue
                action = group[PROCESSED_ACTIONS]
                if action.ndim != 2 or action.shape[1] != cfg.features.action_dim:
                    _fail(f"{file}:{name} action shape={action.shape}")
                count = action.shape[0]
                if ACTIVE_JOINT_POS not in group:
                    _fail(f"{file}:{name} missing {ACTIVE_JOINT_POS}")
                active_state = group[ACTIVE_JOINT_POS]
                if active_state.shape != (count, cfg.features.active_state_dim):
                    _fail(f"{file}:{name} active state shape={active_state.shape}")
                for key, feature in zip(
                    (CHEST_FRONT_RGB, LEFT_WRIST_RGB, RIGHT_WRIST_RGB), cfg.features.camera_keys, strict=True
                ):
                    image = group[key]
                    expected = (count, *cfg.features.camera_shapes[feature])
                    if image.shape != expected:
                        _fail(f"{file}:{name}:{key} shape={image.shape}, expected={expected}")
                if not np.isfinite(action[:]).all():
                    _fail(f"{file}:{name} contains NaN/Inf actions")
                if not np.isfinite(active_state[:]).all():
                    _fail(f"{file}:{name} contains NaN/Inf active states")
                if DRAWER_TASK_OBJECT_POSE in group:
                    object_pose = group[DRAWER_TASK_OBJECT_POSE]
                    if object_pose.shape != (count, 7) or not np.isfinite(object_pose[:]).all():
                        _fail(f"{file}:{name} invalid drawer task object pose shape/data={object_pose.shape}")
                resolve_demo_language_tasks(
                    group,
                    frame_count=count,
                    default_task=cfg.dataset.task,
                    language_contract=language_contract,
                    source=f"{file}:{name}",
                )
                raw_metadata = group.attrs.get("episode_metadata")
                if raw_metadata is not None:
                    if isinstance(raw_metadata, bytes):
                        raw_metadata = raw_metadata.decode("utf-8")
                    metadata = json.loads(raw_metadata)
                    randomization = metadata.get("randomization", {})
                    if env_args.get("task") == "drawer_insert_close":
                        final_success = metadata.get("final_success")
                        if not isinstance(final_success, dict) or not final_success.get("accepted", False):
                            _fail(f"{file}:{name} is missing an accepted final_success record")
                        if not final_success.get("can_in_drawer", False):
                            _fail(f"{file}:{name} has invalid final_success={final_success}")
                    cell = randomization.get("can_grid_cell")
                    if cell is not None:
                        if not isinstance(cell, list) or len(cell) != 2:
                            _fail(f"{file}:{name} invalid can_grid_cell={cell!r}")
                        grid_samples.append(
                            (int(randomization.get("can_grid_cycle", 0)), int(cell[0]), int(cell[1]))
                        )
                episodes += 1
                frames += count
    if grid_samples:
        if len(grid_samples) != episodes:
            _fail(f"Grid metadata only present for {len(grid_samples)}/{episodes} HDF5 episodes")
        by_cycle: dict[int, list[tuple[int, int]]] = {}
        for cycle, cell_x, cell_y in grid_samples:
            by_cycle.setdefault(cycle, []).append((cell_x, cell_y))
        for cycle, cells in sorted(by_cycle.items()):
            if len(cells) != len(set(cells)):
                _fail(f"Grid cycle {cycle} contains duplicate accepted cells: {cells}")
        if expected_grid_shape is not None:
            expected_cells = {
                (x, y)
                for x in range(expected_grid_shape[0])
                for y in range(expected_grid_shape[1])
            }
            cells_per_cycle = len(expected_cells)
            if episodes % cells_per_cycle == 0:
                expected_cycles = episodes // cells_per_cycle
                if len(by_cycle) != expected_cycles:
                    _fail(
                        f"Expected {expected_cycles} complete grid cycle(s) for {episodes} episodes, "
                        f"found cycles={sorted(by_cycle)}"
                    )
                for cycle, cells in sorted(by_cycle.items()):
                    if set(cells) != expected_cells:
                        missing = sorted(expected_cells - set(cells))
                        _fail(f"Grid cycle {cycle} is incomplete; missing cells={missing}")
    print(f"[OK] HDF5 files={len(files)} episodes={episodes} frames={frames} action=26D cameras=3 schema={cfg.dataset.schema_version}")
    if grid_samples:
        print(f"[OK] stratified-grid metadata episodes={len(grid_samples)} cycles={len(by_cycle)} no_duplicate_cells_per_cycle")
    return episodes, frames


def _check_lerobot(path: Path, cfg, checkpoint: Path | None) -> tuple[int, int]:
    import av
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    info = json.loads((path / "meta/info.json").read_text(encoding="utf-8"))
    if cfg.dataset.action_semantics != "absolute_joint_target":
        _fail(f"unsupported action semantics: {cfg.dataset.action_semantics}")
    features = info["features"]
    if int(info["fps"]) != cfg.dataset.fps:
        _fail(f"fps={info['fps']}, expected={cfg.dataset.fps}")
    for key in ("observation.state", "action"):
        if features[key]["shape"] != [26]:
            _fail(f"{key} shape={features[key]['shape']}, expected=[26]")
    for key in cfg.features.camera_keys:
        if features[key]["shape"] != list(cfg.features.camera_shapes[key]):
            _fail(f"{key} shape={features[key]['shape']}")
    parquet_files = sorted((path / "data").rglob("*.parquet"))
    if not parquet_files:
        _fail("No frame parquet files")
    tables = [pq.read_table(p, columns=["timestamp", "episode_index", "frame_index", "task_index", "observation.state", "action"]) for p in parquet_files]
    table = pa.concat_tables(tables)
    for key in ("observation.state", "action"):
        values = np.asarray(table.column(key).to_pylist(), dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 26 or not np.isfinite(values).all():
            _fail(f"{key} invalid shape or NaN/Inf: {values.shape}")
    timestamps = table.column("timestamp").to_pylist()
    episodes = table.column("episode_index").to_pylist()
    frames = table.column("frame_index").to_pylist()
    previous: dict[int, tuple[float, int]] = {}
    for timestamp, episode, frame in zip(timestamps, episodes, frames, strict=True):
        ep = int(episode)
        if ep in previous and (float(timestamp) <= previous[ep][0] or int(frame) != previous[ep][1] + 1):
            _fail(f"non-monotonic episode={ep} frame={frame} timestamp={timestamp}")
        previous[ep] = (float(timestamp), int(frame))
    video_files = []
    for key in cfg.features.camera_keys:
        candidates = sorted((path / "videos" / key).rglob("*.mp4"))
        if not candidates:
            _fail(f"No videos for {key}")
        video_files.extend(candidates)
        encoded_frames = 0
        for candidate in candidates:
            try:
                with av.open(str(candidate)) as container:
                    stream = container.streams.video[0]
                    encoded_frames += int(stream.frames)
                    first = next(container.decode(video=0))
                    if (first.height, first.width, 3) != cfg.features.camera_shapes[key]:
                        _fail(f"decoded {key} shape={(first.height, first.width, 3)} in {candidate}")
                    if stream.duration is not None:
                        container.seek(max(int(stream.duration) - 2, 0), stream=stream, backward=True)
                        last = None
                        for last in container.decode(video=0):
                            pass
                        if last is None:
                            _fail(f"could not decode the tail of video: {candidate}")
            except (av.error.FFmpegError, EOFError, StopIteration) as exc:
                _fail(f"video decode failed for {candidate}: {exc}")
        if encoded_frames != table.num_rows:
            _fail(f"video frame count for {key}={encoded_frames}, expected={table.num_rows}")
    task_rows = pq.read_table(path / "meta/tasks.parquet")
    if task_rows.num_rows == 0:
        _fail("tasks.parquet is empty")
    language_contract = _active_language_contract(cfg)
    task_pairs = [
        (int(index), str(task))
        for index, task in zip(
            task_rows.column("task_index").to_pylist(),
            task_rows.column("task").to_pylist(),
            strict=True,
        )
    ]
    expected_tasks = [phase.task for phase in language_contract.phases]
    _validate_task_sequences(
        task_pairs,
        [int(value) for value in table.column("episode_index").to_pylist()],
        [int(value) for value in table.column("task_index").to_pylist()],
        expected_tasks,
    )
    if checkpoint:
        model = checkpoint / "pretrained_model" if (checkpoint / "pretrained_model").is_dir() else checkpoint
        ckpt = json.loads((model / "config.json").read_text(encoding="utf-8"))
        inputs = ckpt["input_features"]
        expected_inputs = {"observation.state", *cfg.features.camera_keys}
        if set(inputs) != expected_inputs or ckpt["output_features"]["action"]["shape"] != [26]:
            _fail(f"checkpoint feature contract mismatch: {model}")
        if inputs["observation.state"]["shape"] != [cfg.features.state_dim]:
            _fail(f"checkpoint state shape mismatch: {inputs['observation.state']['shape']}")
        for key in cfg.features.camera_keys:
            expected_chw = [
                cfg.features.camera_shapes[key][2],
                cfg.features.camera_shapes[key][0],
                cfg.features.camera_shapes[key][1],
            ]
            if inputs[key]["shape"] != expected_chw:
                _fail(f"checkpoint {key} shape={inputs[key]['shape']}, expected={expected_chw}")
        required_checkpoint_files = (
            "policy_preprocessor.json",
            "policy_postprocessor.json",
            "policy_preprocessor_step_5_normalizer_processor.safetensors",
            "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        )
        missing = [name for name in required_checkpoint_files if not (model / name).is_file()]
        if missing:
            _fail(f"checkpoint missing inference processors: {missing}")
    contract_path = path / "meta/s4_contract.json"
    if not contract_path.is_file():
        _fail(f"dataset is missing portable contract: {contract_path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    _validate_portable_contract(contract, cfg, language_contract)
    if checkpoint:
        checkpoint_contract = next(
            (
                parent / "s4_dataset_contract.json"
                for parent in [checkpoint, *checkpoint.parents]
                if (parent / "s4_dataset_contract.json").is_file()
            ),
            None,
        )
        if checkpoint_contract is None:
            _fail(f"checkpoint has no s4_dataset_contract.json provenance: {checkpoint}")
        trained_contract = json.loads(checkpoint_contract.read_text(encoding="utf-8"))
        if trained_contract != contract:
            _fail(f"checkpoint dataset contract differs from {contract_path}")
    print(f"[OK] LeRobotDataset episodes={len(previous)} frames={table.num_rows} fps={info['fps']} action/state=26D schema={cfg.dataset.schema_version}")
    print(f"[OK] cameras=3 shape=480x680x3 decoded_files={len(video_files)} tasks={task_rows.num_rows}")
    if checkpoint:
        print(f"[OK] checkpoint compatible: {checkpoint}")
    return len(previous), table.num_rows


def _check_failure_summary(
    path: Path,
    *,
    expected_episodes: int | None,
    max_failed_attempts: int | None,
    allow_skipped_grid_cells: bool,
    expected_hdf5: Path | None = None,
) -> None:
    if not path.is_file():
        _fail(f"Failure summary does not exist: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not summary.get("completed", False):
        _fail(f"Collection did not complete according to {path}")
    accepted = int(summary.get("accepted_episodes", -1))
    target = int(summary.get("target_episodes", -1))
    if accepted != target:
        _fail(f"Failure summary accepted={accepted}, target={target}")
    if expected_episodes is not None and (accepted != expected_episodes or target != expected_episodes):
        _fail(
            f"Failure summary accepted/target={accepted}/{target}, expected={expected_episodes}"
        )
    if expected_hdf5 is not None:
        reported_hdf5 = summary.get("hdf5_path")
        if not reported_hdf5 or Path(str(reported_hdf5)).resolve() != expected_hdf5.resolve():
            _fail(
                f"Failure summary HDF5={reported_hdf5!r}, checked HDF5={str(expected_hdf5)!r}"
            )
    failures = int(summary.get("failed_attempts", -1))
    if failures < 0:
        _fail("Failure summary has no valid failed_attempts count")
    if max_failed_attempts is not None and failures > max_failed_attempts:
        _fail(f"failed_attempts={failures} exceeds allowed maximum={max_failed_attempts}")
    skipped = summary.get("skipped_grid_cells", [])
    if skipped and not allow_skipped_grid_cells:
        _fail(f"Collection skipped {len(skipped)} grid cell(s): {skipped}")
    failure_log = Path(str(summary.get("failure_log", "")))
    if not failure_log.is_file():
        _fail(f"Failure event log referenced by summary is missing: {failure_log}")
    lines = [line for line in failure_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != failures:
        _fail(f"Failure log lines={len(lines)}, summary failed_attempts={failures}")
    for line_number, line in enumerate(lines, 1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            _fail(f"Invalid failure JSONL line {failure_log}:{line_number}: {exc}")
        for key in ("failure_type", "reason", "phase_name", "can_position_world_m"):
            if key not in event:
                _fail(f"Failure event {failure_log}:{line_number} missing {key}")
    print(
        f"[OK] failure report completed=true accepted={accepted}/{target} "
        f"failed_attempts={failures} skipped_grid_cells={len(skipped)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate active-task HDF5 or LeRobotDataset data.")
    parser.add_argument("path", nargs="?", type=Path)
    parser.add_argument("--hdf5", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--failure-summary", type=Path)
    parser.add_argument("--max-failed-attempts", type=int)
    parser.add_argument("--allow-skipped-grid-cells", action="store_true")
    args = parser.parse_args()
    cfg = load_project_config()
    default = cfg.dataset.staging_root if args.hdf5 else cfg.dataset.lerobot_root / cfg.dataset.repo_id.split("/")[-1]
    path = (args.path or default).expanduser().resolve()
    if not path.exists():
        _fail(f"Path does not exist: {path}")
    if args.expected_episodes is not None and args.expected_episodes <= 0:
        _fail("--expected-episodes must be positive")
    if args.max_failed_attempts is not None and args.max_failed_attempts < 0:
        _fail("--max-failed-attempts must be non-negative")
    episodes, _frames = (
        _check_hdf5(path, cfg) if args.hdf5 else _check_lerobot(path, cfg, args.checkpoint)
    )
    if args.expected_episodes is not None and episodes != args.expected_episodes:
        _fail(f"Dataset episodes={episodes}, expected={args.expected_episodes}")
    if args.failure_summary is not None:
        _check_failure_summary(
            args.failure_summary.expanduser().resolve(),
            expected_episodes=args.expected_episodes,
            max_failed_attempts=args.max_failed_attempts,
            allow_skipped_grid_cells=args.allow_skipped_grid_cells,
            expected_hdf5=path if args.hdf5 and path.is_file() else None,
        )


if __name__ == "__main__":
    main()
