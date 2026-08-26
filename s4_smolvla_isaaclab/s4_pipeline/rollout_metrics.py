"""Pure helpers for online rollout randomization and success-rate reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from s4_pipeline.randomization import StratifiedGrid2D, sample_separated_xy
from s4_pipeline.drawer_distractors import (
    DEFAULT_DISTRACTOR_RANGES,
    DEFAULT_DISTRACTOR_XY,
    DISTRACTOR_OBJECT_NAMES,
    GRASP_CAN_NOMINAL_POSITION,
)


_LONG_DRAWER_EXTENSION_PHASES = frozenset(
    {"left_hold_handle_pregrasp", "left_preload_handle", "left_hold_drawer_open", "left_close_drawer"}
)
_LONG_DRAWER_EXTENSION_LANGUAGE_PHASES = frozenset(
    {"left_pregrasp_handle", "left_acquire_handle", "left_pull_drawer", "left_close_drawer"}
)


def rollout_phase_extension_frames(
    phase: dict[str, Any],
    scripted_cfg: dict[str, Any],
    default_frames: int,
    *,
    drawer_frames: int = 80,
) -> int:
    """Return the state-gate extension budget for one rollout phase.

    The initial left-hand handle approach and the physical drawer pull need
    extra time to settle. Other phases retain the global extension budget.
    Dataset schedules contain task text rather than scripted phase names, so
    resolve the name through the current scripted task configuration.
    """
    extension_kind = str(phase.get("rollout_extension", ""))
    if extension_kind == "drawer":
        return max(int(drawer_frames), 0)
    if extension_kind == "default":
        return max(int(default_frames), 0)
    language_phase_id = str(phase.get("language_phase_id", ""))
    if language_phase_id in _LONG_DRAWER_EXTENSION_LANGUAGE_PHASES:
        return max(int(drawer_frames), 0)
    phase_task = str(phase.get("task", ""))
    phase_cfg = next(
        (
            item
            for item in scripted_cfg.get("phases", [])
            if str(item.get("task", "")) == phase_task
        ),
        {},
    )
    if str(phase_cfg.get("name", "")) in _LONG_DRAWER_EXTENSION_PHASES:
        return max(int(drawer_frames), 0)
    return max(int(default_frames), 0)


def resolve_randomization_cfg(
    scripted_cfg: dict[str, Any],
    *,
    randomize_task: bool,
    can_x_range: tuple[float, float] | list[float] | None = None,
    can_y_range: tuple[float, float] | list[float] | None = None,
) -> dict[str, Any]:
    """Merge YAML can randomization with optional CLI range overrides.

    The drawer initial opening is deterministic and lives under ``drawer`` in
    the scripted config. When ``randomize_task`` is False, only the can range
    collapses to zero.
    """
    base = dict(scripted_cfg.get("randomization", {}) or {})
    can_cfg = dict(base.get("can_xy", {}) or {})
    drawer_cfg = dict(scripted_cfg.get("drawer", {}) or {})

    if can_x_range is not None:
        if len(can_x_range) != 2:
            raise ValueError(f"--can-x-range expects 2 values, got {can_x_range!r}")
        can_cfg["x_range"] = [float(can_x_range[0]), float(can_x_range[1])]
        can_cfg["enabled"] = True
    if can_y_range is not None:
        if len(can_y_range) != 2:
            raise ValueError(f"--can-y-range expects 2 values, got {can_y_range!r}")
        can_cfg["y_range"] = [float(can_y_range[0]), float(can_y_range[1])]
        can_cfg["enabled"] = True
    if not randomize_task:
        can_cfg["x_range"] = [0.0, 0.0]
        can_cfg["y_range"] = [0.0, 0.0]
        can_cfg["enabled"] = False
    else:
        # Preserve the YAML/CLI can switch. Missing can_xy.enabled defaults on
        # so collection and rollout match the random-can recipe.
        can_cfg.setdefault("enabled", True)
        can_cfg.setdefault("x_range", [0.0, 0.0])
        can_cfg.setdefault("y_range", [0.0, 0.0])

    resolved = dict(base)
    resolved["can_xy"] = can_cfg
    resolved["fixed_drawer_initial_open_m"] = float(drawer_cfg.get("initial_open_m", 0.0))
    resolved["enabled"] = bool(randomize_task)
    return resolved


def make_randomization_rng(seed: int = 42):
    """Create the shared RNG for a rollout experiment.

    The experiment seed stays fixed (default 42) whether randomization is on or
    off. Multi-episode diversity comes from advancing this shared stream.
    """
    import numpy as np

    return np.random.default_rng(int(seed))


def make_can_grid_sampler(random_cfg: dict[str, Any], rng: Any) -> StratifiedGrid2D | None:
    """Build the stateful stratified sampler requested by the task config."""
    can_cfg = random_cfg.get("can_xy", {}) or {}
    if (
        not bool(random_cfg.get("enabled", False))
        or not bool(can_cfg.get("enabled", False))
        or str(can_cfg.get("sampling", "uniform")) != "stratified_grid"
    ):
        return None
    cells = can_cfg.get("grid_cells", [5, 5])
    if not isinstance(cells, (list, tuple)) or len(cells) != 2:
        raise ValueError("randomization.can_xy.grid_cells must contain [x_cells, y_cells]")
    return StratifiedGrid2D(
        rng,
        x_range=tuple(float(value) for value in can_cfg.get("x_range", [-0.05, 0.05])),
        y_range=tuple(float(value) for value in can_cfg.get("y_range", [-0.05, 0.05])),
        cells_x=int(cells[0]),
        cells_y=int(cells[1]),
    )


def sample_randomization(
    random_cfg: dict[str, Any],
    *,
    seed: int = 42,
    rng: Any | None = None,
    can_grid_sampler: StratifiedGrid2D | None = None,
) -> dict[str, Any]:
    """Sample can XY offset for one episode.

    The drawer always uses the deterministic initial opening copied into the
    resolved config. ``seed`` is recorded as-is.
    """
    import numpy as np

    generator = rng if rng is not None else np.random.default_rng(int(seed))
    can_cfg = random_cfg.get("can_xy", {}) or {}
    distractor_cfg = random_cfg.get("distractor_cans", {}) or {}
    can_nominal = random_cfg.get("grasp_can_nominal_position", GRASP_CAN_NOMINAL_POSITION)

    can_x = 0.0
    can_y = 0.0
    drawer_open = float(random_cfg.get("fixed_drawer_initial_open_m", 0.0))
    grid_sample = None
    if bool(random_cfg.get("enabled", False)):
        if bool(can_cfg.get("enabled", False)):
            if can_grid_sampler is not None:
                grid_sample = can_grid_sampler.sample()
                can_x, can_y = (float(value) for value in grid_sample.xy)
            else:
                x_range = can_cfg.get("x_range", [0.0, 0.0])
                y_range = can_cfg.get("y_range", [0.0, 0.0])
                can_x = float(generator.uniform(*x_range))
                can_y = float(generator.uniform(*y_range))

    distractor_positions: dict[str, list[float]] = {}
    if bool(random_cfg.get("distractor_cans_enabled", False)):
        # Presence comes from the dataset contract (collection-time switch). Pose
        # randomization follows --randomize-task; the live YAML enabled flag only
        # gates new recording / scene spawn, not replay of older distractor demos.
        if bool(random_cfg.get("enabled", False)):
            points = sample_separated_xy(
                generator,
                ranges=distractor_cfg.get(
                    "ranges",
                    DEFAULT_DISTRACTOR_RANGES,
                ),
                forbidden_xy=[
                    [float(can_nominal[0]) + can_x, float(can_nominal[1]) + can_y]
                ],
                min_center_distance=float(distractor_cfg.get("min_center_distance_m", 0.16)),
            )
            region_order = generator.permutation(len(DISTRACTOR_OBJECT_NAMES))
        else:
            points = np.asarray(DEFAULT_DISTRACTOR_XY, dtype=np.float32)
            region_order = np.arange(len(DISTRACTOR_OBJECT_NAMES))
        distractor_positions = {
            name: points[int(region_order[index])].tolist()
            for index, name in enumerate(DISTRACTOR_OBJECT_NAMES)
        }

    result: dict[str, Any] = {
        "seed": int(seed),
        "can_x_offset_m": can_x,
        "can_y_offset_m": can_y,
        "drawer_open_m": drawer_open,
        "distractor_can_xy": distractor_positions,
    }
    if grid_sample is not None:
        result.update(
            can_grid_cell=[grid_sample.cell_x, grid_sample.cell_y],
            can_grid_cycle=grid_sample.cycle,
            can_grid_index_in_cycle=grid_sample.index_in_cycle,
        )
    return result


def evaluate_drawer_success(
    *,
    drawer_open_m: float,
    can_world_z_m: float,
    success_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Apply the active task final success criteria."""
    drawer_limit = float(success_cfg.get("drawer_open_abs_max", 0.04))
    can_limits = success_cfg.get("can_world_z", {}) or {}
    can_min = float(can_limits.get("min_m", 1.00))
    can_max = float(can_limits.get("max_m", 1.04))
    drawer_ok = abs(float(drawer_open_m)) < drawer_limit
    can_ok = can_min < float(can_world_z_m) < can_max
    return {
        "drawer_ok": drawer_ok,
        "can_ok": can_ok,
        "success": bool(drawer_ok and can_ok),
        "drawer_open_m": float(drawer_open_m),
        "drawer_limit_m": drawer_limit,
        "can_world_z_m": float(can_world_z_m),
        "can_z_min_m": can_min,
        "can_z_max_m": can_max,
    }


def checkpoint_step_tag(checkpoint: str | Path) -> str:
    """Extract a short checkpoint label such as ``ckpt360000`` from a path."""
    path = Path(checkpoint)
    for part in [path.name, *[p.name for p in path.parents]]:
        if part.isdigit():
            return f"ckpt{part}"
    return "ckptunknown"


def build_rollout_run_name(
    *,
    randomize_task: bool,
    episodes: int,
    checkpoint: str | Path,
    timestamp: str | None = None,
) -> str:
    """Build a unique, readable run folder name under ``outputs/eval``."""
    from datetime import datetime

    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = f"rand{int(episodes)}" if bool(randomize_task) and int(episodes) > 1 else (
        "rand" if bool(randomize_task) else "det"
    )
    return f"rollout_{stamp}_{mode}_{checkpoint_step_tag(checkpoint)}"


def resolve_rollout_run_dir(
    *,
    eval_root: Path,
    checkpoint: str | Path,
    episodes: int,
    randomize_task: bool,
    output_dir: Path | None = None,
    output_video: Path | None = None,
    timestamp: str | None = None,
) -> Path:
    """Resolve the single directory that stores one rollout run's artifacts.

    Layout example::

        outputs/eval/rollout_20260808_113645_rand20_ckpt360000/
          ep001.avi
          ep001_actions.csv
          ep001_actions.png
          ...
          summary.json

    Priority:
    1. ``--output-dir``
    2. ``--output-video`` stem used as a folder under its parent
       (``.../foo.avi`` → ``.../foo/``)
    3. auto name under ``eval_root``
    """
    if output_dir is not None:
        run_dir = Path(output_dir).expanduser()
    elif output_video is not None:
        video = Path(output_video).expanduser()
        # Treat a path ending with "/" or without a media suffix as a directory.
        if str(output_video).endswith(("/", "\\")) or video.suffix.lower() not in {
            ".avi",
            ".mp4",
            ".mov",
            ".mkv",
        }:
            run_dir = video
        else:
            run_dir = video.parent / video.stem
    else:
        run_dir = Path(eval_root).expanduser() / build_rollout_run_name(
            randomize_task=randomize_task,
            episodes=episodes,
            checkpoint=checkpoint,
            timestamp=timestamp,
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir.resolve()


def episode_artifact_paths(
    run_dir: Path,
    *,
    episode_index: int,
    episodes: int,
    video_suffix: str = ".avi",
) -> dict[str, Path]:
    """Resolve per-episode video/csv/png paths inside one shared run directory.

    Single-episode: ``rollout.avi`` / ``rollout_actions.csv`` / ``rollout_actions.png``.
    Multi-episode: ``ep001.avi`` ... all under the same ``run_dir``.
    """
    run_dir = Path(run_dir)
    if int(episodes) <= 1:
        stem = "rollout"
    else:
        stem = f"ep{episode_index + 1:03d}"
    video = run_dir / f"{stem}{video_suffix}"
    return {
        "video": video,
        "diagnostics_csv": run_dir / f"{stem}_actions.csv",
        "diagnostics_plot": run_dir / f"{stem}_actions.png",
    }


def default_summary_json_path(run_dir: Path) -> Path:
    return Path(run_dir) / "summary.json"


def aggregate_rollout_summary(
    episode_results: list[dict[str, Any]],
    *,
    checkpoint: str | Path,
    randomize_task: bool,
    base_seed: int,
    randomization: dict[str, Any],
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build the machine-readable multi-episode success-rate report."""
    total = len(episode_results)
    successes = sum(1 for row in episode_results if row.get("success"))
    completes = sum(1 for row in episode_results if row.get("complete"))
    both = sum(1 for row in episode_results if row.get("success") and row.get("complete"))
    return {
        "checkpoint": str(checkpoint),
        "output_dir": str(output_dir) if output_dir is not None else None,
        "episodes": total,
        "randomize_task": bool(randomize_task),
        "seed": int(base_seed),
        "randomization": {
            "variables": ["can_xy_offset_m", "distractor_can_xy"],
            "can_xy": randomization.get("can_xy", {}),
            "fixed_drawer_initial_open_m": float(
                randomization.get("fixed_drawer_initial_open_m", 0.0)
            ),
        },
        "success_count": successes,
        "complete_count": completes,
        "complete_and_success_count": both,
        "success_rate": (successes / total) if total else 0.0,
        "complete_rate": (completes / total) if total else 0.0,
        "complete_and_success_rate": (both / total) if total else 0.0,
        "episodes_detail": episode_results,
    }


def write_summary_json(summary: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
