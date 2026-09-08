from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PERCENTILES = (50, 90, 95, 99, 100)


def _summary(values: np.ndarray) -> dict[str, list[float]]:
    if values.ndim != 2 or values.shape[1] != 7:
        raise ValueError(f"expected [N,7] arm dynamics, got {values.shape}")
    if len(values) == 0:
        return {f"p{value}": [0.0] * 7 for value in PERCENTILES}
    result = np.percentile(np.abs(values), PERCENTILES, axis=0)
    return {
        ("max" if percentile == 100 else f"p{percentile}"): row.tolist()
        for percentile, row in zip(PERCENTILES, result, strict=True)
    }


def analyze_action_dynamics(dataset_root: Path, *, fps: float) -> dict[str, Any]:
    """Measure action dynamics per episode so boundaries never create fake spikes."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    files = sorted(Path(dataset_root).expanduser().resolve().joinpath("data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files below {dataset_root}")
    table = pa.concat_tables(
        [pq.read_table(path, columns=["episode_index", "frame_index", "action"]) for path in files]
    )
    episodes: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
    for row in table.to_pylist():
        episodes[int(row["episode_index"])].append(
            (int(row["frame_index"]), np.asarray(row["action"], dtype=np.float64))
        )

    velocities: list[np.ndarray] = []
    accelerations: list[np.ndarray] = []
    behavior: list[dict[str, Any]] = []
    for episode, rows in sorted(episodes.items()):
        rows.sort(key=lambda item: item[0])
        actions = np.stack([item[1] for item in rows])
        arm = actions[:, :7]
        velocity = np.diff(arm, axis=0) * float(fps)
        acceleration = np.diff(velocity, axis=0) * float(fps)
        velocities.append(velocity)
        accelerations.append(acceleration)
        grasp = actions[:, 7] >= 0.5
        open_to_grasp = np.flatnonzero(~grasp[:-1] & grasp[1:]) + 1
        grasp_to_open = np.flatnonzero(grasp[:-1] & ~grasp[1:]) + 1
        direction = np.sign(np.diff(arm, axis=0))
        reversals = np.sum(direction[1:] * direction[:-1] < 0, axis=0) if len(direction) > 1 else np.zeros(7)
        behavior.append(
            {
                "episode_index": episode,
                "frames": len(actions),
                "duration_s": len(actions) / float(fps),
                "first_grasp_s": None if len(open_to_grasp) == 0 else float(open_to_grasp[0] / fps),
                "grasp_duration_s": float(np.count_nonzero(grasp) / fps),
                "release_s": None if len(grasp_to_open) == 0 else float(grasp_to_open[-1] / fps),
                "open_to_grasp_transitions": int(len(open_to_grasp)),
                "grasp_to_open_transitions": int(len(grasp_to_open)),
                "joint_path_length_rad": np.sum(np.abs(np.diff(arm, axis=0)), axis=0).tolist(),
                "joint_direction_reversals": reversals.astype(int).tolist(),
            }
        )

    velocity_all = np.concatenate(velocities) if velocities else np.empty((0, 7))
    acceleration_all = np.concatenate(accelerations) if accelerations else np.empty((0, 7))
    return {
        "dataset_root": str(Path(dataset_root).expanduser().resolve()),
        "fps": float(fps),
        "episodes": len(episodes),
        "joint_velocity_rad_s": _summary(velocity_all),
        "joint_acceleration_rad_s2": _summary(acceleration_all),
        "behavior": behavior,
    }
