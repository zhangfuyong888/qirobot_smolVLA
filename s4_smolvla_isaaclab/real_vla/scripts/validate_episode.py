#!/usr/bin/env python
"""Re-run quality checks on a saved episode directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from real_vla.collection.quality import evaluate_episode  # noqa: E402
from real_vla.config_loader import load_collection_config  # noqa: E402
from real_vla.data.episode_reader import load_meta, load_trajectory  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)
    config = load_collection_config(args.config)
    meta = load_meta(args.episode_dir)
    traj = load_trajectory(args.episode_dir)
    camera_ts = {}
    camera_seq = {}
    for key, value in traj.items():
        if key.endswith("_timestamp_ns") and key.startswith("camera_"):
            name = key[len("camera_") : -len("_timestamp_ns")]
            camera_ts[name] = value
        if key.endswith("_capture_seq") and key.startswith("camera_"):
            name = key[len("camera_") : -len("_capture_seq")]
            camera_seq[name] = value
    video_ok = {
        name: (args.episode_dir / f"{name}.mkv").is_file()
        or (args.episode_dir / "head.mkv").is_file()
        for name in camera_ts
    }
    result = evaluate_episode(
        quality=config.quality,
        duration_s=float(meta.get("duration_s", 0.0)),
        arm_q=np.asarray(traj.get("robot_state_arm_q", np.zeros((0, 7)))),
        action_q=np.asarray(traj.get("action_arm_target_q", np.zeros((0, 7)))),
        state_ts=np.asarray(traj.get("robot_state_timestamp_ns", np.zeros((0,)))),
        action_ts=np.asarray(traj.get("action_timestamp_ns", np.zeros((0,)))),
        camera_ts=camera_ts,
        camera_seq=camera_seq,
        writer_drops={k: int(v) for k, v in dict(meta.get("writer_drops") or {}).items()},
        video_ok=video_ok,
    )
    print(result.label)
    for note in result.notes:
        print(f"- {note}")
    return 0 if result.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
