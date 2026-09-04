#!/usr/bin/env python
"""Print one saved episode's meta and trajectory counts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from real_vla.data.episode_reader import load_meta, load_trajectory  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_dir", type=Path)
    args = parser.parse_args(argv)
    meta = load_meta(args.episode_dir)
    print(json.dumps(meta, indent=2))
    traj = load_trajectory(args.episode_dir)
    for key, value in traj.items():
        print(f"{key}: shape={value.shape} dtype={value.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
