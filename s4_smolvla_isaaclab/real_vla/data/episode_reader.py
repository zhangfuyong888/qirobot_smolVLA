"""Read a saved real_vla episode directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_meta(episode_dir: Path) -> dict[str, Any]:
    return json.loads((episode_dir / "meta.json").read_text(encoding="utf-8"))


def load_trajectory(episode_dir: Path) -> dict[str, np.ndarray]:
    h5_path = episode_dir / "trajectory.h5"
    npz_path = episode_dir / "trajectory.npz"
    if h5_path.is_file():
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError("trajectory.h5 present but h5py is not installed") from exc
        with h5py.File(h5_path, "r") as handle:
            return {key: np.asarray(handle[key][...]) for key in handle.keys()}
    if npz_path.is_file():
        with np.load(npz_path) as handle:
            return {key: handle[key] for key in handle.files}
    raise FileNotFoundError(f"no trajectory file in {episode_dir}")
