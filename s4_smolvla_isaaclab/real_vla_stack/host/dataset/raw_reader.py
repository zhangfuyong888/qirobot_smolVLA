from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from real_vla.data.episode_reader import load_trajectory

from ...common.errors import DataValidationError


@dataclass(frozen=True)
class RawEpisode:
    path: Path
    session: str
    meta: dict[str, Any]
    trajectory: dict[str, np.ndarray]

    def camera_video(self, source: str) -> Path:
        name = "head.mkv" if source == "head" else f"{source}.mkv"
        return self.path / name


def discover_raw_episodes(raw_root: Path, *, require_saved: bool = True) -> list[RawEpisode]:
    root = Path(raw_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"raw dataset root does not exist: {root}")
    episodes: list[RawEpisode] = []
    rejected: list[str] = []
    for path in sorted(root.glob("session_*/episodes/episode_*")):
        meta_path = path / "meta.json"
        if not meta_path.is_file():
            rejected.append(f"{path}: missing meta.json")
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if require_saved and meta.get("result") != "saved":
            continue
        if require_saved and not bool(meta.get("quality_valid", False)):
            rejected.append(f"{path}: saved episode is not quality_valid")
            continue
        episodes.append(RawEpisode(path, path.parents[1].name, meta, load_trajectory(path)))
    if rejected:
        raise DataValidationError("invalid saved raw episodes:\n" + "\n".join(rejected))
    if not episodes:
        raise DataValidationError(f"no saved quality-valid episodes under {root}")
    return episodes
