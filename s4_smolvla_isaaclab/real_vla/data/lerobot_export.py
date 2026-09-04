"""LeRobot export is intentionally unimplemented in the collection milestone."""

from __future__ import annotations

from pathlib import Path


def export_session_to_lerobot(session_dir: Path, output_dir: Path) -> None:
    del session_dir, output_dir
    raise NotImplementedError(
        "LeRobot conversion is a later milestone. Raw episodes stay in session/episodes/ "
        "as async head/wrist MKV + trajectory.h5 until the 20 Hz causal exporter lands."
    )
