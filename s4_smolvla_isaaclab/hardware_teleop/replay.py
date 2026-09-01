"""JSONL recording and offline FK validation for Pink hardware shadow runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from hardware_teleop.config_loader import HardwareTeleopConfig
from hardware_teleop.ik import create_pure_hardware_ik_backend
from teleoperation.mapping import TcpPose


def _pose_dict(pose: TcpPose) -> dict[str, list[float]]:
    return {
        "position": np.asarray(pose.position, dtype=np.float64).tolist(),
        "quat_wxyz": np.asarray(pose.quat_wxyz, dtype=np.float64).tolist(),
    }


class PinkStateRecorder:
    """Write one self-contained control state per line for offline replay."""

    def __init__(self, path: Path, *, overwrite: bool = False) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("w" if overwrite else "x", encoding="utf-8")

    def write(
        self,
        *,
        monotonic_s: float,
        q14: np.ndarray,
        left_tcp: TcpPose,
        right_tcp: TcpPose,
        left_target: TcpPose,
        right_target: TcpPose,
        solved_q14: np.ndarray | None,
        commanded_q14: np.ndarray,
        fault: str | None,
    ) -> None:
        record: dict[str, Any] = {
            "schema": "s4_pink_hardware_shadow_v1",
            "monotonic_s": float(monotonic_s),
            "q14": np.asarray(q14, dtype=np.float64).tolist(),
            "left_tcp": _pose_dict(left_tcp),
            "right_tcp": _pose_dict(right_tcp),
            "left_target": _pose_dict(left_target),
            "right_target": _pose_dict(right_target),
            "solved_q14": (
                None
                if solved_q14 is None
                else np.asarray(solved_q14, dtype=np.float64).tolist()
            ),
            "commanded_q14": np.asarray(commanded_q14, dtype=np.float64).tolist(),
            "fault": fault,
        }
        self._stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()


def load_records(path: Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("schema") != "s4_pink_hardware_shadow_v1":
                raise ValueError(f"unsupported replay schema at line {line_number}")
            yield record


@dataclass(frozen=True)
class ReplayValidation:
    frames: int
    max_fk_position_error_m: float
    max_fk_rotation_component_error: float


def validate_fk_replay(
    config: HardwareTeleopConfig,
    path: Path,
) -> ReplayValidation:
    backend = create_pure_hardware_ik_backend(config)
    frames = 0
    max_position = 0.0
    max_rotation = 0.0
    for record in load_records(path):
        q14 = np.asarray(record["q14"], dtype=np.float64)
        left, right = backend.forward(q14)
        for side, pose in (("left", left), ("right", right)):
            expected = record[f"{side}_tcp"]
            max_position = max(
                max_position,
                float(
                    np.linalg.norm(
                        pose.position - np.asarray(expected["position"], dtype=np.float64)
                    )
                ),
            )
            actual_quat = np.asarray(pose.quat_wxyz, dtype=np.float64)
            expected_quat = np.asarray(expected["quat_wxyz"], dtype=np.float64)
            max_rotation = max(
                max_rotation,
                float(
                    min(
                        np.linalg.norm(actual_quat - expected_quat),
                        np.linalg.norm(actual_quat + expected_quat),
                    )
                ),
            )
        frames += 1
    if frames == 0:
        raise ValueError(f"Pink replay contains no frames: {path}")
    return ReplayValidation(frames, max_position, max_rotation)
