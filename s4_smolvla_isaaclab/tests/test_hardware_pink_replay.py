from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("pinocchio")

from hardware_teleop.config_loader import load_hardware_teleop_config
from hardware_teleop.ik import create_pure_hardware_ik_backend
from hardware_teleop.joint_mapping import bimanual_to_arm_q14
from hardware_teleop.replay import PinkStateRecorder, validate_fk_replay
from s4_robot.control_mapping import bimanual_default_action


ROOT = Path(__file__).resolve().parents[1]


def test_pink_shadow_recording_replays_fk_exactly(tmp_path: Path) -> None:
    config = load_hardware_teleop_config(ROOT / "hardware_teleop/config/quest_hardware.yaml")
    backend = create_pure_hardware_ik_backend(config)
    q14 = bimanual_to_arm_q14(bimanual_default_action())
    left, right = backend.forward(q14)
    path = tmp_path / "shadow.jsonl"
    recorder = PinkStateRecorder(path)
    try:
        recorder.write(
            monotonic_s=1.0,
            q14=q14,
            left_tcp=left,
            right_tcp=right,
            left_target=left,
            right_target=right,
            solved_q14=q14,
            commanded_q14=q14,
            fault=None,
        )
    finally:
        recorder.close()

    result = validate_fk_replay(config, path)
    assert result.frames == 1
    assert result.max_fk_position_error_m < 1.0e-12
    assert result.max_fk_rotation_component_error < 1.0e-12


def test_pink_recorder_refuses_to_overwrite_by_default(tmp_path: Path) -> None:
    path = tmp_path / "existing.jsonl"
    path.write_text("user data", encoding="utf-8")
    with pytest.raises(FileExistsError):
        PinkStateRecorder(path)
