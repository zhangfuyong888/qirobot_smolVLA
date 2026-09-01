from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("pinocchio")

from hardware_teleop.config_loader import load_hardware_teleop_config
from hardware_teleop.ik import create_pure_hardware_ik_backend
from hardware_teleop.joint_mapping import bimanual_to_arm_q14
from s4_robot.control_mapping import bimanual_default_action
from teleoperation.mapping import TcpPose


ROOT = Path(__file__).resolve().parents[1]


def test_pure_hardware_pink_backend_fk_and_ik() -> None:
    config = load_hardware_teleop_config(ROOT / "hardware_teleop/config/quest_hardware.yaml")
    backend = create_pure_hardware_ik_backend(config)
    q = bimanual_to_arm_q14(bimanual_default_action())
    backend.set_posture_reference(q)
    left, right = backend.forward(q)
    target = TcpPose(left.position + np.array([0.01, 0.0, 0.0]), left.quat_wxyz)

    q_next = backend.compute(q, 1.0 / 30.0, target, right)
    moved_left, moved_right = backend.forward(q_next)

    assert backend.name == "pink"
    assert q_next.shape == (14,)
    assert moved_left.position[0] > left.position[0]
    assert moved_right.position == pytest.approx(right.position, abs=2.0e-5)
    assert backend.diagnostics()["runtime"] == "hardware_no_isaac"
