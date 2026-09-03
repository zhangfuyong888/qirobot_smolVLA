from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("pinocchio")

from hardware_teleop.config_loader import load_hardware_teleop_config
from hardware_teleop.ik import create_pure_hardware_ik_backend
from hardware_teleop.joint_mapping import bimanual_to_arm_q14
from hardware_teleop.pink_main import _max_active_proximal_tracking_error
from s4_robot.control_mapping import bimanual_default_action
from teleoperation.mapping import TcpPose


ROOT = Path(__file__).resolve().parents[1]


def test_proximal_tracking_error_only_checks_active_shoulder_and_elbow() -> None:
    actual = np.zeros(14, dtype=np.float64)
    command = np.zeros(14, dtype=np.float64)
    command[3] = 0.19
    command[10] = 0.25
    assert _max_active_proximal_tracking_error(
        actual, command, left_active=True, right_active=False
    ) == pytest.approx(0.19)
    assert _max_active_proximal_tracking_error(
        actual, command, left_active=False, right_active=True
    ) == pytest.approx(0.25)
    assert _max_active_proximal_tracking_error(
        actual, command, left_active=False, right_active=False
    ) == 0.0


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
    assert backend._controller.config.max_joint_velocity_rad_s == pytest.approx(0.90)


def test_hardware_joint_limit_task_fades_in_toward_soft_boundary() -> None:
    config = load_hardware_teleop_config(ROOT / "hardware_teleop/config/quest_hardware.yaml")
    backend = create_pure_hardware_ik_backend(config)
    controller = backend._controller
    lower = controller._arm_lower_limits
    upper = controller._arm_upper_limits
    center = 0.5 * (lower + upper)
    half_range = 0.5 * (upper - lower)
    q = center.copy()
    q[0] = center[0] + 0.9 * half_range[0]
    controller.configuration.update(controller._bounded_full_q(q))

    backend._update_joint_limit_task()

    target = backend._limit_task.target_q[controller._arm_q_indices]
    costs = np.asarray(backend._limit_task.cost)[controller._arm_v_indices]
    assert target[0] == pytest.approx(center[0] + 0.8 * half_range[0])
    assert costs[0] > 0.0
    np.testing.assert_allclose(costs[1:], 0.0)
    assert backend.diagnostics()["joint_limit_active_joints"] == 1.0


def test_hardware_joint_limit_task_moves_inward_without_large_tcp_drift() -> None:
    config = load_hardware_teleop_config(ROOT / "hardware_teleop/config/quest_hardware.yaml")
    backend = create_pure_hardware_ik_backend(config)
    controller = backend._controller
    lower = controller._arm_lower_limits
    upper = controller._arm_upper_limits
    center = 0.5 * (lower + upper)
    half_range = 0.5 * (upper - lower)
    q = center.copy()
    q[0] = center[0] + 0.95 * half_range[0]
    backend.set_posture_reference(q)
    left, right = backend.forward(q)
    initial_normalized = abs((q[0] - center[0]) / half_range[0])

    for _ in range(60):
        q = backend.compute(q, 1.0 / 30.0, left, right)

    final_left, final_right = backend.forward(q)
    final_normalized = abs((q[0] - center[0]) / half_range[0])
    assert final_normalized < initial_normalized
    assert np.linalg.norm(final_left.position - left.position) < 0.005
    assert np.linalg.norm(final_right.position - right.position) < 0.005


def test_hardware_elbows_cannot_cross_straight_arm_boundary() -> None:
    config = load_hardware_teleop_config(ROOT / "hardware_teleop/config/quest_hardware.yaml")
    backend = create_pure_hardware_ik_backend(config)
    controller = backend._controller
    for name in ("left_elbow_joint", "right_elbow_joint"):
        q_index = controller._q_index_by_name[name]
        v_index = controller._v_index_by_name[name]
        arm_index = int(np.flatnonzero(controller._arm_q_indices == q_index)[0])
        assert controller.model.upperPositionLimit[q_index] == pytest.approx(-0.08)
        assert controller.configuration_limit.upper[q_index] == pytest.approx(-0.08)
        assert controller._arm_upper_limits[arm_index] == pytest.approx(-0.08)
        assert controller.velocity_limit.maximum[v_index] == pytest.approx(0.65)


def test_hardware_shoulder_flip_limit_tracks_clutch_reference() -> None:
    config = load_hardware_teleop_config(ROOT / "hardware_teleop/config/quest_hardware.yaml")
    backend = create_pure_hardware_ik_backend(config)
    controller = backend._controller
    q = 0.5 * (controller._arm_lower_limits + controller._arm_upper_limits)
    backend.set_posture_reference(q)
    selected = [
        controller._q_index_by_name[name]
        for name in (
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
        )
    ]
    np.testing.assert_allclose(
        backend._shoulder_reference_limit._reference,
        controller._posture_reference[selected],
    )
    for name in (
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
    ):
        assert controller.velocity_limit.maximum[
            controller._v_index_by_name[name]
        ] == pytest.approx(0.55)


def test_hardware_shoulder_flip_limit_drives_overshoot_back_inward() -> None:
    config = load_hardware_teleop_config(ROOT / "hardware_teleop/config/quest_hardware.yaml")
    backend = create_pure_hardware_ik_backend(config)
    controller = backend._controller
    reference = 0.5 * (
        controller._arm_lower_limits + controller._arm_upper_limits
    )
    backend.set_posture_reference(reference)
    shoulder_arm_index = 1
    q = reference.copy()
    q[shoulder_arm_index] += 0.90
    left, right = backend.forward(q)

    q_next = backend.compute(q, 1.0 / 30.0, left, right)

    assert q_next[shoulder_arm_index] < q[shoulder_arm_index]


def test_hardware_elbow_limit_blocks_reverse_branch_target() -> None:
    config = load_hardware_teleop_config(ROOT / "hardware_teleop/config/quest_hardware.yaml")
    backend = create_pure_hardware_ik_backend(config)
    controller = backend._controller
    q = 0.5 * (controller._arm_lower_limits + controller._arm_upper_limits)
    backend.set_posture_reference(q)
    left, right = backend.forward(q)
    reverse_q = q.copy()
    reverse_q[3] = 0.30
    reverse_left, _ = backend.forward(reverse_q)

    for _ in range(120):
        q = backend.compute(q, 1.0 / 30.0, reverse_left, right)

    assert q[3] <= config.ik.elbow_max_angle_rad + 1.0e-6
