from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("pinocchio")

from s4_robot.s4_robot_cfg import DEFAULT_POSE, LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS
from teleoperation.config import load_teleop_config
from teleoperation.controllers.pink_solver import PinkBimanualSolver
from teleoperation.mapping import (
    BimanualTeleopMapper,
    TcpPose,
    matrix_to_quat_wxyz,
    quat_wxyz_to_matrix,
)
from teleoperation.protocol import ControllerFrame, ControllerSample


ROOT = Path(__file__).resolve().parents[1]


def arm_home() -> np.ndarray:
    return np.asarray(
        [DEFAULT_POSE[name] for name in LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS],
        dtype=np.float64,
    )


def solver() -> PinkBimanualSolver:
    config = load_teleop_config(ROOT / "configs/teleoperation/meta_quest3.yaml")
    return PinkBimanualSolver(config.controller.pink)


def test_vendored_pink_solver_has_reduced_fourteen_dof_model() -> None:
    control = solver()
    assert control.model.nq == 14
    assert control.model.nv == 14
    left, right = control.forward(arm_home())
    assert np.isfinite(left.position).all()
    assert np.isfinite(right.position).all()


def test_pink_step_moves_requested_arm_and_respects_velocity_limit() -> None:
    control = solver()
    q = arm_home()
    control.set_posture_reference(q)
    left, right = control.forward(q)
    target = TcpPose(left.position + np.array([0.01, 0.0, 0.0]), left.quat_wxyz)
    dt = 1.0 / 120.0
    q_next = control.compute(q, dt, target, right)
    moved_left, moved_right = control.forward(q_next)

    assert q_next.shape == (14,)
    assert np.max(np.abs(q_next - q)) <= control.config.max_joint_velocity_rad_s * dt + 1.0e-6
    assert moved_left.position[0] > left.position[0]
    assert moved_right.position == pytest.approx(right.position, abs=1.0e-6)


def test_pink_unreachable_target_stays_finite_and_within_joint_limits() -> None:
    control = solver()
    q = arm_home()
    control.set_posture_reference(q)
    left, right = control.forward(q)
    unreachable = TcpPose(np.array([4.0, 4.0, 4.0]), left.quat_wxyz)
    for _ in range(20):
        q = control.compute(q, 1.0 / 120.0, unreachable, right)
    assert np.isfinite(q).all()
    assert np.all(q >= control.model.lowerPositionLimit - 1.0e-6)
    assert np.all(q <= control.model.upperPositionLimit + 1.0e-6)


def test_pink_recovers_small_measured_joint_limit_overshoot() -> None:
    control = solver()
    q = arm_home()
    q[5] = float(control.model.lowerPositionLimit[5]) - 8.0e-5
    left, right = control.forward(q)

    q_next = control.compute(q, 1.0 / 40.0, left, right)

    assert q_next[5] >= float(control.model.lowerPositionLimit[5])
    assert control.diagnostics()["last_input_limit_correction_rad"] == pytest.approx(8.0e-5)


def test_pink_rejects_large_measured_joint_limit_overshoot() -> None:
    control = solver()
    q = arm_home()
    q[5] = float(control.model.lowerPositionLimit[5]) - 0.02
    left, right = control.forward(q)

    with pytest.raises(RuntimeError, match="left_wrist_pitch_joint"):
        control.compute(q, 1.0 / 40.0, left, right)


def test_pink_elbow_barriers_resist_simultaneous_inward_targets() -> None:
    control = solver()
    q = arm_home()
    control.set_posture_reference(q)
    left, right = control.forward(q)
    left_inward = TcpPose(
        left.position + np.array([0.0, -0.35, 0.0]), left.quat_wxyz
    )
    right_inward = TcpPose(
        right.position + np.array([0.0, 0.35, 0.0]), right.quat_wxyz
    )

    for _ in range(150):
        q = control.compute(q, 1.0 / 60.0, left_inward, right_inward)
    control.forward(q)
    left_elbow_y = float(
        control.configuration.get_transform_frame_to_world("left_elbow_link").translation[1]
    )
    right_elbow_y = float(
        control.configuration.get_transform_frame_to_world("right_elbow_link").translation[1]
    )
    minimum = control.config.elbow_avoidance.min_lateral_distance_base_m

    assert left_elbow_y >= minimum
    assert right_elbow_y <= -minimum
    assert left_elbow_y == pytest.approx(-right_elbow_y, abs=1.0e-4)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_pink_cartesian_translation_direction(side: str, axis: int, sign: float) -> None:
    control = solver()
    q = arm_home()
    control.set_posture_reference(q)
    initial_left, initial_right = control.forward(q)
    initial = initial_left if side == "left" else initial_right
    offset = np.zeros(3)
    offset[axis] = sign * 0.01
    moved_target = TcpPose(initial.position + offset, initial.quat_wxyz)
    for _ in range(12):
        left_target = moved_target if side == "left" else initial_left
        right_target = moved_target if side == "right" else initial_right
        q = control.compute(q, 1.0 / 60.0, left_target, right_target)
    final_left, final_right = control.forward(q)
    final = final_left if side == "left" else final_right
    inactive_final = final_right if side == "left" else final_left
    inactive_initial = initial_right if side == "left" else initial_left
    displacement = final.position - initial.position

    assert sign * displacement[axis] > 0.005
    assert np.linalg.norm(inactive_final.position - inactive_initial.position) < 1.0e-6


def _axis_angle(axis: int, angle: float) -> np.ndarray:
    vector = np.zeros(3)
    vector[axis] = 1.0
    skew = np.array(
        [
            [0.0, -vector[2], vector[1]],
            [vector[2], 0.0, -vector[0]],
            [-vector[1], vector[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_pink_cartesian_rotation_direction(side: str, axis: int, sign: float) -> None:
    control = solver()
    q = arm_home()
    control.set_posture_reference(q)
    initial_left, initial_right = control.forward(q)
    initial = initial_left if side == "left" else initial_right
    initial_rotation = quat_wxyz_to_matrix(initial.quat_wxyz)
    requested_rotation = _axis_angle(axis, sign * 0.08) @ initial_rotation
    moved_target = TcpPose(initial.position, matrix_to_quat_wxyz(requested_rotation))
    for _ in range(20):
        left_target = moved_target if side == "left" else initial_left
        right_target = moved_target if side == "right" else initial_right
        q = control.compute(q, 1.0 / 60.0, left_target, right_target)
    final_left, final_right = control.forward(q)
    final = final_left if side == "left" else final_right
    relative = quat_wxyz_to_matrix(final.quat_wxyz) @ initial_rotation.T
    signed_sine = np.array(
        [
            relative[2, 1] - relative[1, 2],
            relative[0, 2] - relative[2, 0],
            relative[1, 0] - relative[0, 1],
        ]
    ) * 0.5

    assert sign * signed_sine[axis] > 0.02


@pytest.mark.parametrize("xr_axis", [0, 1, 2])
def test_quest_translation_mapping_and_pink_direction_agree(xr_axis: int) -> None:
    config = load_teleop_config(ROOT / "configs/teleoperation/meta_quest3.yaml")
    control = PinkBimanualSolver(config.controller.pink)
    q = arm_home()
    control.set_posture_reference(q)
    initial_left, initial_right = control.forward(q)
    opened = np.zeros(6)
    closed = np.ones(6)
    mapper = BimanualTeleopMapper(config, opened, closed, opened, closed)

    reference_position = np.array([0.0, 1.2, 0.0])

    def sample(position: np.ndarray, squeeze: float) -> ControllerSample:
        return ControllerSample(
            valid=True,
            position=tuple(float(value) for value in position),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            trigger=0.0,
            squeeze=squeeze,
        )

    engaged = ControllerFrame(
        "pink-direction",
        1,
        0.0,
        "local-floor",
        sample(reference_position, 1.0),
        sample(reference_position, 0.0),
        1.0,
    )
    mapper.update(engaged, initial_left, initial_right, 1.0 / 60.0, 1.0)
    moved_position = reference_position.copy()
    moved_position[xr_axis] += 0.01
    moved = ControllerFrame(
        "pink-direction",
        2,
        0.0,
        "local-floor",
        sample(moved_position, 1.0),
        sample(reference_position, 0.0),
        1.01,
    )
    mapped = mapper.update(moved, initial_left, initial_right, 1.0, 1.01)
    expected_base_delta = (
        config.mapping.controller_to_base_rotation
        @ np.eye(3, dtype=np.float64)[:, xr_axis]
    )
    for _ in range(15):
        q = control.compute(q, 1.0 / 60.0, mapped.left.target, initial_right)
    final_left, _ = control.forward(q)
    actual_delta = final_left.position - initial_left.position

    assert float(actual_delta @ expected_base_delta) > 0.005
