from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from teleoperation.config import load_teleop_config
from teleoperation.mapping import (
    BimanualTeleopMapper,
    TcpPose,
    _rotation_vector,
    map_xr_rotation,
    map_xr_translation,
)
from teleoperation.protocol import ControllerFrame, ControllerSample


ROOT = Path(__file__).resolve().parents[1]


def sample(position=(0.0, 1.2, 0.0), trigger=0.0, squeeze=0.0, valid=True) -> ControllerSample:
    return ControllerSample(valid, position, (0.0, 0.0, 0.0, 1.0), trigger, squeeze)


def frame(left: ControllerSample, right: ControllerSample, received=1.0) -> ControllerFrame:
    return ControllerFrame("test", 1, 0.0, "local-floor", left, right, received)


def mapper() -> BimanualTeleopMapper:
    config = load_teleop_config(ROOT / "configs/teleoperation/meta_quest3.yaml")
    opened = np.zeros(6)
    closed = np.ones(6)
    return BimanualTeleopMapper(config, opened, closed, opened, closed)


def test_clutch_release_snaps_target_to_current_tcp() -> None:
    control = mapper()
    tcp = TcpPose(np.array([0.4, 0.2, 0.2]), np.array([1.0, 0.0, 0.0, 0.0]))
    engaged = frame(sample(squeeze=1.0), sample(squeeze=0.0))
    control.update(engaged, tcp, tcp, 0.01, 1.0)
    moved_tcp = TcpPose(np.array([0.45, 0.25, 0.22]), np.array([1.0, 0.0, 0.0, 0.0]))
    moved = frame(sample(position=(0.0, 1.2, -0.2), squeeze=1.0), sample(squeeze=0.0), received=1.01)
    control.update(moved, moved_tcp, moved_tcp, 0.01, 1.01)
    released = frame(sample(squeeze=0.0), sample(squeeze=0.0), received=1.02)
    result = control.update(released, moved_tcp, moved_tcp, 0.01, 1.02)
    assert result.left.clutch_falling
    assert not result.left.clutch
    assert result.left.target.position == pytest.approx(moved_tcp.position)
    assert result.left.target.quat_wxyz == pytest.approx(moved_tcp.quat_wxyz)


def test_quest_forward_axis_maps_controller_away_to_robot_back() -> None:
    """Shared basis (no invert): WebXR -Z (push away) -> robot -X."""
    config = load_teleop_config(ROOT / "configs/teleoperation/meta_quest3.yaml")
    basis = config.mapping.controller_to_base_rotation
    assert basis @ np.array([0.0, 0.0, -1.0]) == pytest.approx([-1.0, 0.0, 0.0])
    assert basis @ np.array([1.0, 0.0, 0.0]) == pytest.approx([0.0, 1.0, 0.0])
    assert basis @ np.array([0.0, 1.0, 0.0]) == pytest.approx([0.0, 0.0, 1.0])
    assert config.mapping.invert_translation is False
    assert config.mapping.invert_orientation is False
    assert config.mapping.translation_sign == (1.0, 1.0, 1.0)


def test_clutch_uses_quest_hardware_calibrated_horizontal_directions() -> None:
    control = mapper()
    left_tcp = TcpPose(np.array([0.4, 0.2, 0.2]), np.array([1.0, 0.0, 0.0, 0.0]))
    right_tcp = TcpPose(np.array([0.4, -0.2, 0.2]), np.array([1.0, 0.0, 0.0, 0.0]))
    engaged = frame(sample(squeeze=1.0), sample(squeeze=0.0))
    first = control.update(engaged, left_tcp, right_tcp, 1.0 / 120.0, 1.0)
    moved = frame(sample(position=(0.1, 1.3, 0.1), squeeze=1.0), sample(), received=1.01)
    second = control.update(moved, left_tcp, right_tcp, 1.0, 1.01)
    assert first.left.clutch_rising
    assert second.left.target.position[0] == pytest.approx(0.60)
    assert second.left.target.position[1] == pytest.approx(0.40)
    assert second.left.target.position[2] == pytest.approx(0.40)
    assert second.left.xr_delta == pytest.approx([0.1, 0.1, 0.1])
    assert second.left.base_delta == pytest.approx([0.2, 0.2, 0.2])


def test_each_arm_has_an_independent_clutch() -> None:
    control = mapper()
    pose = TcpPose(np.array([0.4, 0.0, 0.2]), np.array([1.0, 0.0, 0.0, 0.0]))
    result = control.update(frame(sample(squeeze=1.0), sample(squeeze=0.0)), pose, pose, 0.01, 1.0)
    assert result.left.clutch
    assert not result.right.clutch


def test_stale_frame_releases_clutches_and_freezes_target() -> None:
    control = mapper()
    pose = TcpPose(np.array([0.4, 0.0, 0.2]), np.array([1.0, 0.0, 0.0, 0.0]))
    active = frame(sample(squeeze=1.0), sample(squeeze=1.0), received=1.0)
    before = control.update(active, pose, pose, 0.01, 1.0)
    after = control.update(active, pose, pose, 0.01, 2.01)
    assert before.left.clutch and before.right.clutch
    assert after.stale
    assert not after.left.clutch and not after.right.clutch
    assert after.left.target.position == pytest.approx(before.left.target.position)
    still_held = control.update(
        frame(sample(squeeze=1.0), sample(squeeze=1.0), received=2.01), pose, pose, 0.01, 2.01
    )
    assert not still_held.left.clutch and not still_held.right.clutch
    released = control.update(
        frame(sample(squeeze=0.0), sample(squeeze=0.0), received=2.02), pose, pose, 0.01, 2.02
    )
    reengaged = control.update(
        frame(sample(squeeze=1.0), sample(squeeze=1.0), received=2.03), pose, pose, 0.01, 2.03
    )
    assert not released.left.clutch
    assert reengaged.left.clutch and reengaged.right.clutch


def test_clutch_translation_is_bounded_per_engagement() -> None:
    control = mapper()
    control.config = replace(
        control.config,
        mapping=replace(control.config.mapping, max_clutch_translation_m=0.05),
    )
    pose = TcpPose(np.array([0.4, 0.0, 0.2]), np.array([1.0, 0.0, 0.0, 0.0]))
    control.update(frame(sample(squeeze=1.0), sample()), pose, pose, 0.01, 1.0)
    moved = frame(
        sample(position=(1.0, 2.0, 1.0), squeeze=1.0),
        sample(),
        received=1.01,
    )
    result = control.update(moved, pose, pose, 1.0, 1.01)
    assert np.linalg.norm(result.left.target.position - pose.position) == pytest.approx(0.05)


def test_trigger_interpolates_six_hand_controls() -> None:
    control = mapper()
    pose = TcpPose(np.array([0.4, 0.0, 0.2]), np.array([1.0, 0.0, 0.0, 0.0]))
    result = control.update(frame(sample(trigger=1.0), sample()), pose, pose, 1.0, 1.0)
    assert result.left.hand6 == pytest.approx(np.ones(6), abs=1.0e-6)


def test_controller_filter_lags_a_position_step() -> None:
    pose = TcpPose(np.array([0.4, 0.0, 0.2]), np.array([1.0, 0.0, 0.0, 0.0]))
    moved = frame(sample(position=(0.0, 1.2, -0.1), squeeze=1.0), sample(), received=1.01)

    filtered = mapper()
    filtered.config = replace(
        filtered.config,
        mapping=replace(filtered.config.mapping, controller_filter_time_constant_s=0.20),
    )
    filtered.update(frame(sample(squeeze=1.0), sample()), pose, pose, 0.01, 1.0)
    filtered_result = filtered.update(moved, pose, pose, 0.01, 1.01)

    raw = mapper()
    raw.update(frame(sample(squeeze=1.0), sample()), pose, pose, 0.01, 1.0)
    raw_result = raw.update(moved, pose, pose, 0.01, 1.01)

    filtered_delta = float(np.linalg.norm(filtered_result.left.target.position - pose.position))
    raw_delta = float(np.linalg.norm(raw_result.left.target.position - pose.position))
    assert filtered_delta > 0.0
    assert filtered_delta < raw_delta


def test_configured_gamepad_button_can_engage_clutch() -> None:
    control = mapper()
    pose = TcpPose(np.array([0.4, 0.0, 0.2]), np.array([1.0, 0.0, 0.0, 0.0]))
    left = sample(squeeze=0.0)
    left = ControllerSample(
        left.valid,
        left.position,
        left.orientation_xyzw,
        left.trigger,
        left.squeeze,
        buttons=(0.0, 1.0),
    )
    result = control.update(frame(left, sample()), pose, pose, 0.01, 1.0)
    assert result.left.clutch


def test_invert_translation_maps_controller_away_to_robot_forward() -> None:
    config = load_teleop_config(ROOT / "configs/teleoperation/meta_quest3.yaml")
    basis = config.mapping.controller_to_base_rotation
    away = np.array([0.0, 0.0, -1.0])
    right = np.array([1.0, 0.0, 0.0])
    up = np.array([0.0, 1.0, 0.0])
    assert map_xr_translation(basis, away, 1.0, False) == pytest.approx([-1.0, 0.0, 0.0])
    assert map_xr_translation(basis, away, 1.0, True) == pytest.approx([1.0, 0.0, 0.0])
    assert map_xr_translation(basis, right, 1.0, True) == pytest.approx([0.0, -1.0, 0.0])
    assert map_xr_translation(basis, up, 1.0, True) == pytest.approx([0.0, 0.0, -1.0])


def test_xy_translation_sign_keeps_up_and_maps_away_to_forward() -> None:
    config = load_teleop_config(ROOT / "configs/teleoperation/meta_quest3.yaml")
    basis = config.mapping.controller_to_base_rotation
    sign = (-1.0, -1.0, 1.0)
    away = np.array([0.0, 0.0, -1.0])
    right = np.array([1.0, 0.0, 0.0])
    up = np.array([0.0, 1.0, 0.0])
    assert map_xr_translation(basis, away, 1.0, False, sign) == pytest.approx([1.0, 0.0, 0.0])
    assert map_xr_translation(basis, right, 1.0, False, sign) == pytest.approx([0.0, -1.0, 0.0])
    assert map_xr_translation(basis, up, 1.0, False, sign) == pytest.approx([0.0, 0.0, 1.0])


def test_xy_translation_sign_reverses_forearm_roll_sense() -> None:
    config = load_teleop_config(ROOT / "configs/teleoperation/meta_quest3.yaml")
    basis = config.mapping.controller_to_base_rotation
    angle = math.pi / 6.0
    cosine = math.cos(angle)
    sine = math.sin(angle)
    forearm = np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
    normal = _rotation_vector(map_xr_rotation(basis, forearm, False))
    signed = _rotation_vector(map_xr_rotation(basis, forearm, False, (-1.0, -1.0, 1.0)))
    axis = np.array([1.0, 0.0, 0.0])
    assert float(normal @ axis) > 0.4
    assert float(signed @ axis) < -0.4
    assert signed == pytest.approx(-normal)


def test_invert_orientation_reverses_rotation_sense_on_same_axis() -> None:
    config = load_teleop_config(ROOT / "configs/teleoperation/meta_quest3.yaml")
    basis = config.mapping.controller_to_base_rotation
    angle = math.pi / 6.0
    cosine = math.cos(angle)
    sine = math.sin(angle)
    # Forearm / pointing axis in WebXR local-floor is ±Z; wrist flexion is ±X.
    rotations_xr = {
        "forearm": np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]),
        "wrist": np.array([[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]]),
    }
    expected_base_axes = {
        "forearm": np.array([1.0, 0.0, 0.0]),
        "wrist": np.array([0.0, 1.0, 0.0]),
    }
    for name, delta_xr in rotations_xr.items():
        normal = _rotation_vector(map_xr_rotation(basis, delta_xr, False))
        inverted = _rotation_vector(map_xr_rotation(basis, delta_xr, True))
        axis = expected_base_axes[name]
        assert float(normal @ axis) > 0.4, name
        assert float(inverted @ axis) < -0.4, name
        assert inverted == pytest.approx(-normal)


def test_clutch_hardware_xy_sign_keeps_z() -> None:
    control = mapper()
    control.config = replace(
        control.config,
        mapping=replace(
            control.config.mapping,
            invert_translation=False,
            invert_orientation=False,
            translation_sign=(-1.0, -1.0, 1.0),
        ),
    )
    left_tcp = TcpPose(np.array([0.4, 0.2, 0.2]), np.array([1.0, 0.0, 0.0, 0.0]))
    right_tcp = TcpPose(np.array([0.4, -0.2, 0.2]), np.array([1.0, 0.0, 0.0, 0.0]))
    control.update(frame(sample(squeeze=1.0), sample(squeeze=0.0)), left_tcp, right_tcp, 1.0 / 120.0, 1.0)
    moved = frame(sample(position=(0.1, 1.3, 0.1), squeeze=1.0), sample(), received=1.01)
    second = control.update(moved, left_tcp, right_tcp, 1.0, 1.01)
    assert second.left.xr_delta == pytest.approx([0.1, 0.1, 0.1])
    assert second.left.base_delta == pytest.approx([-0.2, -0.2, 0.2])
    assert second.left.target.position == pytest.approx([0.20, 0.00, 0.40])
