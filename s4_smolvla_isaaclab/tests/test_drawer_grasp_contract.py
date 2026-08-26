from __future__ import annotations

import numpy as np

from s4_pipeline.drawer_distractors import (
    GRASP_CAN_SCALE,
    GRASP_CAN_SCALE_Y_ENV,
    LEGACY_GRASP_CAN_SCALE,
    scene_grasp_can_scale,
)
from tasks.drawer_insert_close_controller import DrawerInsertCloseController, load_scripted_config


class _FakeTcpController:
    isaac_order_joint_ids = list(range(14))

    def set_posture_reference(self, _curr_joint_pos):
        return None

    def compute(self, _curr_joint_pos, _dt, _left_goal, _right_goal):
        return np.zeros(14, dtype=np.float32)


def _anchors() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return {
        "can": (np.asarray([0.54, -0.13, 0.18], dtype=np.float32), quat),
        "drawer_handle_initial": (np.asarray([0.44, 0.38, 0.05], dtype=np.float32), quat),
        "drawer_handle_open": (np.asarray([0.26, 0.38, 0.05], dtype=np.float32), quat),
        "drawer_handle_closed": (np.asarray([0.44, 0.38, 0.05], dtype=np.float32), quat),
    }


def test_original_can_scale_and_rollout_override(monkeypatch):
    assert GRASP_CAN_SCALE == (1.0, 0.90, 1.0)
    assert LEGACY_GRASP_CAN_SCALE == (1.0, 0.90, 1.0)
    assert scene_grasp_can_scale() == GRASP_CAN_SCALE
    monkeypatch.setenv(GRASP_CAN_SCALE_Y_ENV, "0.90")
    assert scene_grasp_can_scale() == LEGACY_GRASP_CAN_SCALE


def test_grasp_config_is_stationary_and_deterministic():
    cfg = load_scripted_config()
    assert cfg["randomization"]["can_xy"]["enabled"] is True
    assert cfg["randomization"]["distractor_cans"]["enabled"] is False
    assert cfg["randomization"]["can_xy"]["x_range"] == [-0.025, -0.005]
    assert cfg["randomization"]["can_xy"]["y_range"] == [-0.17, 0.01]
    x_width = np.ptp(cfg["randomization"]["can_xy"]["x_range"])
    y_width = np.ptp(cfg["randomization"]["can_xy"]["y_range"])
    assert np.isclose(x_width * y_width, 0.0036)  # 36 cm^2 reserved re-enable area
    assert cfg["randomization"]["can_xy"]["max_grasp_retries_same_position"] == 3
    cabinet_support_near_x = 0.4752
    tomato_can_radius_x = 0.03383
    minimum_center_x = 0.54 + cfg["randomization"]["can_xy"]["x_range"][0]
    assert minimum_center_x - tomato_can_radius_x >= cabinet_support_near_x + 0.005
    assert cfg["randomization"]["right_can_lift"]["enabled"] is False
    assert "drawer_initial_open" not in cfg["randomization"]
    assert cfg["drawer"]["initial_open_m"] == 0.0
    assert cfg["drawer"]["target_open_m"] == 0.18
    assert cfg["hands"]["left_close"] == [1.0, 0.22, 0.75, 0.75, 0.75, 0.75]
    assert cfg["targets"]["left_handle_transition_1"]["rpy"] == [-1.5, -0.05, 1.5]
    assert cfg["targets"]["left_handle_transition_2"]["rpy"] == [-1.5, -0.15, 1.5]
    assert cfg["targets"]["left_handle_transition_3"]["offset"] == [-0.0975, -0.0185, 0.0368]
    assert cfg["targets"]["left_handle_transition_3"]["rpy"] == [-1.5, -0.20, 1.5]
    assert cfg["targets"]["left_handle_preload"]["offset"] == [-0.1055, -0.0185, 0.0368]
    assert cfg["targets"]["left_drawer_open"]["offset"] == [-0.0975, -0.0185, 0.0368]
    assert cfg["targets"]["left_drawer_open"]["rpy"] == [-1.5, -0.20, 1.5]
    assert cfg["hands"]["right_open"] == [0.95, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert cfg["hands"]["right_close"] == [1.0, 0.42, 0.85, 0.85, 0.85, 0.85]
    assert cfg["targets"]["right_can_grasp"]["offset"] == [-0.050, -0.038, 0.030]
    assert cfg["targets"]["right_can_pregrasp"]["offset"] == [-0.12, -0.12, 0.10]
    assert cfg["targets"]["right_can_pregrasp"]["rpy"] == [0.0, -1.4, 0.0]
    assert cfg["targets"]["right_can_grasp"]["rpy"] == [0.0, -1.4, 0.0]

    phases = {phase["name"]: phase for phase in cfg["phases"]}
    assert phases["left_grasp_handle"]["tolerance"] == 0.020
    assert phases["left_grasp_handle"]["orientation_tolerance"] == 0.30
    assert phases["left_grasp_handle"]["hold_seconds"] == 0.5
    assert phases["left_hold_handle_pregrasp"]["hold_seconds"] == 0.5
    assert phases["left_hold_handle_pregrasp"]["hold_current_left_pose"] is True
    assert phases["left_close_hand"]["tolerance"] == 0.020
    assert phases["left_close_hand"]["orientation_tolerance"] == 0.30
    assert phases["left_preload_handle"]["drawer_open_min"] == 0.003
    assert phases["left_preload_handle"]["hold_seconds"] == 0.5
    assert phases["left_preload_handle"]["tolerance"] == 0.020
    assert phases["pull_drawer"]["tolerance"] == 0.025
    assert phases["pull_drawer"]["orientation_tolerance"] == 0.35
    assert phases["left_hold_drawer_open"]["hold_seconds"] == 0.5
    assert phases["left_hold_drawer_open"]["drawer_open_min"] == 0.08
    assert phases["left_hold_drawer_open"]["tolerance"] == 0.025
    assert phases["initial_open_hands"]["hand_actual_tolerance"] == 0.10
    assert phases["right_pregrasp_can"]["tolerance"] == 0.010
    assert phases["right_hold_can_pregrasp"]["hold_seconds"] == 0.5
    assert phases["right_hold_can_pregrasp"]["hold_current_right_pose"] is True
    assert phases["right_grasp_can"]["tolerance"] == 0.028
    assert phases["right_pregrasp_can"]["task_object_max_displacement_from_start_m"] == 0.020
    assert phases["right_grasp_can"]["task_object_max_displacement_from_start_m"] == 0.020
    assert phases["right_grasp_can"]["keep_ik_posture_reference"] is True
    assert phases["right_lift_can"]["keep_ik_posture_reference"] is True
    for name in (
        "right_pregrasp_can",
        "right_grasp_can",
        "right_settle_before_close",
        "right_close_hand",
        "right_hold_grasp",
    ):
        assert phases[name]["require_left_tcp_reached"] is False
        assert phases[name]["left"] == {"target": "left_drawer_open"}
    for name in (
        "right_pregrasp_can",
        "right_settle_before_close",
        "right_close_hand",
        "right_hold_grasp",
    ):
        assert phases[name]["tolerance"] == 0.010
    assert phases["left_close_drawer"]["drawer_open_max"] == 0.020
    assert phases["right_open_hand"]["hold_seconds"] == 1.5
    assert phases["right_open_hand"]["hold_current_right_pose"] is True
    assert phases["right_open_hand"]["require_right_tcp_reached"] is False
    assert phases["right_open_hand"]["require_right_hand_actual_reached"] is True
    assert phases["right_open_hand"]["task_object_world_bounds"]["z"] == [1.00, 1.04]
    assert phases["right_open_hand"]["task_object_max_speed_m_s"] == 0.05
    assert phases["right_lift_can"]["task_object_world_bounds"]["z"] == [1.20, 1.35]
    assert phases["right_lift_clear_drawer"]["right_offset_from_current"] == [0.0, 0.0, 0.10]
    assert phases["right_retreat_clear_drawer"]["right_offset_from_current"] == [-0.10, -0.18, 0.02]
    assert phases["right_home_after_retreat"]["right_arm_home"] is True
    assert phases["right_home_after_retreat"]["drawer_open_min"] == 0.08
    assert "right_arm_home" not in phases["left_close_drawer"]
    assert phases["left_open_hand"]["hold_current_left_pose"] is True
    assert phases["left_open_hand"]["hold_seconds"] == 1.0
    assert phases["left_open_hand"]["require_left_hand_actual_reached"] is False
    clear = phases["left_clear_handle_after_release"]
    assert clear["left_offset_from_current"] == [-0.040, 0.000, 0.050]
    assert clear["require_left_hand_actual_reached"] is True
    assert clear["hand_actual_tolerance"] == 0.05
    assert clear["drawer_open_max"] == 0.040
    assert clear["target_alpha"] == 0.12
    assert clear["max_joint_step"] == 0.015
    assert clear["hold_seconds"] == 0.0
    transition = phases["left_joint_transition_after_release"]
    assert transition["left_arm_joint_target"] == [
        0.430,
        0.677,
        0.100,
        -1.782,
        -0.029,
        -0.098,
        -0.402,
    ]
    assert transition["right_arm_home"] is True
    assert transition["require_left_hand_actual_reached"] is True
    assert transition["drawer_open_max"] == 0.040
    assert transition["arm_joint_tolerance"] == 0.040
    assert transition["target_alpha"] == 0.12
    assert transition["max_joint_step"] == 0.015
    assert phases["left_home"]["require_left_hand_actual_reached"] is True
    assert phases["left_home"]["drawer_open_max"] == 0.040
    for name in ("right_settle_before_close", "right_close_hand", "right_hold_grasp"):
        assert phases[name]["hold_current_right_pose"] is True
    assert phases["right_settle_before_close"]["right_hand"] == "open"
    assert phases["right_settle_before_close"]["hold_seconds"] == 1.0
    assert phases["right_close_hand"]["right_hand"] == "close"
    assert phases["right_hold_grasp"]["right_hand"] == "close"


def test_left_release_wait_does_not_deadlock_on_handle_contact():
    controller = DrawerInsertCloseController(
        _FakeTcpController(),
        initial_action=np.zeros(26, dtype=np.float32),
        anchors=_anchors(),
    )
    phase_index = next(
        index for index, phase in enumerate(controller.phases) if phase.name == "left_open_hand"
    )
    controller.phase_index = phase_index
    pose = (
        controller.current_phase.left.pos.copy(),
        controller.current_phase.left.quat_wxyz.copy(),
    )
    controller._prepare_current_phase(pose, None, drawer_open_m=0.0)
    phase = controller.current_phase
    controller._dt = 1.0 / 120.0
    controller.phase_steps = int(np.ceil(phase.hold_seconds * 120.0))

    commanded = np.zeros(26, dtype=np.float32)
    commanded[7:13] = phase.left_hand
    actual = commanded.copy()
    # Simulate four fingers remaining partly flexed while they still touch the
    # handle.  The release wait must advance so the next phase can disengage.
    actual[9:13] = 0.55

    assert controller._advance_if_ready(
        pose,
        None,
        drawer_open_m=0.0,
        curr_joint_pos=np.zeros(14, dtype=np.float32),
        commanded_action=commanded,
        actual_action=actual,
    ) is True
    assert controller.current_phase.name == "left_clear_handle_after_release"
    assert controller.current_phase.require_left_hand_actual_reached is True


def test_left_clear_handle_moves_backward_up_before_joint_transition():
    controller = DrawerInsertCloseController(
        _FakeTcpController(),
        initial_action=np.zeros(26, dtype=np.float32),
        anchors=_anchors(),
    )
    phase_index = next(
        index
        for index, phase in enumerate(controller.phases)
        if phase.name == "left_clear_handle_after_release"
    )
    controller.phase_index = phase_index
    entry_pose = (
        np.asarray([0.355, 0.358, 0.104], dtype=np.float32),
        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    controller._prepare_current_phase(entry_pose, None, drawer_open_m=0.0)
    phase = controller.current_phase
    assert np.allclose(phase.left.pos, entry_pose[0] + [-0.040, 0.000, 0.050])
    assert np.allclose(phase.left.quat_wxyz, entry_pose[1])

    controller._dt = 1.0 / 120.0
    controller.phase_steps = phase.min_steps
    commanded = np.zeros(26, dtype=np.float32)
    commanded[7:13] = phase.left_hand
    blocked = commanded.copy()
    blocked[9:13] = 0.40
    target_pose = (phase.left.pos.copy(), phase.left.quat_wxyz.copy())

    assert controller._advance_if_ready(
        target_pose,
        None,
        drawer_open_m=0.0,
        curr_joint_pos=np.zeros(14, dtype=np.float32),
        commanded_action=commanded,
        actual_action=blocked,
    ) is False
    assert controller._advance_if_ready(
        target_pose,
        None,
        drawer_open_m=0.0,
        curr_joint_pos=np.zeros(14, dtype=np.float32),
        commanded_action=commanded,
        actual_action=commanded,
    ) is True
    assert controller.current_phase.name == "left_joint_transition_after_release"


def test_left_release_transition_commands_and_gates_on_joint_target():
    controller = DrawerInsertCloseController(
        _FakeTcpController(),
        initial_action=np.zeros(26, dtype=np.float32),
        anchors=_anchors(),
    )
    phase_index = next(
        index
        for index, phase in enumerate(controller.phases)
        if phase.name == "left_joint_transition_after_release"
    )
    controller.phase_index = phase_index
    phase = controller.current_phase
    controller.phase_steps = 0

    current = np.zeros(14, dtype=np.float32)
    current[7:14] = controller.home_targets["right"]
    actual_action = np.zeros(26, dtype=np.float32)
    actual_action[7:13] = phase.left_hand
    action, name, _, done = controller.step(
        current,
        1.0 / 120.0,
        None,
        None,
        drawer_open_m=0.0,
        commanded_action=actual_action,
        actual_action=actual_action,
    )
    assert name == "left_joint_transition_after_release"
    assert done is False
    assert np.allclose(action[0:7], phase.left_arm_joint_target)
    assert np.allclose(action[13:20], controller.home_targets["right"])

    controller.phase_steps = phase.min_steps
    almost_reached = current.copy()
    almost_reached[:7] = phase.left_arm_joint_target
    almost_reached[3] += phase.arm_joint_tolerance + 0.01
    assert controller._advance_if_ready(
        None,
        None,
        drawer_open_m=0.0,
        curr_joint_pos=almost_reached,
        commanded_action=actual_action,
        actual_action=actual_action,
    ) is False

    reached = current.copy()
    reached[:7] = phase.left_arm_joint_target
    assert controller._advance_if_ready(
        None,
        None,
        drawer_open_m=0.0,
        curr_joint_pos=reached,
        commanded_action=actual_action,
        actual_action=actual_action,
    ) is True
    assert controller.current_phase.name == "left_home"


def test_left_preload_requires_measured_drawer_response_before_full_pull():
    controller = DrawerInsertCloseController(
        _FakeTcpController(),
        initial_action=np.zeros(26, dtype=np.float32),
        anchors=_anchors(),
    )
    phase_index = next(
        index for index, phase in enumerate(controller.phases) if phase.name == "left_preload_handle"
    )
    controller.phase_index = phase_index
    phase = controller.current_phase
    pose = (phase.left.pos.copy(), phase.left.quat_wxyz.copy())
    commanded = np.zeros(26, dtype=np.float32)
    commanded[7:13] = phase.left_hand
    controller.phase_steps = max(phase.min_steps, int(np.ceil(phase.hold_seconds * 120.0)))
    controller._dt = 1.0 / 120.0

    assert controller._advance_if_ready(
        pose,
        None,
        drawer_open_m=0.002,
        curr_joint_pos=np.zeros(14, dtype=np.float32),
        commanded_action=commanded,
        actual_action=commanded,
    ) is False
    assert controller.current_phase.name == "left_preload_handle"

    assert controller._advance_if_ready(
        pose,
        None,
        drawer_open_m=0.004,
        curr_joint_pos=np.zeros(14, dtype=np.float32),
        commanded_action=commanded,
        actual_action=commanded,
    ) is True
    assert controller.current_phase.name == "pull_drawer"


def test_release_waits_for_actual_open_hand_and_can_inside_drawer():
    controller = DrawerInsertCloseController(
        _FakeTcpController(),
        initial_action=np.zeros(26, dtype=np.float32),
        anchors=_anchors(),
    )
    phase_index = next(index for index, phase in enumerate(controller.phases) if phase.name == "right_open_hand")
    controller.phase_index = phase_index
    pose = (np.asarray([0.42, 0.05, 0.22], dtype=np.float32), np.asarray([1.0, 0.0, 0.0, 0.0]))
    controller._prepare_current_phase(pose, pose, drawer_open_m=0.18)
    controller.phase_steps = 200
    controller._dt = 1.0 / 120.0
    commanded = np.zeros(26, dtype=np.float32)
    commanded[20:26] = controller.current_phase.right_hand
    actual = commanded.copy()
    actual[20] += 0.10
    assert controller._advance_if_ready(
        pose,
        pose,
        drawer_open_m=0.18,
        commanded_action=commanded,
        actual_action=actual,
        task_object_position_world=np.asarray([0.40, 0.14, 1.02]),
        task_object_linear_velocity_world=np.zeros(3),
    ) is False
    actual[20:26] = controller.current_phase.right_hand
    assert controller._advance_if_ready(
        pose,
        pose,
        drawer_open_m=0.18,
        commanded_action=commanded,
        actual_action=actual,
        task_object_position_world=np.asarray([0.40, 0.14, 0.03]),
        task_object_linear_velocity_world=np.zeros(3),
    ) is False
    assert controller._advance_if_ready(
        pose,
        pose,
        drawer_open_m=0.18,
        commanded_action=commanded,
        actual_action=actual,
        task_object_position_world=np.asarray([0.40, 0.14, 1.02]),
        task_object_linear_velocity_world=np.asarray([0.10, 0.0, 0.0]),
    ) is False
    assert controller._advance_if_ready(
        pose,
        pose,
        drawer_open_m=0.18,
        commanded_action=commanded,
        actual_action=actual,
        task_object_position_world=np.asarray([0.40, 0.14, 1.02]),
        task_object_linear_velocity_world=np.asarray([0.01, 0.0, 0.0]),
    ) is True
    assert controller.current_phase.name == "right_lift_clear_drawer"


def test_hold_current_right_pose_freezes_phase_entry_tcp():
    controller = DrawerInsertCloseController(
        _FakeTcpController(),
        initial_action=np.zeros(26, dtype=np.float32),
        anchors=_anchors(),
    )
    phase_index = next(
        index for index, phase in enumerate(controller.phases) if phase.name == "right_close_hand"
    )
    controller.phase_index = phase_index
    measured_pos = np.asarray([0.47, -0.16, 0.22], dtype=np.float32)
    measured_quat = np.asarray([0.7, 0.0, -0.7, 0.0], dtype=np.float32)
    controller._prepare_current_phase(
        (np.zeros(3, dtype=np.float32), np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),
        (measured_pos, measured_quat),
        drawer_open_m=0.18,
    )
    np.testing.assert_allclose(controller.current_phase.right.pos, measured_pos)
    np.testing.assert_allclose(controller.current_phase.right.quat_wxyz, measured_quat)


def test_pregrasp_fails_fast_when_open_hand_pushes_can():
    controller = DrawerInsertCloseController(
        _FakeTcpController(),
        initial_action=np.zeros(26, dtype=np.float32),
        anchors=_anchors(),
    )
    phase_index = next(
        index for index, phase in enumerate(controller.phases) if phase.name == "right_pregrasp_can"
    )
    controller.phase_index = phase_index
    controller._task_object_start_position_world = np.asarray([0.52, -0.18, 1.1515], dtype=np.float32)
    controller.phase_steps = 20
    controller._dt = 1.0 / 120.0
    pose = (
        controller.current_phase.right.pos.copy(),
        controller.current_phase.right.quat_wxyz.copy(),
    )
    # The configured gate allows 20 mm; 21 mm must fail immediately.
    displaced = np.asarray([0.541, -0.18, 1.1515], dtype=np.float32)
    assert controller._advance_if_ready(
        pose,
        pose,
        drawer_open_m=0.18,
        task_object_position_world=displaced,
        task_object_linear_velocity_world=np.zeros(3),
    ) is True
    assert controller.done is True
    assert controller.failed is True
    assert "task object displaced" in controller.failure_reason


def test_right_home_finishes_before_the_separate_drawer_close_phase():
    controller = DrawerInsertCloseController(
        _FakeTcpController(),
        initial_action=np.zeros(26, dtype=np.float32),
        anchors=_anchors(),
    )
    phase_index = next(
        index
        for index, phase in enumerate(controller.phases)
        if phase.name == "right_home_after_retreat"
    )
    controller.phase_index = phase_index
    controller.phase_steps = 0
    controller._dt = 1.0 / 120.0
    pose = (np.zeros(3, dtype=np.float32), np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    current = np.zeros(14, dtype=np.float32)
    action, phase_name, _, done = controller.step(current, 1.0 / 120.0, pose, pose, drawer_open_m=0.118)
    assert phase_name == "right_home_after_retreat"
    assert done is False
    np.testing.assert_allclose(action[13:20], controller.home_targets["right"])

    controller.phase_steps = max(
        controller.current_phase.min_steps,
        int(np.ceil(controller.current_phase.hold_seconds * 120.0)),
    )
    hand_action = np.zeros(26, dtype=np.float32)
    hand_action[20:26] = controller.current_phase.right_hand
    # The open drawer must be held while the independently moving right arm
    # returns Home.
    assert controller._advance_if_ready(
        pose,
        pose,
        drawer_open_m=0.118,
        curr_joint_pos=current,
        commanded_action=hand_action,
        actual_action=hand_action,
    ) is False
    assert controller.phase_index == phase_index
    current[7:14] = controller.home_targets["right"]
    # Closing prematurely is rejected during the right-home phase.
    assert controller._advance_if_ready(
        pose,
        pose,
        drawer_open_m=0.010,
        curr_joint_pos=current,
        commanded_action=hand_action,
        actual_action=hand_action,
    ) is False
    assert controller._advance_if_ready(
        pose,
        pose,
        drawer_open_m=0.118,
        curr_joint_pos=current,
        commanded_action=hand_action,
        actual_action=hand_action,
    ) is True
    assert controller.phase_index == phase_index + 1
    assert controller.current_phase.name == "left_close_drawer"
    assert controller.current_phase.right_arm_home is False
