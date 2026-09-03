from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from hardware_teleop.config_loader import load_hardware_teleop_config
from hardware_teleop.ros import robot_bridge
from hardware_teleop.joint_mapping import ARM_JOINT_NAMES
from s4_robot.control_mapping import bimanual_default_action


ROOT = Path(__file__).resolve().parents[1]


class _Message:
    pass


class _Publisher:
    def __init__(self) -> None:
        self.messages = []

    def publish(self, message) -> None:
        self.messages.append(message)


class _Clock:
    class _Now:
        nanoseconds = 123

    def now(self):
        return self._Now()


class _Node:
    def __init__(self) -> None:
        self.subscription_count = 0
        self.publisher_count = 0
        self.publisher_infos = []
        self.publishers = []

    def create_subscription(self, *_args, **_kwargs):
        self.subscription_count += 1
        return object()

    def create_publisher(self, *_args, **_kwargs):
        self.publisher_count += 1
        publisher = _Publisher()
        self.publishers.append(publisher)
        return publisher

    def get_clock(self):
        return _Clock()

    def get_publishers_info_by_topic(self, _topic: str):
        return list(self.publisher_infos)

    def destroy_node(self) -> None:
        pass


class _Rclpy:
    def __init__(self) -> None:
        self.node = _Node()
        self._ok = False

    def ok(self) -> bool:
        return self._ok

    def init(self, args=None) -> None:
        del args
        self._ok = True

    def create_node(self, _name: str) -> _Node:
        return self.node

    def spin_once(self, node: _Node, timeout_sec: float = 0.0) -> None:
        assert node is self.node
        assert timeout_sec >= 0.0

    def shutdown(self) -> None:
        self._ok = False


def test_shadow_bridge_creates_state_subscription_but_no_command_publishers(monkeypatch) -> None:
    config = load_hardware_teleop_config(ROOT / "hardware_teleop/config/quest_hardware.yaml")
    fake_rclpy = _Rclpy()
    monkeypatch.setattr(
        robot_bridge,
        "_require_ros_types",
        lambda: (fake_rclpy, _Message, _Message, _Message, _Message, _Message, _Message),
    )
    gravity_disabled = replace(config.gravity, enabled=False)

    bridge = robot_bridge.HardwareRobotBridge(
        config.hardware,
        config.hands,
        gravity_cfg=gravity_disabled,
        project_root=config.project_root,
        check_lowcmd_publishers=True,
        command_output_enabled=False,
    )
    try:
        assert fake_rclpy.node.subscription_count == 2
        assert fake_rclpy.node.publisher_count == 0
        assert bridge.diagnostics()["command_output_enabled"] is False
        bridge.publish_hands(1.0, 1.0)
        bridge.hold_current_arms()
        assert fake_rclpy.node.publisher_count == 0
    finally:
        bridge.close()


class _Motor:
    def __init__(self, *, mode: int = 1, q: float = 0.0) -> None:
        self.mode = mode
        self.q = q
        self.dq = 0.0
        self.tau = 0.0
        self.kp = 20.0
        self.kd = 1.0


class _LowCmdFrame:
    def __init__(self, *, mode: int = 1, mode_ctrl: int = 4, motor_mode: int = 1) -> None:
        self.mode = mode
        self.mode_ctrl = mode_ctrl
        self.motors = [_Motor(mode=motor_mode) for _ in range(26)]


def _make_bridge(monkeypatch, *, strict_policy_health: bool = False):
    config = load_hardware_teleop_config(ROOT / "hardware_teleop/config/quest_hardware.yaml")
    fake_rclpy = _Rclpy()
    monkeypatch.setattr(
        robot_bridge,
        "_require_ros_types",
        lambda: (fake_rclpy, _Message, _Message, _Message, _Message, _Message, _Message),
    )
    bridge = robot_bridge.HardwareRobotBridge(
        config.hardware,
        config.hands,
        gravity_cfg=replace(config.gravity, enabled=False),
        startup_cfg=config.startup if strict_policy_health else None,
        project_root=config.project_root,
        check_lowcmd_publishers=False,
        command_output_enabled=False,
    )
    return config, bridge


def test_policy_gate_requires_healthy_stable_robot_feedback(monkeypatch) -> None:
    _config, bridge = _make_bridge(monkeypatch, strict_policy_health=True)
    try:
        bridge._last_state_time = time.monotonic()
        bridge._state_leg_positions = (0.0,) * 12
        bridge._state_health_rejection = "robot roll/pitch exceeds standing threshold"
        bridge._on_observed_lowcmd(_LowCmdFrame(mode_ctrl=4))
        assert bridge.diagnostics()["valid_policy_frames"] == 0
        assert "roll/pitch" in str(bridge.diagnostics()["last_policy_rejection"])

        bridge._state_health_rejection = ""
        bridge._on_observed_lowcmd(_LowCmdFrame(mode_ctrl=4))
        diagnostics = bridge.diagnostics()
        assert diagnostics["valid_policy_frames"] == 1
        assert diagnostics["policy_arm_target_ready"] is True
        assert diagnostics["policy_stable_s"] >= 0.0
    finally:
        bridge.close()


def test_policy_gate_accepts_only_enabled_non_mode5_leg_packets(monkeypatch) -> None:
    _config, bridge = _make_bridge(monkeypatch)
    try:
        bridge._on_observed_lowcmd(_LowCmdFrame(mode_ctrl=4))
        assert bridge.diagnostics()["valid_policy_frames"] == 1

        bridge._on_observed_lowcmd(_LowCmdFrame(mode_ctrl=4, motor_mode=0))
        assert bridge.diagnostics()["valid_policy_frames"] == 0
        assert "disabled" in str(bridge.diagnostics()["last_policy_rejection"])

        bridge._on_observed_lowcmd(_LowCmdFrame(mode_ctrl=5))
        with np.testing.assert_raises_regex(RuntimeError, "another mode_ctrl=5"):
            bridge.wait_for_policy_lowcmd(0.01, 1, 0.2)
    finally:
        bridge.close()


def test_inactive_hold_reanchors_step_limiter_to_measured_state(monkeypatch) -> None:
    _config, bridge = _make_bridge(monkeypatch)
    try:
        measured = {name: 0.0 for name in ARM_JOINT_NAMES}
        bridge._accept_arm_positions(measured)
        bridge._commanded_arms = {name: 1.0 for name in ARM_JOINT_NAMES}
        bridge._publish_lowcmd = lambda _targets: None

        action = bimanual_default_action()
        held = bridge.publish_arm_command(action, allow_motion=False)
        assert max(abs(value) for value in held.values()) == 0.0
        assert max(abs(value) for value in bridge._commanded_arms.values()) == 0.0
    finally:
        bridge.close()


def test_runtime_graph_monitor_rejects_second_external_lowcmd_source(monkeypatch) -> None:
    _config, bridge = _make_bridge(monkeypatch)
    try:
        node = bridge._node
        node.publisher_infos = [
            SimpleNamespace(node_namespace="/robot", node_name="standing_policy"),
            SimpleNamespace(node_namespace="/legacy", node_name="old_teleop"),
            SimpleNamespace(
                node_namespace="/", node_name="hardware_quest_teleop_bridge"
            ),
        ]
        assert bridge.is_lowcmd_graph_conflicted(check_period_s=0.0)
        assert bridge.diagnostics()["external_lowcmd_publishers"] == (
            "/robot/standing_policy",
            "/legacy/old_teleop",
        )
    finally:
        bridge.close()


def test_controlled_release_slews_to_cached_policy_target(monkeypatch) -> None:
    config = load_hardware_teleop_config(ROOT / "hardware_teleop/config/quest_hardware.yaml")
    fake_rclpy = _Rclpy()
    monkeypatch.setattr(
        robot_bridge,
        "_require_ros_types",
        lambda: (fake_rclpy, _Message, _Message, _Message, _Message, _Message, _Message),
    )
    bridge = robot_bridge.HardwareRobotBridge(
        replace(config.hardware, release_duration_s=0.2, release_tolerance_rad=0.001),
        config.hands,
        gravity_cfg=replace(config.gravity, enabled=False),
        project_root=config.project_root,
        check_lowcmd_publishers=False,
        command_output_enabled=True,
    )
    try:
        bridge._commanded_arms = {name: 0.012 for name in ARM_JOINT_NAMES}
        bridge._policy_arm_targets = {name: 0.0 for name in ARM_JOINT_NAMES}
        bridge._last_policy_time = time.monotonic()
        bridge._has_published_lowcmd = True
        assert bridge.release_to_policy("unit test")
        assert bridge.diagnostics()["command_output_relinquished"] is True
        assert len(fake_rclpy.node.publishers[0].messages) >= 2
        with np.testing.assert_raises_regex(RuntimeError, "relinquished"):
            bridge.publish_arm_command(bimanual_default_action())
    finally:
        bridge.close()
