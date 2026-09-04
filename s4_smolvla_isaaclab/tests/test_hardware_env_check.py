from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from hardware_teleop.env_check import _check_live_ros_graph, _is_relative_to
from hardware_teleop.pink_main import build_parser


@dataclass
class _PublisherInfo:
    node_namespace: str
    node_name: str


class _GraphNode:
    def __init__(self, *, include_state: bool = True, include_command_source: bool = False) -> None:
        self.include_state = include_state
        self.include_command_source = include_command_source
        self.destroyed = False
        self.publisher_queries: list[str] = []

    def get_topic_names_and_types(self):
        topics = [("/rosout", ["rcl_interfaces/msg/Log"])]
        if self.include_state:
            topics.append(("/lowstate", ["qi/msg/LowState"]))
        return topics

    def get_publishers_info_by_topic(self, topic: str):
        self.publisher_queries.append(topic)
        if self.include_command_source:
            return [_PublisherInfo("/legacy", "old_teleop")]
        return []

    def destroy_node(self) -> None:
        self.destroyed = True


class _FakeRclpy:
    def __init__(self, node: _GraphNode) -> None:
        self.node = node
        self.initialized = False
        self.shutdown_called = False

    def ok(self) -> bool:
        return self.initialized

    def init(self, args=None) -> None:
        del args
        self.initialized = True

    def create_node(self, name: str) -> _GraphNode:
        assert name == "hardware_pink_doctor_read_only"
        return self.node

    def spin_once(self, node: _GraphNode, timeout_sec: float) -> None:
        assert node is self.node
        assert timeout_sec >= 0.0

    def shutdown(self) -> None:
        self.initialized = False
        self.shutdown_called = True


def test_path_containment_does_not_accept_similar_prefix(tmp_path: Path) -> None:
    parent = tmp_path / "packages"
    assert _is_relative_to(parent / "scipy/__init__.py", parent)
    assert not _is_relative_to(tmp_path / "packages-other/scipy/__init__.py", parent)


def test_live_graph_check_is_read_only_and_accepts_free_arm_topic() -> None:
    node = _GraphNode()
    rclpy = _FakeRclpy(node)
    _check_live_ros_graph(
        rclpy,
        lowstate_topic="lowstate",
        arm_command_topic="/lowcmd_replay",
        discovery_timeout_s=0.001,
    )
    assert node.publisher_queries == ["/lowcmd_replay"]
    assert node.destroyed
    assert rclpy.shutdown_called


def test_live_graph_check_rejects_missing_state_and_still_cleans_up() -> None:
    node = _GraphNode(include_state=False)
    rclpy = _FakeRclpy(node)
    with pytest.raises(RuntimeError, match="live robot state topic is missing"):
        _check_live_ros_graph(
            rclpy,
            lowstate_topic="lowstate",
            arm_command_topic="/lowcmd_replay",
            discovery_timeout_s=0.001,
        )
    assert node.destroyed
    assert rclpy.shutdown_called


def test_live_graph_check_rejects_existing_arm_command_source() -> None:
    node = _GraphNode(include_command_source=True)
    rclpy = _FakeRclpy(node)
    with pytest.raises(RuntimeError, match="already has publisher"):
        _check_live_ros_graph(
            rclpy,
            lowstate_topic="lowstate",
            arm_command_topic="/lowcmd_replay",
            discovery_timeout_s=0.001,
        )
    assert node.destroyed
    assert rclpy.shutdown_called


def test_hardware_cli_has_no_safety_gate_bypass() -> None:
    default_args = build_parser().parse_args([])
    assert default_args.arm_output is False
    assert default_args.enabled_arms == "both"
    assert default_args.enable_hands is False
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--allow-unverified-sdk-arm-replay"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--allow-existing-arm-command-publishers"])
