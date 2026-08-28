"""ROS2 bridge for real-robot state feedback and lowcmd/hand command output."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Mapping

import numpy as np

from hardware_teleop.config_loader import HardwareGravityCompConfig, HardwareHandsConfig, HardwareRosConfig
from hardware_teleop.hand_mapping import left_hand_positions, right_hand_positions
from hardware_teleop.joint_mapping import (
    ARM_JOINT_NAMES,
    arms_to_bimanual_state,
    bimanual_arm_targets,
    build_sign_map,
    limit_arm_step,
    teleop_to_robot_sign,
)
from hardware_teleop.ros.env import local_ros_install_hint
from hardware_teleop.vendored.gravity_comp import ArmGravityCompensator
from hardware_teleop.vendored.joint_layout import REAL_ROBOT_BODY_JOINT_ORDER
from hardware_teleop.vendored.joint_state_safety import JointStateFrameGuard
from hardware_teleop.vendored.lowstate_decode import decode_lowstate_arm_positions
from s4_robot.control_mapping import ACTION_SLICES


class RosImportError(ImportError):
    """Raised when local ROS2 runtime or qi message types are unavailable."""


def _require_ros_types():
    try:
        import rclpy
        from qi.msg import HandCmd, HandsCmd, LowCmd, LowState, MotorCmd
        from sensor_msgs.msg import JointState
    except ImportError as exc:
        raise RosImportError(
            "ROS2 bridge requires rclpy and the local vendored qi message package.\n"
            + local_ros_install_hint()
        ) from exc
    return rclpy, JointState, LowCmd, LowState, MotorCmd, HandCmd, HandsCmd


class HardwareRobotBridge:
    """Subscribe to robot state and publish lowcmd + handscmd."""

    def __init__(
        self,
        ros_cfg: HardwareRosConfig,
        hands_cfg: HardwareHandsConfig,
        *,
        gravity_cfg: HardwareGravityCompConfig | None = None,
        project_root: Path | None = None,
        check_lowcmd_publishers: bool = True,
    ) -> None:
        rclpy, JointState, LowCmd, LowState, MotorCmd, HandCmd, HandsCmd = _require_ros_types()
        self._rclpy = rclpy
        self._JointState = JointState
        self._LowCmd = LowCmd
        self._LowState = LowState
        self._MotorCmd = MotorCmd
        self._HandCmd = HandCmd
        self._HandsCmd = HandsCmd

        self._ros_cfg = ros_cfg
        self._hands_cfg = hands_cfg
        self._gravity_cfg = gravity_cfg
        self._sign_by_name = build_sign_map(ros_cfg.reversed_joint_names)
        self._joint_order = list(REAL_ROBOT_BODY_JOINT_ORDER)
        self._arm_joint_names = set(ARM_JOINT_NAMES)

        self._gravity_comp: ArmGravityCompensator | None = None
        self._gravity_enabled = False
        self._gravity_ramp_start: float | None = None
        if gravity_cfg is not None and gravity_cfg.enabled:
            self._initialize_gravity_compensation(gravity_cfg, project_root)

        self._guard = JointStateFrameGuard(tuple(ARM_JOINT_NAMES))
        self._positions: dict[str, float] = {}
        self._last_state_time = 0.0
        self._state_ready = False
        self._lock = threading.Lock()

        self._left_hand_rad = np.zeros(6, dtype=np.float32)
        self._right_hand_rad = np.zeros(6, dtype=np.float32)
        self._commanded_arms: dict[str, float] = {}
        self._last_hand_trigger = (0.0, 0.0)

        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = rclpy.create_node("hardware_quest_teleop_bridge")

        if ros_cfg.state_source == "lowstate":
            self._node.create_subscription(
                LowState,
                ros_cfg.lowstate_topic,
                self._on_lowstate,
                50,
            )
        elif ros_cfg.state_source == "joint_states":
            self._node.create_subscription(
                JointState,
                ros_cfg.joint_states_topic,
                self._on_joint_state,
                10,
            )
        else:
            raise ValueError(f"unsupported hardware.state_source: {ros_cfg.state_source!r}")

        if check_lowcmd_publishers:
            self._assert_no_existing_lowcmd_publishers()

        self._lowcmd_pub = self._node.create_publisher(LowCmd, ros_cfg.lowcmd_topic, 10)
        self._hands_pub = self._node.create_publisher(HandsCmd, ros_cfg.hands_cmd_topic, 10)

    def _initialize_gravity_compensation(
        self,
        gravity_cfg: HardwareGravityCompConfig,
        project_root: Path | None,
    ) -> None:
        if project_root is None:
            print("[HW-TELEOP][WARN] gravity compensation disabled: project_root is missing", flush=True)
            return
        urdf_path = (project_root / gravity_cfg.urdf_path).resolve()
        try:
            self._gravity_comp = ArmGravityCompensator(
                urdf_path,
                list(ARM_JOINT_NAMES),
                gravity_vector=gravity_cfg.gravity_vector,
                scale=gravity_cfg.scale,
                sign=gravity_cfg.sign,
                tau_limit=gravity_cfg.tau_limit,
            )
        except Exception as exc:
            print(f"[HW-TELEOP][WARN] gravity compensation disabled: {exc}", flush=True)
            self._gravity_comp = None
            return
        self._gravity_enabled = True
        print(
            f"[HW-TELEOP] gravity compensation enabled scale={gravity_cfg.scale:.2f} "
            f"tau_limit={gravity_cfg.tau_limit:.1f} ramp={gravity_cfg.ramp_time_s:.1f}s "
            f"source={gravity_cfg.source}",
            flush=True,
        )

    def _assert_no_existing_lowcmd_publishers(self) -> None:
        topic = self._ros_cfg.lowcmd_topic
        candidates = [topic]
        if topic.startswith("/"):
            candidates.append(topic.lstrip("/"))
        else:
            candidates.append(f"/{topic}")

        for _ in range(20):
            self._rclpy.spin_once(self._node, timeout_sec=0.05)

        seen: list[str] = []
        for topic_name in candidates:
            try:
                publishers = self._node.get_publishers_info_by_topic(topic_name)
            except Exception:
                continue
            for info in publishers:
                label = f"{info.node_namespace}/{info.node_name}".replace("//", "/")
                if label not in seen:
                    seen.append(label)
        if seen:
            joined = ", ".join(seen)
            raise RuntimeError(
                f"Refusing to start: lowcmd topic {topic!r} already has publisher(s): {joined}. "
                "Stop moveit_mit_arm_bridge or any other lowcmd publisher before hardware teleop."
            )

    @property
    def state_ready(self) -> bool:
        with self._lock:
            return self._state_ready

    @property
    def last_state_age_s(self) -> float:
        with self._lock:
            if self._last_state_time <= 0.0:
                return float("inf")
            return max(time.monotonic() - self._last_state_time, 0.0)

    def is_state_feed_stale(self, max_age_s: float) -> bool:
        if max_age_s <= 0.0:
            return False
        return self.last_state_age_s > max_age_s

    def spin_once(self, timeout_sec: float = 0.0) -> None:
        self._rclpy.spin_once(self._node, timeout_sec=timeout_sec)

    def wait_for_initial_state(self, timeout_s: float) -> None:
        deadline = time.monotonic() + max(timeout_s, 0.0)
        source = self._ros_cfg.state_source
        topic = (
            self._ros_cfg.lowstate_topic
            if source == "lowstate"
            else self._ros_cfg.joint_states_topic
        )
        while time.monotonic() < deadline:
            self.spin_once(timeout_sec=0.05)
            if self.state_ready:
                return
            time.sleep(0.01)
        raise TimeoutError(f"timed out waiting for robot state ({source}) on {topic}")

    def read_bimanual_state(self) -> np.ndarray:
        with self._lock:
            if not self._state_ready:
                raise RuntimeError("hardware joint state is not ready")
            arm_positions = {name: self._positions[name] for name in ARM_JOINT_NAMES}
            return arms_to_bimanual_state(
                arm_positions,
                left_hand=self._left_hand_rad.copy(),
                right_hand=self._right_hand_rad.copy(),
            )

    def update_hand_state_from_rad(self, left_hand: np.ndarray, right_hand: np.ndarray) -> None:
        with self._lock:
            self._left_hand_rad = np.asarray(left_hand, dtype=np.float32).copy()
            self._right_hand_rad = np.asarray(right_hand, dtype=np.float32).copy()

    def publish_arm_command(
        self,
        action: np.ndarray,
        *,
        allow_motion: bool = True,
        hold_commanded: bool = False,
    ) -> dict[str, float]:
        desired = bimanual_arm_targets(action)
        with self._lock:
            if not self._commanded_arms and self._state_ready:
                self._commanded_arms = {name: self._positions[name] for name in ARM_JOINT_NAMES}
            limited = limit_arm_step(
                desired,
                self._commanded_arms,
                max_step_rad=self._ros_cfg.max_joint_step_rad,
            )
            if allow_motion:
                self._commanded_arms = dict(limited)
            if allow_motion:
                publish_targets = dict(self._commanded_arms)
            elif hold_commanded and self._commanded_arms:
                publish_targets = {name: self._commanded_arms[name] for name in ARM_JOINT_NAMES}
            else:
                publish_targets = {
                    name: self._positions.get(name, limited[name]) for name in ARM_JOINT_NAMES
                }
        self._publish_lowcmd(publish_targets)
        return dict(publish_targets)

    def hold_current_arms(self) -> None:
        """Publish one lowcmd frame holding the last measured arm pose (best-effort shutdown)."""
        with self._lock:
            if not self._state_ready:
                return
            targets = {name: self._positions[name] for name in ARM_JOINT_NAMES}
            self._commanded_arms = dict(targets)
        self._publish_lowcmd(targets)

    def publish_hands(self, left_trigger: float, right_trigger: float) -> None:
        left_positions = left_hand_positions(self._hands_cfg, left_trigger)
        right_positions = right_hand_positions(self._hands_cfg, right_trigger)
        self._last_hand_trigger = (float(left_trigger), float(right_trigger))
        msg = self._HandsCmd()
        msg.mode = int(self._hands_cfg.hands_mode)
        msg.mode_ctrl = int(self._hands_cfg.hands_mode_ctrl)
        msg.timestamp = int(self._node.get_clock().now().nanoseconds)

        left = self._HandCmd()
        left.positions = left_positions
        left.durations = [int(self._hands_cfg.duration_ms)] * 6
        left.mode = int(self._hands_cfg.hand_mode)
        left.hand_id = int(self._hands_cfg.left_hand_id)

        right = self._HandCmd()
        right.positions = right_positions
        right.durations = [int(self._hands_cfg.duration_ms)] * 6
        right.mode = int(self._hands_cfg.hand_mode)
        right.hand_id = int(self._hands_cfg.right_hand_id)

        hands = [self._HandCmd(), self._HandCmd()]
        hands[int(self._hands_cfg.left_hand_array_index)] = left
        hands[int(self._hands_cfg.right_hand_array_index)] = right
        msg.hands = hands
        self._hands_pub.publish(msg)

    def close(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()

    def _accept_arm_positions(self, arm_positions: Mapping[str, float]) -> None:
        with self._lock:
            self._positions.update({name: float(arm_positions[name]) for name in ARM_JOINT_NAMES})
            self._last_state_time = time.monotonic()
            self._state_ready = True

    def _on_lowstate(self, msg) -> None:
        arm_positions, validation = decode_lowstate_arm_positions(
            list(msg.motors),
            body_dof=self._ros_cfg.body_dof,
            sign_by_name=self._sign_by_name,
            guard=self._guard,
            arm_names=ARM_JOINT_NAMES,
        )
        if arm_positions is None:
            return
        self._accept_arm_positions(arm_positions)

    def _on_joint_state(self, msg) -> None:
        positions = {name: float(value) for name, value in zip(msg.name, msg.position, strict=False)}
        validation = self._guard.validate(positions)
        if not validation.accepted:
            return
        try:
            arm_positions = {name: positions[name] for name in ARM_JOINT_NAMES}
        except KeyError:
            return
        self._accept_arm_positions(arm_positions)

    def _publish_lowcmd(self, arm_targets: Mapping[str, float]) -> None:
        gravity_efforts = self._gravity_efforts_for_targets(arm_targets)
        msg = self._LowCmd()
        msg.mode = 1
        msg.mode_ak = 1
        msg.mode_ctrl = 5
        msg.timestamp = int(self._node.get_clock().now().nanoseconds)
        motors = []
        for name in self._joint_order:
            motor = self._MotorCmd()
            if name in self._arm_joint_names:
                motor.mode = 1
                sign = teleop_to_robot_sign(name, self._sign_by_name)
                motor.q = float(sign * arm_targets[name])
                motor.dq = 0.0
                motor.tau = float(sign * gravity_efforts.get(name, 0.0))
                motor.kp = float(self._ros_cfg.arm_kp)
                motor.kd = float(self._ros_cfg.arm_kd)
            else:
                motor.mode = 0
                motor.q = 0.0
                motor.dq = 0.0
                motor.tau = 0.0
                motor.kp = 0.0
                motor.kd = 0.0
            motors.append(motor)
        msg.motors = motors
        self._lowcmd_pub.publish(msg)

    def _gravity_ramp_multiplier(self) -> float:
        if self._gravity_cfg is None:
            return 0.0
        ramp_time = self._gravity_cfg.ramp_time_s
        if ramp_time <= 0.0:
            return 1.0
        if self._gravity_ramp_start is None:
            self._gravity_ramp_start = time.monotonic()
        elapsed = time.monotonic() - self._gravity_ramp_start
        return min(1.0, max(0.0, elapsed / ramp_time))

    def _gravity_efforts_for_targets(self, arm_targets: Mapping[str, float]) -> dict[str, float]:
        if not self._gravity_enabled or self._gravity_comp is None or self._gravity_cfg is None:
            return {}
        if self._gravity_cfg.source == "target":
            positions = {name: float(arm_targets[name]) for name in ARM_JOINT_NAMES}
        else:
            with self._lock:
                positions = {
                    name: float(self._positions.get(name, arm_targets[name])) for name in ARM_JOINT_NAMES
                }
        try:
            return self._gravity_comp.compute(
                positions,
                scale_multiplier=self._gravity_ramp_multiplier(),
            )
        except Exception as exc:
            print(f"[HW-TELEOP][WARN] gravity compensation compute failed: {exc}", flush=True)
            self._gravity_enabled = False
            return {}

    def diagnostics(self) -> dict[str, float | str | bool]:
        with self._lock:
            state_age_s = max(time.monotonic() - self._last_state_time, 0.0) if self._last_state_time > 0.0 else float("inf")
            return {
                "state_ready": self._state_ready,
                "state_source": self._ros_cfg.state_source,
                "state_age_s": state_age_s,
                "left_hand_trigger": self._last_hand_trigger[0],
                "right_hand_trigger": self._last_hand_trigger[1],
                "lowcmd_topic": self._ros_cfg.lowcmd_topic,
                "hands_cmd_topic": self._ros_cfg.hands_cmd_topic,
                "gravity_comp_enabled": self._gravity_enabled,
            }
