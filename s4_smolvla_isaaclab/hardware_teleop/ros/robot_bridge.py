"""ROS2 bridge for real-robot state feedback and arm-replay/hand output."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Mapping

import numpy as np

from hardware_teleop.config_loader import (
    HardwareGravityCompConfig,
    HardwareHandsConfig,
    HardwareRosConfig,
)
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


_READY_CALLBACK_DRAIN_LIMIT = 8


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
        check_arm_command_publishers: bool = True,
        command_output_enabled: bool = True,
        expected_command_publishers: tuple[str, ...] = (),
        expected_command_mode_ctrl: int | None = None,
        forbidden_command_publishers: tuple[tuple[str, str], ...] = (),
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
        self._command_output_enabled = bool(command_output_enabled)
        self._expected_command_publishers = tuple(
            self._normalize_node_label(label) for label in expected_command_publishers
        )
        self._expected_command_mode_ctrl = (
            None
            if expected_command_mode_ctrl is None
            else int(expected_command_mode_ctrl)
        )
        if self._expected_command_publishers and self._expected_command_mode_ctrl is None:
            raise ValueError(
                "expected_command_mode_ctrl is required with expected command publishers"
            )
        self._forbidden_command_publishers = tuple(
            (str(topic), self._normalize_node_label(label))
            for topic, label in forbidden_command_publishers
        )

        self._gravity_comp: ArmGravityCompensator | None = None
        self._gravity_enabled = False
        self._gravity_ramp_start: float | None = None
        self._last_gravity_motor_tau: dict[str, float] = {}
        if self._command_output_enabled and gravity_cfg is not None and gravity_cfg.enabled:
            self._initialize_gravity_compensation(gravity_cfg, project_root)

        self._guard = JointStateFrameGuard(
            tuple(ARM_JOINT_NAMES),
            max_position_jump=ros_cfg.max_state_joint_jump_rad,
        )
        self._positions: dict[str, float] = {}
        self._last_state_time = 0.0
        self._state_ready = False
        self._state_rejection_count = 0
        self._last_state_rejection = ""
        self._has_published_lowcmd = False
        self._last_graph_check_time = 0.0
        self._last_external_arm_command_publishers: tuple[str, ...] = ()
        self._last_command_publisher_conflicts: tuple[str, ...] = ()
        self._last_expected_command_time = 0.0
        self._last_expected_command_mode_ctrl: int | None = None
        self._lock = threading.Lock()
        self._spin_lock = threading.Lock()
        self._command_lock = threading.RLock()
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: threading.Thread | None = None
        self._last_command_heartbeat = 0.0
        self._release_active = False
        self._output_relinquished = False
        self._release_reason = ""

        self._left_hand_rad = np.zeros(6, dtype=np.float32)
        self._right_hand_rad = np.zeros(6, dtype=np.float32)
        self._commanded_arms: dict[str, float] = {}
        self._last_hand_trigger = (0.0, 0.0)
        self._hands_ever_commanded = False

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

        if self._expected_command_publishers:
            self._node.create_subscription(
                LowCmd,
                ros_cfg.arm_command_topic,
                self._on_expected_command,
                10,
            )

        if self._command_output_enabled and (
            check_arm_command_publishers or self._forbidden_command_publishers
        ):
            self._check_arm_command_publisher_count()

        self._lowcmd_pub = None
        self._hands_pub = None
        if self._command_output_enabled:
            self._lowcmd_pub = self._node.create_publisher(
                LowCmd, ros_cfg.arm_command_topic, 10
            )
            self._hands_pub = self._node.create_publisher(HandsCmd, ros_cfg.hands_cmd_topic, 10)
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_loop,
                name="hardware-command-watchdog",
                daemon=True,
            )
            self._watchdog_thread.start()

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
            f"sign={gravity_cfg.sign:.1f} tau_limit={gravity_cfg.tau_limit:.1f} "
            f"ramp={gravity_cfg.ramp_time_s:.1f}s source={gravity_cfg.source}",
            flush=True,
        )

    def _on_expected_command(self, msg) -> None:
        mode_ctrl = int(msg.mode_ctrl)
        if mode_ctrl != self._expected_command_mode_ctrl:
            return
        with self._lock:
            self._last_expected_command_time = time.monotonic()
            self._last_expected_command_mode_ctrl = mode_ctrl

    def _arm_command_publisher_labels(
        self, *, discovery_timeout_s: float = 0.0
    ) -> tuple[str, ...]:
        return self._publisher_labels(
            self._ros_cfg.arm_command_topic,
            discovery_timeout_s=discovery_timeout_s,
        )

    def _publisher_labels(
        self,
        topic: str,
        *,
        discovery_timeout_s: float = 0.0,
    ) -> tuple[str, ...]:
        candidates = [topic]
        if topic.startswith("/"):
            candidates.append(topic.lstrip("/"))
        else:
            candidates.append(f"/{topic}")

        deadline = time.monotonic() + max(float(discovery_timeout_s), 0.0)
        while time.monotonic() < deadline:
            self._rclpy.spin_once(
                self._node,
                timeout_sec=min(0.05, max(deadline - time.monotonic(), 0.0)),
            )

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
        return tuple(seen)

    @staticmethod
    def _normalize_node_label(label: str) -> str:
        return "/" + str(label).strip("/")

    @staticmethod
    def _external_arm_command_publishers(labels: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            label
            for label in labels
            if label != "/hardware_quest_teleop_bridge"
        )

    def _command_publisher_conflicts(
        self,
        labels: tuple[str, ...],
    ) -> tuple[str, ...]:
        observed = tuple(self._normalize_node_label(label) for label in labels)
        expected = self._expected_command_publishers
        unexpected = tuple(label for label in observed if label not in expected)
        missing = tuple(f"missing:{label}" for label in expected if label not in observed)
        return unexpected + missing

    def _forbidden_publisher_conflicts(self) -> tuple[str, ...]:
        conflicts: list[str] = []
        for topic, forbidden_label in self._forbidden_command_publishers:
            labels = tuple(
                self._normalize_node_label(label)
                for label in self._publisher_labels(topic)
            )
            if forbidden_label in labels:
                conflicts.append(f"forbidden:{forbidden_label}@{topic}")
        return tuple(conflicts)

    def _check_arm_command_publisher_count(self) -> None:
        topic = self._ros_cfg.arm_command_topic
        discovery_timeout_s = 3.0 if self._expected_command_publishers else 1.0
        seen = self._external_arm_command_publishers(
            self._arm_command_publisher_labels(
                discovery_timeout_s=discovery_timeout_s
            )
        )
        conflicts = (
            self._command_publisher_conflicts(seen)
            + self._forbidden_publisher_conflicts()
        )
        with self._lock:
            self._last_external_arm_command_publishers = seen
            self._last_command_publisher_conflicts = conflicts
        if conflicts:
            raise RuntimeError(
                f"Refusing to start: command publisher contract for {topic!r} is not met: "
                f"{', '.join(conflicts)}."
            )

    def is_arm_command_graph_conflicted(self, check_period_s: float = 0.2) -> bool:
        """Return whether another arm-replay command source is visible."""
        now = time.monotonic()
        with self._lock:
            if now - self._last_graph_check_time < max(float(check_period_s), 0.0):
                return bool(self._last_command_publisher_conflicts)
        labels = self._external_arm_command_publishers(
            self._arm_command_publisher_labels()
        )
        conflicts = (
            self._command_publisher_conflicts(labels)
            + self._forbidden_publisher_conflicts()
        )
        with self._lock:
            self._last_graph_check_time = now
            self._last_external_arm_command_publishers = labels
            self._last_command_publisher_conflicts = conflicts
        return bool(conflicts)

    @property
    def command_publisher_conflicts(self) -> tuple[str, ...]:
        with self._lock:
            return self._last_command_publisher_conflicts

    @property
    def expected_command_age_s(self) -> float:
        if not self._expected_command_publishers:
            return 0.0
        with self._lock:
            if self._last_expected_command_time <= 0.0:
                return float("inf")
            return max(time.monotonic() - self._last_expected_command_time, 0.0)

    def is_expected_command_feed_stale(self, max_age_s: float) -> bool:
        if not self._expected_command_publishers:
            return False
        return self.expected_command_age_s > max(float(max_age_s), 0.0)

    def wait_for_expected_command(self, timeout_s: float) -> None:
        if not self._expected_command_publishers:
            return
        deadline = time.monotonic() + max(float(timeout_s), 0.0)
        while time.monotonic() < deadline:
            self.spin_once(timeout_sec=0.05)
            if self._last_expected_command_time > 0.0:
                return
        topic = self._ros_cfg.arm_command_topic
        raise TimeoutError(
            f"timed out waiting for leg-deploy mode_ctrl="
            f"{self._expected_command_mode_ctrl} on {topic}"
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

    @property
    def output_relinquished(self) -> bool:
        return bool(self._output_relinquished)

    def spin_once(self, timeout_sec: float = 0.0) -> None:
        with self._spin_lock:
            self._rclpy.spin_once(self._node, timeout_sec=timeout_sec)
            if timeout_sec <= 0.0:
                for _ in range(_READY_CALLBACK_DRAIN_LIMIT - 1):
                    self._rclpy.spin_once(self._node, timeout_sec=0.0)

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
        with self._command_lock:
            if self._output_relinquished or self._release_active:
                return {}
            desired = bimanual_arm_targets(action)
            with self._lock:
                if not self._commanded_arms and self._state_ready:
                    self._commanded_arms = {
                        name: self._positions[name] for name in ARM_JOINT_NAMES
                    }
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
                    publish_targets = {
                        name: self._commanded_arms[name] for name in ARM_JOINT_NAMES
                    }
                else:
                    publish_targets = {
                        name: self._positions.get(name, limited[name])
                        for name in ARM_JOINT_NAMES
                    }
                    self._commanded_arms = dict(publish_targets)
            self._publish_lowcmd(publish_targets)
            self._last_command_heartbeat = time.monotonic()
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
        if not self._command_output_enabled:
            return
        with self._command_lock:
            if self._output_relinquished:
                return
            self._hands_ever_commanded = True
            self._publish_hands_unlocked(left_trigger, right_trigger)

    def _publish_hands_unlocked(self, left_trigger: float, right_trigger: float) -> None:
        left_positions = left_hand_positions(self._hands_cfg, left_trigger)
        right_positions = right_hand_positions(self._hands_cfg, right_trigger)
        self._last_hand_trigger = (float(left_trigger), float(right_trigger))
        msg = self._HandsCmd()
        msg.mode = int(self._hands_cfg.hands_mode)
        msg.mode_ctrl = int(self._hands_cfg.hands_mode_ctrl)
        msg.timestamp = int(self._node.get_clock().now().nanoseconds)

        left = self._HandCmd()
        duration = int(np.clip(self._hands_cfg.duration_ms, 0, 255))
        left.positions = left_positions
        left.durations = [duration] * 6
        left.mode = int(self._hands_cfg.hand_mode)
        left.hand_id = int(self._hands_cfg.left_hand_id)

        right = self._HandCmd()
        right.positions = right_positions
        right.durations = [duration] * 6
        right.mode = int(self._hands_cfg.hand_mode)
        right.hand_id = int(self._hands_cfg.right_hand_id)

        hands = [self._HandCmd(), self._HandCmd()]
        hands[int(self._hands_cfg.left_hand_array_index)] = left
        hands[int(self._hands_cfg.right_hand_array_index)] = right
        msg.hands = hands
        assert self._hands_pub is not None
        self._hands_pub.publish(msg)

    def hold_current_and_relinquish(self, reason: str = "controlled shutdown") -> bool:
        """Briefly hold measured arms, then stop this process from publishing."""
        if not self._command_output_enabled:
            return True
        with self._command_lock:
            if self._output_relinquished:
                return True
            with self._lock:
                has_lowcmd_output = self._has_published_lowcmd
            if not has_lowcmd_output:
                if self._hands_ever_commanded:
                    self._publish_hands_unlocked(0.0, 0.0)
                self._output_relinquished = True
                return True
            self._release_active = True
            self._release_reason = reason
            print(f"[HW-TELEOP][SAFETY] releasing command output: {reason}", flush=True)
            deadline = time.monotonic() + self._ros_cfg.shutdown_hold_duration_s
            published_hold = False
            while time.monotonic() < deadline and not self._watchdog_stop.is_set():
                self.spin_once(timeout_sec=0.0)
                with self._lock:
                    target = dict(self._positions or self._commanded_arms)
                if self._hands_ever_commanded:
                    self._publish_hands_unlocked(0.0, 0.0)
                if len(target) != len(ARM_JOINT_NAMES):
                    break
                with self._lock:
                    self._commanded_arms = dict(target)
                self._publish_lowcmd(target)
                published_hold = True
                time.sleep(1.0 / self._ros_cfg.control_rate_hz)
            if self._hands_ever_commanded:
                self._publish_hands_unlocked(0.0, 0.0)
            self._output_relinquished = True
            self._release_active = False
            status = "current pose held" if published_hold else "no valid hold pose"
            print(f"[HW-TELEOP][SAFETY] arm-replay {status}; publisher relinquished", flush=True)
            return published_hold

    def relinquish_without_arm_hold(self, reason: str) -> None:
        """Stop this publisher immediately when another arm source is detected."""
        with self._command_lock:
            if self._output_relinquished:
                return
            self._release_active = True
            self._release_reason = reason
            if self._hands_ever_commanded:
                self._publish_hands_unlocked(0.0, 0.0)
            self._output_relinquished = True
            self._release_active = False
            print(
                f"[HW-TELEOP][SAFETY] arm-replay publisher relinquished immediately: {reason}",
                flush=True,
            )

    def _watchdog_loop(self) -> None:
        period = min(max(self._ros_cfg.command_watchdog_timeout_s / 4.0, 0.01), 0.05)
        while not self._watchdog_stop.wait(period):
            with self._lock:
                has_output = self._has_published_lowcmd
            if not has_output or self._output_relinquished or self._release_active:
                continue
            heartbeat = self._last_command_heartbeat
            if heartbeat > 0.0 and time.monotonic() - heartbeat > self._ros_cfg.command_watchdog_timeout_s:
                try:
                    self.hold_current_and_relinquish("command heartbeat timeout")
                except Exception as exc:
                    self._output_relinquished = True
                    print(f"[HW-TELEOP][SAFETY] watchdog fallback failed: {exc}", flush=True)
                return

    def close(self) -> None:
        self._watchdog_stop.set()
        if (
            self._watchdog_thread is not None
            and self._watchdog_thread is not threading.current_thread()
        ):
            self._watchdog_thread.join(timeout=1.0)
        if self._node is not None:
            self._node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()

    def _accept_arm_positions(self, arm_positions: Mapping[str, float]) -> None:
        with self._lock:
            self._positions.update({name: float(arm_positions[name]) for name in ARM_JOINT_NAMES})
            self._last_state_time = time.monotonic()
            self._state_ready = True
            self._last_state_rejection = ""

    def _on_lowstate(self, msg) -> None:
        arm_positions, validation = decode_lowstate_arm_positions(
            list(msg.motors),
            body_dof=self._ros_cfg.body_dof,
            sign_by_name=self._sign_by_name,
            guard=self._guard,
            arm_names=ARM_JOINT_NAMES,
        )
        if arm_positions is None:
            with self._lock:
                self._state_rejection_count += 1
                self._last_state_rejection = validation.reason
            return
        self._accept_arm_positions(arm_positions)

    def _on_joint_state(self, msg) -> None:
        positions = {name: float(value) for name, value in zip(msg.name, msg.position, strict=False)}
        validation = self._guard.validate(positions)
        if not validation.accepted:
            with self._lock:
                self._state_rejection_count += 1
                self._last_state_rejection = validation.reason
            return
        try:
            arm_positions = {name: positions[name] for name in ARM_JOINT_NAMES}
        except KeyError:
            return
        self._accept_arm_positions(arm_positions)

    def _publish_lowcmd(self, arm_targets: Mapping[str, float]) -> None:
        if not self._command_output_enabled:
            return
        gravity_efforts = self._gravity_efforts_for_targets(arm_targets)
        msg = self._LowCmd()
        msg.mode = 1
        msg.mode_ak = 1
        msg.mode_ctrl = int(self._ros_cfg.arm_command_mode_ctrl)
        msg.timestamp = int(self._node.get_clock().now().nanoseconds)
        motors = []
        gravity_motor_tau: dict[str, float] = {}
        for name in self._joint_order:
            motor = self._MotorCmd()
            if name in self._arm_joint_names:
                motor.mode = 1
                sign = teleop_to_robot_sign(name, self._sign_by_name)
                motor.q = float(sign * arm_targets[name])
                motor.dq = 0.0
                motor.tau = float(sign * gravity_efforts.get(name, 0.0))
                gravity_motor_tau[name] = motor.tau
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
        assert self._lowcmd_pub is not None
        with self._lock:
            self._has_published_lowcmd = True
            self._last_gravity_motor_tau = gravity_motor_tau
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
                "state_rejection_count": self._state_rejection_count,
                "last_state_rejection": self._last_state_rejection,
                "external_arm_command_publishers": self._last_external_arm_command_publishers,
                "expected_command_publishers": self._expected_command_publishers,
                "expected_command_required_mode_ctrl": self._expected_command_mode_ctrl,
                "forbidden_command_publishers": self._forbidden_command_publishers,
                "command_publisher_conflicts": self._last_command_publisher_conflicts,
                "expected_command_age_s": (
                    0.0
                    if not self._expected_command_publishers
                    else (
                        float("inf")
                        if self._last_expected_command_time <= 0.0
                        else max(
                            time.monotonic() - self._last_expected_command_time,
                            0.0,
                        )
                    )
                ),
                "expected_command_mode_ctrl": self._last_expected_command_mode_ctrl,
                "left_hand_trigger": self._last_hand_trigger[0],
                "right_hand_trigger": self._last_hand_trigger[1],
                "arm_command_topic": self._ros_cfg.arm_command_topic,
                "arm_command_mode_ctrl": self._ros_cfg.arm_command_mode_ctrl,
                "hands_cmd_topic": self._ros_cfg.hands_cmd_topic,
                "gravity_comp_enabled": self._gravity_enabled,
                "gravity_sign": (
                    0.0 if self._gravity_cfg is None else float(self._gravity_cfg.sign)
                ),
                "gravity_tau_abs_max": (
                    max((abs(value) for value in self._last_gravity_motor_tau.values()), default=0.0)
                ),
                "gravity_tau_left_shoulder_pitch": self._last_gravity_motor_tau.get(
                    "left_shoulder_pitch_joint", 0.0
                ),
                "gravity_tau_right_shoulder_pitch": self._last_gravity_motor_tau.get(
                    "right_shoulder_pitch_joint", 0.0
                ),
                "command_output_enabled": self._command_output_enabled,
                "command_output_relinquished": self._output_relinquished,
                "release_reason": self._release_reason,
            }
