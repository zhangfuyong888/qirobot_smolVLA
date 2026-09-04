"""Isaac-independent Pink IK backend for real-robot teleoperation."""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np

from hardware_teleop.config_loader import HardwareTeleopConfig
from teleoperation.controllers.pink_solver import ARM_JOINT_NAMES, PinkBimanualSolver
from teleoperation.mapping import TcpPose
from teleoperation.pink import pink
from teleoperation.pink.pink.limits import Limit
from teleoperation.pink.pink.tasks import PostureTask


_HARD_LIMIT_MARGIN_RAD = 2.0e-6
_ELBOW_JOINT_NAMES = ("left_elbow_joint", "right_elbow_joint")
_WRIST_PITCH_YAW_JOINT_NAMES = (
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
_SHOULDER_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
)
_SHOULDER_FLIP_JOINT_NAMES = (
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
)


class _ReferenceDeviationLimit(Limit):
    """Bound selected revolute joints around the latest clutch reference."""

    def __init__(
        self,
        model,
        q_indices: np.ndarray,
        v_indices: np.ndarray,
        maximum_deviation_rad: float,
        maximum_recovery_velocity_rad_s: float,
        gain: float = 0.5,
    ) -> None:
        self._q_indices = np.asarray(q_indices, dtype=np.int64)
        self._v_indices = np.asarray(v_indices, dtype=np.int64)
        self._maximum_deviation = float(maximum_deviation_rad)
        self._maximum_recovery_velocity = float(maximum_recovery_velocity_rad_s)
        self._gain = float(gain)
        self._reference = np.zeros(len(self._q_indices), dtype=np.float64)
        self._rows = np.zeros((len(self._v_indices), model.nv), dtype=np.float64)
        self._rows[np.arange(len(self._v_indices)), self._v_indices] = 1.0

    def set_reference(self, q: np.ndarray) -> None:
        self._reference = np.asarray(q, dtype=np.float64)[self._q_indices].copy()

    def compute_qp_inequalities(self, configuration, dt: float):
        current = np.asarray(configuration.q, dtype=np.float64)[self._q_indices]
        lower = self._reference - self._maximum_deviation
        upper = self._reference + self._maximum_deviation
        maximum = self._gain * (upper - current)
        minimum = self._gain * (current - lower)
        recovery_step = max(float(dt), 0.0) * self._maximum_recovery_velocity
        maximum = np.where(maximum < 0.0, np.maximum(maximum, -recovery_step), maximum)
        minimum = np.where(minimum < 0.0, np.maximum(minimum, -recovery_step), minimum)
        return np.vstack((self._rows, -self._rows)), np.hstack((maximum, minimum))


class PinkHardwareIkBackend:
    """Expose the vendored Pink solver with a strict LA7+RA7 contract."""

    name = "pink"

    def __init__(self, config: HardwareTeleopConfig) -> None:
        pink_config = replace(
            config.teleop.controller.pink,
            max_joint_velocity_rad_s=config.ik.max_joint_velocity_rad_s,
        )
        self._controller = PinkBimanualSolver(pink_config)
        self._apply_hardware_joint_limits(config)
        self._limit_cost = float(config.ik.joint_limit_avoidance_cost)
        self._limit_activation_ratio = float(config.ik.joint_limit_activation_ratio)
        self._limit_task = PostureTask(
            cost=np.zeros(self._controller.model.nv, dtype=np.float64),
            gain=float(config.ik.joint_limit_avoidance_gain),
        )
        posture_cost = np.zeros(self._controller.model.nv, dtype=np.float64)
        for name in _SHOULDER_JOINT_NAMES:
            posture_cost[self._controller._v_index_by_name[name]] = float(
                config.ik.shoulder_posture_cost
            )
        for name in _ELBOW_JOINT_NAMES:
            posture_cost[self._controller._v_index_by_name[name]] = float(
                config.ik.elbow_posture_cost
            )
        for name in _WRIST_PITCH_YAW_JOINT_NAMES:
            posture_cost[self._controller._v_index_by_name[name]] = float(
                config.ik.wrist_pitch_yaw_posture_cost
            )
        self._stabilization_posture_task = PostureTask(
            cost=posture_cost,
            gain=self._controller.config.task_gain,
        )
        shoulder_q_indices = np.asarray(
            [
                self._controller._q_index_by_name[name]
                for name in _SHOULDER_FLIP_JOINT_NAMES
            ],
            dtype=np.int64,
        )
        shoulder_v_indices = np.asarray(
            [
                self._controller._v_index_by_name[name]
                for name in _SHOULDER_FLIP_JOINT_NAMES
            ],
            dtype=np.int64,
        )
        self._shoulder_reference_limit = _ReferenceDeviationLimit(
            self._controller.model,
            shoulder_q_indices,
            shoulder_v_indices,
            config.ik.shoulder_max_reference_deviation_rad,
            config.ik.shoulder_max_velocity_rad_s * 0.9,
        )
        self._controller.limits.append(self._shoulder_reference_limit)
        self._shoulder_reference_limit.set_reference(
            self._controller._posture_reference
        )
        self._limit_active_joints = 0
        self._minimum_limit_margin_rad = float("inf")

    def _apply_hardware_joint_limits(self, config: HardwareTeleopConfig) -> None:
        controller = self._controller
        for name in _ELBOW_JOINT_NAMES:
            q_index = controller._q_index_by_name[name]
            arm_index = ARM_JOINT_NAMES.index(name)
            upper = min(
                float(controller._arm_upper_limits[arm_index]),
                float(config.ik.elbow_max_angle_rad),
            )
            controller.model.upperPositionLimit[q_index] = upper
            controller.configuration_limit.upper[q_index] = upper
            controller._arm_upper_limits[arm_index] = upper
        for name in _SHOULDER_JOINT_NAMES:
            v_index = controller._v_index_by_name[name]
            controller.velocity_limit.maximum[v_index] = min(
                controller.velocity_limit.maximum[v_index],
                float(config.ik.shoulder_max_velocity_rad_s),
            )
        for name in _ELBOW_JOINT_NAMES:
            v_index = controller._v_index_by_name[name]
            controller.velocity_limit.maximum[v_index] = min(
                controller.velocity_limit.maximum[v_index],
                float(config.ik.elbow_max_velocity_rad_s),
            )
        for name in _WRIST_PITCH_YAW_JOINT_NAMES:
            v_index = controller._v_index_by_name[name]
            controller.velocity_limit.maximum[v_index] = min(
                controller.velocity_limit.maximum[v_index],
                float(config.ik.wrist_pitch_yaw_max_velocity_rad_s),
            )

    def forward(self, arm_q14: np.ndarray) -> tuple[TcpPose, TcpPose]:
        return self._controller.forward(arm_q14)

    def set_posture_reference(self, arm_q14: np.ndarray) -> None:
        self._controller.set_posture_reference(arm_q14)
        self._shoulder_reference_limit.set_reference(
            self._controller._posture_reference
        )

    def compute(
        self,
        arm_q14: np.ndarray,
        dt: float,
        left_target: TcpPose,
        right_target: TcpPose,
    ) -> np.ndarray:
        controller = self._controller
        solve_dt = float(np.clip(dt, 1.0e-4, 0.05))
        controller.configuration.update(controller._bounded_full_q(arm_q14))
        controller.left_task.set_target(controller._target_se3(left_target))
        controller.right_task.set_target(controller._target_se3(right_target))
        controller.posture_task.set_target(controller._posture_reference)
        self._stabilization_posture_task.set_target(controller._posture_reference)
        self._update_joint_limit_task()

        start = time.perf_counter()
        velocity = pink.solve_ik(
            controller.configuration,
            [
                controller.left_task,
                controller.right_task,
                controller.posture_task,
                self._stabilization_posture_task,
                self._limit_task,
            ],
            solve_dt,
            solver=controller.config.solver,
            damping=controller.config.damping,
            limits=controller.limits,
            barriers=controller.elbow_barriers,
        )
        controller._last_solve_ms = (time.perf_counter() - start) * 1000.0
        arm_velocity = np.asarray(velocity, dtype=np.float64)[
            controller._arm_v_indices
        ]
        controller._last_max_velocity = float(np.max(np.abs(arm_velocity)))
        q_next = controller.configuration.integrate(velocity, solve_dt)
        result = np.asarray(q_next, dtype=np.float64)[controller._arm_q_indices]
        result = np.clip(
            result,
            controller._arm_lower_limits + _HARD_LIMIT_MARGIN_RAD,
            controller._arm_upper_limits - _HARD_LIMIT_MARGIN_RAD,
        )
        if result.shape != (14,) or not np.isfinite(result).all():
            raise RuntimeError(f"Pink produced invalid LA7+RA7 target: {result}")
        return result.astype(np.float32)

    def _update_joint_limit_task(self) -> None:
        """Fade in a weak inward posture objective near either hard limit."""
        controller = self._controller
        arm_q = np.asarray(controller.configuration.q, dtype=np.float64)[
            controller._arm_q_indices
        ]
        lower = controller._arm_lower_limits
        upper = controller._arm_upper_limits
        center = 0.5 * (lower + upper)
        half_range = 0.5 * (upper - lower)
        normalized = np.abs((arm_q - center) / half_range)
        phase = np.clip(
            (normalized - self._limit_activation_ratio)
            / (1.0 - self._limit_activation_ratio),
            0.0,
            1.0,
        )
        smooth_weight = phase * phase * (3.0 - 2.0 * phase)

        safe_arm_q = arm_q.copy()
        active = phase > 0.0
        safe_arm_q[active] = center[active] + (
            np.sign(arm_q[active] - center[active])
            * self._limit_activation_ratio
            * half_range[active]
        )
        target_q = np.asarray(controller.configuration.q, dtype=np.float64).copy()
        target_q[controller._arm_q_indices] = safe_arm_q
        cost = np.zeros(controller.model.nv, dtype=np.float64)
        cost[controller._arm_v_indices] = self._limit_cost * smooth_weight
        self._limit_task.cost = cost
        self._limit_task.set_target(target_q)

        self._limit_active_joints = int(np.count_nonzero(active))
        self._minimum_limit_margin_rad = float(
            np.min(np.minimum(arm_q - lower, upper - arm_q))
        )

    def diagnostics(self) -> dict[str, str | float]:
        details = self._controller.diagnostics()
        details["runtime"] = "hardware_no_isaac"
        details["joint_limit_avoidance_cost"] = self._limit_cost
        details["joint_limit_activation_ratio"] = self._limit_activation_ratio
        details["joint_limit_active_joints"] = float(self._limit_active_joints)
        details["minimum_joint_limit_margin_rad"] = self._minimum_limit_margin_rad
        details["elbow_max_angle_rad"] = float(
            self._controller._arm_upper_limits[
                ARM_JOINT_NAMES.index("left_elbow_joint")
            ]
        )
        details["wrist_pitch_yaw_posture_cost"] = float(
            self._stabilization_posture_task.cost[
                self._controller._v_index_by_name["left_wrist_pitch_joint"]
            ]
        )
        details["wrist_pitch_yaw_max_velocity_rad_s"] = float(
            self._controller.velocity_limit.maximum[
                self._controller._v_index_by_name["left_wrist_pitch_joint"]
            ]
        )
        return details
