"""Isaac-independent bimanual IK using the repository-vendored Pink source."""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pinocchio as pin
import qpsolvers

from s4_robot.s4_robot_cfg import LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS
from teleoperation.config import PinkConfig
from teleoperation.mapping import TcpPose, matrix_to_quat_wxyz, quat_wxyz_to_matrix
from teleoperation.pink import pink
from teleoperation.pink.pink.barriers import PositionBarrier
from teleoperation.pink.pink.limits import Limit
from teleoperation.pink.pink.tasks import FrameTask, PostureTask


ARM_JOINT_NAMES = tuple(LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS)
LEFT_TCP_FRAME = "s4_left_teleop_tcp"
RIGHT_TCP_FRAME = "s4_right_teleop_tcp"
_JOINT_LIMIT_MARGIN_RAD = 2.0e-6
_MAX_RECOVERABLE_LIMIT_OVERSHOOT_RAD = 1.0e-2


def _movable_joint_names(urdf_file: Path) -> list[str]:
    root = ET.parse(str(urdf_file)).getroot()
    return [
        str(joint.get("name"))
        for joint in root.iter("joint")
        if joint.get("name") and joint.get("type", "") != "fixed"
    ]


class _ArrayConfigurationLimit(Limit):
    """Fixed-base revolute-joint limits without Pinocchio vector conversions."""

    def __init__(self, model: pin.Model, gain: float = 0.5) -> None:
        self.lower = np.asarray(model.lowerPositionLimit, dtype=np.float64).copy()
        self.upper = np.asarray(model.upperPositionLimit, dtype=np.float64).copy()
        self.gain = float(gain)
        self.eye = np.eye(model.nv, dtype=np.float64)

    def compute_qp_inequalities(self, configuration, dt: float):
        del dt
        q = np.asarray(configuration.q, dtype=np.float64)
        maximum = self.gain * (self.upper - q)
        minimum = self.gain * (q - self.lower)
        return np.vstack((self.eye, -self.eye)), np.hstack((maximum, minimum))


class _ArrayVelocityLimit(Limit):
    """Uniform arm velocity limits for the 14-DOF reduced model."""

    def __init__(self, model: pin.Model, maximum_rad_s: float) -> None:
        self.maximum = np.full(model.nv, float(maximum_rad_s), dtype=np.float64)
        self.eye = np.eye(model.nv, dtype=np.float64)

    def compute_qp_inequalities(self, configuration, dt: float):
        del configuration
        step = float(dt) * self.maximum
        return np.vstack((self.eye, -self.eye)), np.hstack((step, step))


class PinkBimanualSolver:
    """Pure Pinocchio/Pink FK and IK with an LA7+RA7 public state contract."""

    def __init__(self, config: PinkConfig) -> None:
        if config.solver not in qpsolvers.available_solvers:
            raise RuntimeError(
                f"Pink QP solver {config.solver!r} is unavailable; "
                f"installed solvers={qpsolvers.available_solvers}"
            )
        self.config = config
        full_model = pin.buildModelFromUrdf(str(config.urdf_file))
        movable_names = _movable_joint_names(config.urdf_file)
        controlled = set(ARM_JOINT_NAMES)
        locked_joint_ids = [
            int(full_model.getJointId(name)) for name in movable_names if name not in controlled
        ]
        self.model = pin.buildReducedModel(
            full_model,
            locked_joint_ids,
            np.asarray(pin.neutral(full_model), dtype=np.float64),
        )
        self._q_index_by_name: dict[str, int] = {}
        self._v_index_by_name: dict[str, int] = {}
        for name in ARM_JOINT_NAMES:
            joint_id = int(self.model.getJointId(name))
            joint = self.model.joints[joint_id]
            if int(joint.nq) == 1 and int(joint.nv) == 1:
                self._q_index_by_name[name] = int(joint.idx_q)
                self._v_index_by_name[name] = int(joint.idx_v)
        missing = [name for name in ARM_JOINT_NAMES if name not in self._q_index_by_name]
        if missing:
            raise RuntimeError(f"Pink URDF is missing controlled arm joints: {missing}")

        self._arm_q_indices = np.asarray(
            [self._q_index_by_name[name] for name in ARM_JOINT_NAMES], dtype=np.int64
        )
        self._arm_v_indices = np.asarray(
            [self._v_index_by_name[name] for name in ARM_JOINT_NAMES], dtype=np.int64
        )
        self._q_template = np.asarray(pin.neutral(self.model), dtype=np.float64)
        self._arm_lower_limits = np.asarray(
            self.model.lowerPositionLimit, dtype=np.float64
        )[self._arm_q_indices].copy()
        self._arm_upper_limits = np.asarray(
            self.model.upperPositionLimit, dtype=np.float64
        )[self._arm_q_indices].copy()
        self._add_tcp_frame(
            LEFT_TCP_FRAME,
            LEFT_ARM_JOINTS[-1],
            config.left_frame_name,
            config.tcp_offset_wrist,
        )
        self._add_tcp_frame(
            RIGHT_TCP_FRAME,
            RIGHT_ARM_JOINTS[-1],
            config.right_frame_name,
            config.tcp_offset_wrist,
        )
        self.data = self.model.createData()
        # Isaac Sim's already-loaded Pinocchio binding cannot convert the
        # std::vector<bool> returned by model.hasConfigurationLimit(). Pink's
        # stock limit constructors call that API. This reduced S4 model is a
        # simpler fixed-base 14x revolute model, so equivalent array-only
        # inequalities are both sufficient and portable to the future
        # non-Isaac hardware process.
        self.configuration_limit = _ArrayConfigurationLimit(self.model)
        self.velocity_limit = _ArrayVelocityLimit(
            self.model, config.max_joint_velocity_rad_s
        )
        self.model.configuration_limit = self.configuration_limit
        self.model.velocity_limit = self.velocity_limit
        self.model.floating_base_velocity_limit = None
        self.configuration = pink.Configuration(self.model, self.data, self._q_template)

        self.left_task = FrameTask(
            LEFT_TCP_FRAME,
            position_cost=config.position_cost,
            orientation_cost=config.orientation_cost,
            lm_damping=config.lm_damping,
            gain=config.task_gain,
        )
        self.right_task = FrameTask(
            RIGHT_TCP_FRAME,
            position_cost=config.position_cost,
            orientation_cost=config.orientation_cost,
            lm_damping=config.lm_damping,
            gain=config.task_gain,
        )
        self.posture_task = PostureTask(cost=config.posture_cost, gain=config.task_gain)
        self.elbow_barriers = []
        elbow = config.elbow_avoidance
        if elbow.enabled:
            missing_elbow_frames = [
                frame
                for frame in (elbow.left_frame_name, elbow.right_frame_name)
                if not self.model.existFrame(frame)
            ]
            if missing_elbow_frames:
                raise RuntimeError(
                    f"Pink URDF is missing elbow avoidance frames: {missing_elbow_frames}"
                )
            distance = elbow.min_lateral_distance_base_m
            self.elbow_barriers = [
                PositionBarrier(
                    frame=elbow.left_frame_name,
                    indices=[1],
                    p_min=np.array([distance], dtype=np.float64),
                    gain=elbow.gain,
                ),
                PositionBarrier(
                    frame=elbow.right_frame_name,
                    indices=[1],
                    p_max=np.array([-distance], dtype=np.float64),
                    gain=elbow.gain,
                ),
            ]
        self.limits = [self.configuration_limit, self.velocity_limit]
        self._posture_reference = self._q_template.copy()
        self._last_solve_ms = 0.0
        self._last_max_velocity = 0.0
        self._last_input_limit_correction = 0.0

    def _add_tcp_frame(
        self,
        tcp_name: str,
        wrist_joint_name: str,
        wrist_frame_name: str,
        offset: np.ndarray,
    ) -> None:
        if not self.model.existFrame(wrist_frame_name):
            raise RuntimeError(f"Pink URDF is missing wrist frame: {wrist_frame_name}")
        placement = pin.SE3(np.eye(3), np.asarray(offset, dtype=np.float64))
        self.model.addFrame(
            pin.Frame(
                tcp_name,
                int(self.model.getJointId(wrist_joint_name)),
                int(self.model.getFrameId(wrist_frame_name)),
                placement,
                pin.FrameType.OP_FRAME,
            )
        )

    def _full_q(self, arm_q14: np.ndarray) -> np.ndarray:
        arm_q = np.asarray(arm_q14, dtype=np.float64)
        if arm_q.shape != (14,) or not np.isfinite(arm_q).all():
            raise ValueError(f"Pink expects finite LA7+RA7 state, got shape={arm_q.shape}")
        q = self._q_template.copy()
        q[self._arm_q_indices] = arm_q
        return q

    def _bounded_full_q(self, arm_q14: np.ndarray) -> np.ndarray:
        """Recover harmless measured-state overshoot at a URDF hard limit."""
        q = self._full_q(arm_q14)
        arm_q = q[self._arm_q_indices]
        below = np.maximum(self._arm_lower_limits - arm_q, 0.0)
        above = np.maximum(arm_q - self._arm_upper_limits, 0.0)
        overshoot = float(np.max(np.maximum(below, above)))
        self._last_input_limit_correction = overshoot
        if overshoot > _MAX_RECOVERABLE_LIMIT_OVERSHOOT_RAD:
            arm_index = int(np.argmax(np.maximum(below, above)))
            raise RuntimeError(
                "Pink measured arm state is too far outside the URDF limit: "
                f"joint={ARM_JOINT_NAMES[arm_index]} value={arm_q[arm_index]:.6f} "
                f"range=[{self._arm_lower_limits[arm_index]:.6f}, "
                f"{self._arm_upper_limits[arm_index]:.6f}] "
                f"overshoot={overshoot:.6f}rad"
            )
        lower = self._arm_lower_limits + _JOINT_LIMIT_MARGIN_RAD
        upper = self._arm_upper_limits - _JOINT_LIMIT_MARGIN_RAD
        q[self._arm_q_indices] = np.clip(arm_q, lower, upper)
        return q

    @staticmethod
    def _target_se3(target: TcpPose) -> pin.SE3:
        return pin.SE3(
            quat_wxyz_to_matrix(target.quat_wxyz),
            np.asarray(target.position, dtype=np.float64),
        )

    def set_posture_reference(self, arm_q14: np.ndarray) -> None:
        self._posture_reference = self._bounded_full_q(arm_q14)

    def forward(self, arm_q14: np.ndarray) -> tuple[TcpPose, TcpPose]:
        self.configuration.update(self._full_q(arm_q14))

        def pose(frame: str) -> TcpPose:
            transform = self.configuration.get_transform_frame_to_world(frame)
            return TcpPose(
                np.asarray(transform.translation, dtype=np.float64).copy(),
                matrix_to_quat_wxyz(np.asarray(transform.rotation, dtype=np.float64)),
            )

        return pose(LEFT_TCP_FRAME), pose(RIGHT_TCP_FRAME)

    def compute(
        self,
        arm_q14: np.ndarray,
        dt: float,
        left_target: TcpPose,
        right_target: TcpPose,
    ) -> np.ndarray:
        solve_dt = float(np.clip(dt, 1.0e-4, 0.05))
        self.configuration.update(self._bounded_full_q(arm_q14))
        self.left_task.set_target(self._target_se3(left_target))
        self.right_task.set_target(self._target_se3(right_target))
        self.posture_task.set_target(self._posture_reference)
        start = time.perf_counter()
        velocity = pink.solve_ik(
            self.configuration,
            [self.left_task, self.right_task, self.posture_task],
            solve_dt,
            solver=self.config.solver,
            damping=self.config.damping,
            limits=self.limits,
            barriers=self.elbow_barriers,
        )
        self._last_solve_ms = (time.perf_counter() - start) * 1000.0
        arm_velocity = np.asarray(velocity, dtype=np.float64)[self._arm_v_indices]
        self._last_max_velocity = float(np.max(np.abs(arm_velocity)))
        q_next = self.configuration.integrate(velocity, solve_dt)
        result = np.asarray(q_next, dtype=np.float64)[self._arm_q_indices]
        result = np.clip(
            result,
            self._arm_lower_limits + _JOINT_LIMIT_MARGIN_RAD,
            self._arm_upper_limits - _JOINT_LIMIT_MARGIN_RAD,
        )
        if result.shape != (14,) or not np.isfinite(result).all():
            raise RuntimeError(f"Pink produced invalid LA7+RA7 target: {result}")
        return result.astype(np.float32)

    def diagnostics(self) -> dict[str, str | float]:
        return {
            "backend": "pink",
            "pink_version": pink.__version__,
            "solver": self.config.solver,
            "dof": 14.0,
            "qp_model_nv": float(self.model.nv),
            "last_solve_ms": self._last_solve_ms,
            "last_max_velocity_rad_s": self._last_max_velocity,
            "last_input_limit_correction_rad": self._last_input_limit_correction,
            "elbow_avoidance_enabled": float(self.config.elbow_avoidance.enabled),
            "elbow_min_lateral_distance_base_m": (
                self.config.elbow_avoidance.min_lateral_distance_base_m
            ),
        }
