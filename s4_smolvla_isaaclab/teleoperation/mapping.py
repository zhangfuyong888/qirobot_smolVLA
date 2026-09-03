"""Relative clutch mapping from WebXR controllers to base_link TCP targets."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import TeleopConfig
from .protocol import ControllerFrame, ControllerSample


@dataclass(frozen=True)
class TcpPose:
    position: np.ndarray
    quat_wxyz: np.ndarray


@dataclass(frozen=True)
class SideMappingResult:
    target: TcpPose
    hand6: np.ndarray
    clutch: bool
    clutch_rising: bool
    clutch_falling: bool
    tracking_valid: bool


@dataclass(frozen=True)
class BimanualMappingResult:
    left: SideMappingResult
    right: SideMappingResult
    stale: bool
    frame_age_s: float


@dataclass
class _SideState:
    target_position: np.ndarray | None = None
    target_rotation: np.ndarray | None = None
    controller_reference_position: np.ndarray | None = None
    controller_reference_rotation: np.ndarray | None = None
    tcp_reference_position: np.ndarray | None = None
    tcp_reference_rotation: np.ndarray | None = None
    clutch: bool = False
    requires_release: bool = False
    trigger_filtered: float = 0.0


def quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quat, dtype=np.float64)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1.0e-9:
        return np.eye(3)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=np.float64)
    return quat_xyzw_to_matrix(np.array([x, y, z, w]))


def matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = np.array([0.25 * s, (matrix[2, 1] - matrix[1, 2]) / s, (matrix[0, 2] - matrix[2, 0]) / s, (matrix[1, 0] - matrix[0, 1]) / s])
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            s = math.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 0.0)) * 2.0
            quat = np.array([(matrix[2, 1] - matrix[1, 2]) / s, 0.25 * s, (matrix[0, 1] + matrix[1, 0]) / s, (matrix[0, 2] + matrix[2, 0]) / s])
        elif index == 1:
            s = math.sqrt(max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 0.0)) * 2.0
            quat = np.array([(matrix[0, 2] - matrix[2, 0]) / s, (matrix[0, 1] + matrix[1, 0]) / s, 0.25 * s, (matrix[1, 2] + matrix[2, 1]) / s])
        else:
            s = math.sqrt(max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 0.0)) * 2.0
            quat = np.array([(matrix[1, 0] - matrix[0, 1]) / s, (matrix[0, 2] + matrix[2, 0]) / s, (matrix[1, 2] + matrix[2, 1]) / s, 0.25 * s])
    quat /= max(float(np.linalg.norm(quat)), 1.0e-9)
    if quat[0] < 0.0:
        quat = -quat
    return quat


def _rotation_vector(matrix: np.ndarray) -> np.ndarray:
    cos_angle = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cos_angle)
    if angle < 1.0e-7:
        return np.zeros(3)
    if math.pi - angle < 1.0e-5:
        values, vectors = np.linalg.eigh((matrix + np.eye(3)) * 0.5)
        axis = vectors[:, int(np.argmax(values))]
        return axis / max(float(np.linalg.norm(axis)), 1.0e-9) * angle
    axis = np.array([matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]])
    axis /= 2.0 * math.sin(angle)
    return axis * angle


def _rotation_from_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    angle = float(np.linalg.norm(vector))
    if angle < 1.0e-9:
        return np.eye(3)
    x, y, z = vector / angle
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


class BimanualTeleopMapper:
    """Map independently clutched Quest controllers to smooth bimanual TCP targets."""

    def __init__(self, config: TeleopConfig, left_open: np.ndarray, left_close: np.ndarray, right_open: np.ndarray, right_close: np.ndarray):
        self.config = config
        self.hand_profiles = {
            "left": (np.asarray(left_open, dtype=np.float64), np.asarray(left_close, dtype=np.float64)),
            "right": (np.asarray(right_open, dtype=np.float64), np.asarray(right_close, dtype=np.float64)),
        }
        for side, (opened, closed) in self.hand_profiles.items():
            if opened.shape != (6,) or closed.shape != (6,):
                raise ValueError(f"{side} hand profiles must have shape (6,)")
        self.states = {"left": _SideState(), "right": _SideState()}

    def clutch_value(self, sample: ControllerSample | None) -> float:
        """Return the configured clutch input, including WebXR event fallback."""
        if sample is None:
            return 0.0
        values = [float(sample.squeeze)]
        values.extend(
            float(sample.buttons[index])
            for index in self.config.mapping.clutch.button_indices
            if index < len(sample.buttons)
        )
        return max(values, default=0.0)

    def _update_side(self, side: str, sample: ControllerSample | None, current_tcp: TcpPose, dt: float, stale: bool) -> SideMappingResult:
        state = self.states[side]
        if state.target_position is None:
            state.target_position = np.asarray(current_tcp.position, dtype=np.float64).copy()
            state.target_rotation = quat_wxyz_to_matrix(current_tcp.quat_wxyz)

        tracking_valid = bool(sample is not None and sample.valid and not stale)
        clutch_input = self.clutch_value(sample)
        previous_clutch = state.clutch
        if not tracking_valid:
            state.clutch = False
            state.requires_release = True
        elif state.requires_release:
            state.clutch = False
            if clutch_input <= self.config.mapping.clutch.release_threshold:
                state.requires_release = False
        elif state.clutch:
            state.clutch = clutch_input > self.config.mapping.clutch.release_threshold
        else:
            state.clutch = clutch_input >= self.config.mapping.clutch.engage_threshold
        clutch_rising = state.clutch and not previous_clutch
        clutch_falling = previous_clutch and not state.clutch

        if tracking_valid:
            trigger = float(sample.trigger)
            deadband = max(float(self.config.smoothing.trigger_deadband), 0.0)
            trigger = 0.0 if trigger <= deadband else (trigger - deadband) / max(1.0 - deadband, 1.0e-6)
            tau = max(float(self.config.smoothing.trigger_time_constant_s), 1.0e-5)
            alpha = 1.0 - math.exp(-max(dt, 0.0) / tau)
            state.trigger_filtered += alpha * (trigger - state.trigger_filtered)

        if clutch_rising and sample is not None:
            state.controller_reference_position = np.asarray(sample.position, dtype=np.float64)
            state.controller_reference_rotation = quat_xyzw_to_matrix(np.asarray(sample.orientation_xyzw))
            state.tcp_reference_position = np.asarray(current_tcp.position, dtype=np.float64).copy()
            state.tcp_reference_rotation = quat_wxyz_to_matrix(current_tcp.quat_wxyz)
            state.target_position = state.tcp_reference_position.copy()
            state.target_rotation = state.tcp_reference_rotation.copy()

        if clutch_falling:
            state.target_position = np.asarray(current_tcp.position, dtype=np.float64).copy()
            state.target_rotation = quat_wxyz_to_matrix(current_tcp.quat_wxyz)

        if state.clutch and sample is not None and state.controller_reference_position is not None:
            basis = self.config.mapping.controller_to_base_rotation
            controller_position = np.asarray(sample.position, dtype=np.float64)
            requested_position = state.tcp_reference_position + self.config.mapping.position_scale * (
                basis @ (controller_position - state.controller_reference_position)
            )
            clutch_delta = requested_position - state.tcp_reference_position
            clutch_distance = float(np.linalg.norm(clutch_delta))
            max_clutch_distance = self.config.mapping.max_clutch_translation_m
            if clutch_distance > max_clutch_distance:
                clutch_delta *= max_clutch_distance / clutch_distance
                requested_position = state.tcp_reference_position + clutch_delta
            requested_position = np.clip(
                requested_position,
                self.config.safety.workspace_min,
                self.config.safety.workspace_max,
            )
            delta = requested_position - state.target_position
            max_distance = max(self.config.safety.max_translation_speed_m_s * dt, 1.0e-6)
            distance = float(np.linalg.norm(delta))
            if distance > max_distance:
                delta *= max_distance / distance
            state.target_position = state.target_position + delta

            if self.config.mapping.orientation_enabled:
                current_controller_rotation = quat_xyzw_to_matrix(np.asarray(sample.orientation_xyzw))
                delta_xr = current_controller_rotation @ state.controller_reference_rotation.T
                delta_base = basis @ delta_xr @ basis.T
                requested_rotation = delta_base @ state.tcp_reference_rotation
                error_vector = _rotation_vector(requested_rotation @ state.target_rotation.T)
                max_angle = max(self.config.safety.max_rotation_speed_rad_s * dt, 1.0e-6)
                angle = float(np.linalg.norm(error_vector))
                if angle > max_angle:
                    error_vector *= max_angle / angle
                state.target_rotation = _rotation_from_vector(error_vector) @ state.target_rotation

        opened, closed = self.hand_profiles[side]
        hand6 = opened + state.trigger_filtered * (closed - opened)
        return SideMappingResult(
            target=TcpPose(state.target_position.copy(), matrix_to_quat_wxyz(state.target_rotation)),
            hand6=hand6.astype(np.float32),
            clutch=state.clutch,
            clutch_rising=clutch_rising,
            clutch_falling=clutch_falling,
            tracking_valid=tracking_valid,
        )

    def update(self, frame: ControllerFrame | None, current_left: TcpPose, current_right: TcpPose, dt: float, now_monotonic: float) -> BimanualMappingResult:
        age = math.inf if frame is None else max(float(now_monotonic - frame.received_monotonic), 0.0)
        stale = frame is None or age > self.config.network.stale_timeout_s
        left = self._update_side("left", None if frame is None else frame.left, current_left, dt, stale)
        right = self._update_side("right", None if frame is None else frame.right, current_right, dt, stale)
        return BimanualMappingResult(left=left, right=right, stale=stale, frame_age_s=age)
