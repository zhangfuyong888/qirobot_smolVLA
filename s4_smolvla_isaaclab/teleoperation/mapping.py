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
    xr_delta: np.ndarray
    base_delta: np.ndarray


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
    filtered_controller_position: np.ndarray | None = None
    filtered_controller_rotation: np.ndarray | None = None
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


def _exp_smooth_vector(previous: np.ndarray, current: np.ndarray, dt: float, tau: float) -> np.ndarray:
    if tau <= 0.0:
        return np.asarray(current, dtype=np.float64)
    alpha = 1.0 - math.exp(-max(float(dt), 0.0) / tau)
    previous_np = np.asarray(previous, dtype=np.float64)
    current_np = np.asarray(current, dtype=np.float64)
    return previous_np + alpha * (current_np - previous_np)


def resolved_translation_sign(
    invert_translation: bool,
    translation_sign: tuple[float, float, float] | np.ndarray = (1.0, 1.0, 1.0),
) -> np.ndarray:
    sign = np.asarray(translation_sign, dtype=np.float64)
    if invert_translation:
        sign = -sign
    return sign


def mapping_basis(basis: np.ndarray, translation_sign: np.ndarray) -> np.ndarray:
    """Apply per-axis signs in the robot base frame after the calibrated rotation."""
    return np.diag(np.asarray(translation_sign, dtype=np.float64)) @ np.asarray(basis, dtype=np.float64)


def map_xr_translation(
    basis: np.ndarray,
    xr_delta: np.ndarray,
    position_scale: float,
    invert_translation: bool = False,
    translation_sign: tuple[float, float, float] | np.ndarray = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Map a WebXR position delta into the robot base frame."""
    sign = resolved_translation_sign(invert_translation, translation_sign)
    mapped = mapping_basis(basis, sign) @ np.asarray(xr_delta, dtype=np.float64)
    return float(position_scale) * mapped


def map_xr_rotation(
    basis: np.ndarray,
    delta_xr: np.ndarray,
    invert_orientation: bool,
    translation_sign: tuple[float, float, float] | np.ndarray = (1.0, 1.0, 1.0),
    invert_translation: bool = False,
) -> np.ndarray:
    """Map a WebXR rotation delta into the robot base frame.

    A proper rotation (det=+1) can bake XY translation signs into the same
    basis used for orientation, which flips roll/pitch sense while keeping
    yaw about vertical. A full XYZ flip has det=-1, so that case keeps the
    original basis and uses ``delta_xr.T`` only when invert_orientation is set.
    """
    sign = resolved_translation_sign(invert_translation, translation_sign)
    basis_np = np.asarray(basis, dtype=np.float64)
    if float(np.linalg.det(np.diag(sign))) > 0.0:
        basis_np = mapping_basis(basis_np, sign)
    delta = np.asarray(delta_xr, dtype=np.float64)
    if invert_orientation:
        delta = delta.T
    return basis_np @ delta @ basis_np.T


def _exp_smooth_rotation(previous: np.ndarray, current: np.ndarray, dt: float, tau: float) -> np.ndarray:
    if tau <= 0.0:
        return np.asarray(current, dtype=np.float64)
    alpha = 1.0 - math.exp(-max(float(dt), 0.0) / tau)
    error_vector = _rotation_vector(np.asarray(current, dtype=np.float64) @ np.asarray(previous, dtype=np.float64).T)
    return _rotation_from_vector(alpha * error_vector) @ previous


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
            raw_position = np.asarray(sample.position, dtype=np.float64)
            raw_rotation = quat_xyzw_to_matrix(np.asarray(sample.orientation_xyzw))
            state.filtered_controller_position = raw_position.copy()
            state.filtered_controller_rotation = raw_rotation.copy()
            state.controller_reference_position = raw_position.copy()
            state.controller_reference_rotation = raw_rotation.copy()
            state.tcp_reference_position = np.asarray(current_tcp.position, dtype=np.float64).copy()
            state.tcp_reference_rotation = quat_wxyz_to_matrix(current_tcp.quat_wxyz)
            state.target_position = state.tcp_reference_position.copy()
            state.target_rotation = state.tcp_reference_rotation.copy()

        if clutch_falling:
            state.target_position = np.asarray(current_tcp.position, dtype=np.float64).copy()
            state.target_rotation = quat_wxyz_to_matrix(current_tcp.quat_wxyz)

        xr_delta = np.zeros(3)
        base_delta = np.zeros(3)
        if state.clutch and sample is not None and state.controller_reference_position is not None:
            tau = float(self.config.mapping.controller_filter_time_constant_s)
            raw_position = np.asarray(sample.position, dtype=np.float64)
            raw_rotation = quat_xyzw_to_matrix(np.asarray(sample.orientation_xyzw))
            if state.filtered_controller_position is None:
                state.filtered_controller_position = raw_position.copy()
                state.filtered_controller_rotation = raw_rotation.copy()
            else:
                state.filtered_controller_position = _exp_smooth_vector(
                    state.filtered_controller_position, raw_position, dt, tau
                )
                state.filtered_controller_rotation = _exp_smooth_rotation(
                    state.filtered_controller_rotation, raw_rotation, dt, tau
                )
            basis = self.config.mapping.controller_to_base_rotation
            controller_position = state.filtered_controller_position
            xr_delta = controller_position - state.controller_reference_position
            base_delta = map_xr_translation(
                basis,
                xr_delta,
                self.config.mapping.position_scale,
                self.config.mapping.invert_translation,
                self.config.mapping.translation_sign,
            )
            requested_position = state.tcp_reference_position + base_delta
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

            if (
                self.config.mapping.orientation_enabled
                and state.filtered_controller_rotation is not None
                and state.controller_reference_rotation is not None
            ):
                current_controller_rotation = state.filtered_controller_rotation
                delta_xr = current_controller_rotation @ state.controller_reference_rotation.T
                delta_base = map_xr_rotation(
                    basis,
                    delta_xr,
                    self.config.mapping.invert_orientation,
                    self.config.mapping.translation_sign,
                    self.config.mapping.invert_translation,
                )
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
            xr_delta=xr_delta,
            base_delta=base_delta,
        )

    def update(self, frame: ControllerFrame | None, current_left: TcpPose, current_right: TcpPose, dt: float, now_monotonic: float) -> BimanualMappingResult:
        age = math.inf if frame is None else max(float(now_monotonic - frame.received_monotonic), 0.0)
        stale = frame is None or age > self.config.network.stale_timeout_s
        left = self._update_side("left", None if frame is None else frame.left, current_left, dt, stale)
        right = self._update_side("right", None if frame is None else frame.right, current_right, dt, stale)
        return BimanualMappingResult(left=left, right=right, stale=stale, frame_age_s=age)
