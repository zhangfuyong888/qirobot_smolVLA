"""Versioned, coherent controller-frame protocol used by the Quest WebXR page."""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from typing import Any


PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class ControllerSample:
    valid: bool
    position: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    trigger: float
    squeeze: float
    buttons: tuple[float, ...] = ()
    axes: tuple[float, ...] = ()
    profiles: tuple[str, ...] = ()


@dataclass(frozen=True)
class ControllerFrame:
    session_id: str
    sequence: int
    client_time_ms: float
    reference_space: str
    left: ControllerSample
    right: ControllerSample
    received_monotonic: float
    calibration_id: int = 0
    calibration_viewer_orientation_xyzw: tuple[float, float, float, float] | None = None
    boundary_safe: bool = True
    boundary_distance_m: float | None = None


def _finite_vector(value: Any, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} contains NaN or Inf")
    return result


def _unit_interval(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return min(max(result, 0.0), 1.0)


def _parse_side(payload: Any, side: str) -> ControllerSample:
    if not isinstance(payload, dict):
        raise ValueError(f"{side} must be an object")
    valid = bool(payload.get("valid", False))
    position = _finite_vector(payload.get("position", [0.0, 0.0, 0.0]), 3, f"{side}.position")
    orientation = _finite_vector(
        payload.get("orientation_xyzw", [0.0, 0.0, 0.0, 1.0]),
        4,
        f"{side}.orientation_xyzw",
    )
    norm = math.sqrt(sum(item * item for item in orientation))
    if valid and norm < 1.0e-6:
        raise ValueError(f"{side}.orientation_xyzw has zero norm")
    if norm >= 1.0e-6:
        orientation = tuple(item / norm for item in orientation)
    buttons_raw = payload.get("buttons", [])
    axes_raw = payload.get("axes", [])
    profiles_raw = payload.get("profiles", [])
    if not isinstance(buttons_raw, list) or not isinstance(axes_raw, list):
        raise ValueError(f"{side}.buttons and {side}.axes must be arrays")
    buttons = tuple(_unit_interval(item, f"{side}.buttons") for item in buttons_raw[:16])
    axes = tuple(float(item) for item in axes_raw[:8])
    if not all(math.isfinite(item) for item in axes):
        raise ValueError(f"{side}.axes contains NaN or Inf")
    profiles = tuple(str(item)[:80] for item in profiles_raw[:8]) if isinstance(profiles_raw, list) else ()
    return ControllerSample(
        valid=valid,
        position=position,
        orientation_xyzw=orientation,
        trigger=_unit_interval(payload.get("trigger", 0.0), f"{side}.trigger"),
        squeeze=_unit_interval(payload.get("squeeze", 0.0), f"{side}.squeeze"),
        buttons=buttons,
        axes=axes,
        profiles=profiles,
    )


def parse_controller_frame(payload: str | bytes | dict[str, Any], received_monotonic: float | None = None) -> ControllerFrame:
    """Validate one complete WebXR sample before publishing it to the control loop."""
    if isinstance(payload, (str, bytes)):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise ValueError("controller frame must be a JSON object")
    if payload.get("type") != "controller_frame":
        raise ValueError("unsupported message type")
    if int(payload.get("version", -1)) != PROTOCOL_VERSION:
        raise ValueError(f"unsupported protocol version: {payload.get('version')!r}")
    session_id = str(payload.get("session_id", ""))[:128]
    if not session_id:
        raise ValueError("session_id is required")
    sequence = int(payload.get("sequence", -1))
    if sequence < 0:
        raise ValueError("sequence must be non-negative")
    client_time_ms = float(payload.get("client_time_ms", 0.0))
    if not math.isfinite(client_time_ms):
        raise ValueError("client_time_ms must be finite")
    calibration_id = int(payload.get("calibration_id", 0))
    if calibration_id < 0:
        raise ValueError("calibration_id must be non-negative")
    calibration_orientation_raw = payload.get("calibration_viewer_orientation_xyzw")
    calibration_orientation = None
    if calibration_orientation_raw is not None:
        calibration_orientation = _finite_vector(
            calibration_orientation_raw,
            4,
            "calibration_viewer_orientation_xyzw",
        )
        norm = math.sqrt(sum(item * item for item in calibration_orientation))
        if norm < 1.0e-6:
            raise ValueError("calibration_viewer_orientation_xyzw has zero norm")
        calibration_orientation = tuple(item / norm for item in calibration_orientation)
    if (calibration_id > 0) != (calibration_orientation is not None):
        raise ValueError(
            "calibration_id and calibration_viewer_orientation_xyzw must be provided together"
        )
    boundary_distance_raw = payload.get("boundary_distance_m")
    boundary_distance_m = None
    if boundary_distance_raw is not None:
        boundary_distance_m = float(boundary_distance_raw)
        if not math.isfinite(boundary_distance_m):
            raise ValueError("boundary_distance_m must be finite")
    boundary_safe = payload.get("boundary_safe", True)
    if not isinstance(boundary_safe, bool):
        raise ValueError("boundary_safe must be a boolean")
    return ControllerFrame(
        session_id=session_id,
        sequence=sequence,
        client_time_ms=client_time_ms,
        reference_space=str(payload.get("reference_space", "unknown"))[:64],
        left=_parse_side(payload.get("left", {}), "left"),
        right=_parse_side(payload.get("right", {}), "right"),
        received_monotonic=time.monotonic() if received_monotonic is None else float(received_monotonic),
        calibration_id=calibration_id,
        calibration_viewer_orientation_xyzw=calibration_orientation,
        boundary_safe=boundary_safe,
        boundary_distance_m=boundary_distance_m,
    )


class LatestFrameStore:
    """Thread-safe latest-frame buffer; old or duplicated sequence numbers are rejected."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: ControllerFrame | None = None
        self._clients = 0
        self._accepted = 0
        self._rejected = 0

    def publish(self, frame: ControllerFrame) -> bool:
        with self._lock:
            if (
                self._frame is not None
                and frame.session_id == self._frame.session_id
                and frame.sequence <= self._frame.sequence
            ):
                self._rejected += 1
                return False
            self._frame = frame
            self._accepted += 1
            return True

    def snapshot(self) -> ControllerFrame | None:
        with self._lock:
            return self._frame

    def client_connected(self, max_clients: int = 1) -> bool:
        with self._lock:
            if self._clients >= max(int(max_clients), 1):
                return False
            self._clients += 1
            return True

    def client_disconnected(self) -> None:
        with self._lock:
            self._clients = max(self._clients - 1, 0)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "clients": self._clients,
                "accepted": self._accepted,
                "rejected": self._rejected,
            }
