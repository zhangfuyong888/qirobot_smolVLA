from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import ContractError


PROTOCOL_VERSION = 4


@dataclass(frozen=True)
class ObservationRequest:
    contract_sha256: str
    session_id: str
    request_id: int
    robot_timestamp_ns: int
    task: str
    state: np.ndarray
    image_timestamps_ns: tuple[int, int]
    rtc_inference_delay_steps: int = 0
    previous_accepted_request_id: int = -1
    rtc_reset_history: bool = False
    execution_lag_rad: float = 0.0


@dataclass(frozen=True)
class ActionResponse:
    contract_sha256: str
    session_id: str
    request_id: int
    inference_ms: float
    policy_fps: int
    action_chunk: np.ndarray
    rtc_enabled: bool = False
    rtc_inference_delay_steps: int = 0
    rtc_execution_horizon: int = 0
    rtc_prev_leftover_steps: int = 0
    raw_chunk_length: int = 0
    checkpoint: str = ""
    rtc_source_request_id: int = -1
    rtc_prev_raw_remaining_steps: int = 0
    rtc_elapsed_policy_position: float = 0.0
    rtc_leftover_start_index: int = 0
    rtc_history_reset: bool = False


def encode_metadata(payload: dict[str, Any]) -> bytes:
    import msgpack

    return msgpack.packb(payload, use_bin_type=True)


def decode_metadata(payload: bytes) -> dict[str, Any]:
    import msgpack

    value = msgpack.unpackb(payload, raw=False, strict_map_key=False)
    if not isinstance(value, dict):
        raise ContractError("wire metadata must be a mapping")
    if int(value.get("protocol_version", -1)) != PROTOCOL_VERSION:
        raise ContractError(f"protocol version mismatch: {value.get('protocol_version')}")
    return value


def pack_observation(request: ObservationRequest, head_jpeg: bytes, wrist_jpeg: bytes) -> list[bytes]:
    state = np.asarray(request.state, dtype="<f4")
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ContractError("wire state must be finite float32[8]")
    if not np.isfinite(request.execution_lag_rad) or request.execution_lag_rad < 0:
        raise ContractError("wire execution lag must be finite and non-negative")
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "type": "observation",
        "contract_sha256": request.contract_sha256,
        "session_id": request.session_id,
        "request_id": int(request.request_id),
        "robot_timestamp_ns": int(request.robot_timestamp_ns),
        "image_timestamps_ns": list(request.image_timestamps_ns),
        "task": request.task,
        "state_dtype": "float32",
        "state_shape": [8],
        "image_transport": "jpeg",
        "rtc_inference_delay_steps": int(request.rtc_inference_delay_steps),
        "previous_accepted_request_id": int(request.previous_accepted_request_id),
        "rtc_reset_history": bool(request.rtc_reset_history),
        "execution_lag_rad": float(request.execution_lag_rad),
    }
    return [encode_metadata(metadata), state.tobytes(), bytes(head_jpeg), bytes(wrist_jpeg)]


def unpack_observation(parts: list[bytes]) -> tuple[ObservationRequest, bytes, bytes]:
    if len(parts) != 4:
        raise ContractError(f"observation must contain four multipart frames, got {len(parts)}")
    meta = decode_metadata(parts[0])
    if meta.get("type") != "observation":
        raise ContractError("expected observation payload")
    state = np.frombuffer(parts[1], dtype="<f4").copy()
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ContractError("invalid observation state bytes")
    timestamps = tuple(int(v) for v in meta["image_timestamps_ns"])
    if len(timestamps) != 2:
        raise ContractError("exactly two image timestamps are required")
    execution_lag_rad = float(meta.get("execution_lag_rad", 0.0))
    if not np.isfinite(execution_lag_rad) or execution_lag_rad < 0:
        raise ContractError("invalid wire execution lag")
    return (
        ObservationRequest(
            str(meta["contract_sha256"]),
            str(meta["session_id"]),
            int(meta["request_id"]),
            int(meta["robot_timestamp_ns"]),
            str(meta["task"]),
            state,
            timestamps,
            int(meta.get("rtc_inference_delay_steps", 0)),
            int(meta.get("previous_accepted_request_id", -1)),
            bool(meta.get("rtc_reset_history", False)),
            execution_lag_rad,
        ),
        parts[2],
        parts[3],
    )


def pack_action_response(response: ActionResponse) -> list[bytes]:
    chunk = np.asarray(response.action_chunk, dtype="<f4")
    if chunk.ndim != 2 or chunk.shape[1] != 8 or not np.isfinite(chunk).all():
        raise ContractError(f"action chunk must be finite [N,8], got {chunk.shape}")
    meta = {
        "protocol_version": PROTOCOL_VERSION,
        "type": "action_chunk",
        "contract_sha256": response.contract_sha256,
        "session_id": response.session_id,
        "request_id": int(response.request_id),
        "inference_ms": float(response.inference_ms),
        "policy_fps": int(response.policy_fps),
        "action_shape": list(chunk.shape),
        "action_dtype": "float32",
        "rtc_enabled": bool(response.rtc_enabled),
        "rtc_inference_delay_steps": int(response.rtc_inference_delay_steps),
        "rtc_execution_horizon": int(response.rtc_execution_horizon),
        "rtc_prev_leftover_steps": int(response.rtc_prev_leftover_steps),
        "raw_chunk_length": int(response.raw_chunk_length or chunk.shape[0]),
        "checkpoint": str(response.checkpoint),
        "rtc_source_request_id": int(response.rtc_source_request_id),
        "rtc_prev_raw_remaining_steps": int(response.rtc_prev_raw_remaining_steps),
        "rtc_elapsed_policy_position": float(response.rtc_elapsed_policy_position),
        "rtc_leftover_start_index": int(response.rtc_leftover_start_index),
        "rtc_history_reset": bool(response.rtc_history_reset),
    }
    return [encode_metadata(meta), chunk.tobytes()]


def unpack_action_response(parts: list[bytes]) -> ActionResponse:
    if parts:
        metadata = decode_metadata(parts[0])
        if metadata.get("type") == "error":
            raise RuntimeError(f"policy server error: {metadata.get('error', 'unknown error')}")
    if len(parts) != 2:
        raise ContractError(f"action response must contain two frames, got {len(parts)}")
    meta = metadata
    shape = tuple(int(v) for v in meta["action_shape"])
    chunk = np.frombuffer(parts[1], dtype="<f4").reshape(shape).copy()
    if chunk.ndim != 2 or chunk.shape[1] != 8 or not np.isfinite(chunk).all():
        raise ContractError("invalid action chunk bytes")
    return ActionResponse(
        str(meta["contract_sha256"]),
        str(meta["session_id"]),
        int(meta["request_id"]),
        float(meta["inference_ms"]),
        int(meta["policy_fps"]),
        chunk,
        bool(meta.get("rtc_enabled", False)),
        int(meta.get("rtc_inference_delay_steps", 0)),
        int(meta.get("rtc_execution_horizon", 0)),
        int(meta.get("rtc_prev_leftover_steps", 0)),
        int(meta.get("raw_chunk_length", shape[0])),
        str(meta.get("checkpoint", "")),
        int(meta.get("rtc_source_request_id", -1)),
        int(meta.get("rtc_prev_raw_remaining_steps", 0)),
        float(meta.get("rtc_elapsed_policy_position", 0.0)),
        int(meta.get("rtc_leftover_start_index", 0)),
        bool(meta.get("rtc_history_reset", False)),
    )
