from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import ContractError


PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class ObservationRequest:
    contract_sha256: str
    session_id: str
    request_id: int
    robot_timestamp_ns: int
    task: str
    state: np.ndarray
    image_timestamps_ns: tuple[int, int]


@dataclass(frozen=True)
class ActionResponse:
    contract_sha256: str
    session_id: str
    request_id: int
    inference_ms: float
    policy_fps: int
    action_chunk: np.ndarray


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
    return (
        ObservationRequest(
            str(meta["contract_sha256"]),
            str(meta["session_id"]),
            int(meta["request_id"]),
            int(meta["robot_timestamp_ns"]),
            str(meta["task"]),
            state,
            timestamps,
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
    )
