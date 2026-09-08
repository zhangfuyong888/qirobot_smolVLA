from __future__ import annotations

import time

import numpy as np

from ...common.protocol import (
    PROTOCOL_VERSION,
    ActionResponse,
    encode_metadata,
    pack_action_response,
    unpack_observation,
)


class RequestSessionGuard:
    """Reset policy state between rollouts and reject replayed requests."""

    def __init__(self) -> None:
        self.session_id: str | None = None
        self.request_id = -1

    def accept(self, runner, request) -> None:
        if request.session_id != self.session_id:
            if request.request_id != 0:
                raise ValueError("a new rollout session must start with request_id=0")
            runner.reset()
            self.session_id = request.session_id
            self.request_id = -1
        if request.request_id <= self.request_id:
            raise ValueError("request is stale, duplicated, or out of order")
        self.request_id = request.request_id


def _decode_jpeg_rgb(payload: bytes) -> np.ndarray:
    import cv2

    bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode JPEG observation")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def serve_policy(runner, *, bind: str, port: int) -> None:
    import zmq

    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(f"tcp://{bind}:{int(port)}")
    sessions = RequestSessionGuard()
    print(f"[REAL-VLA-SERVER] ready tcp://{bind}:{port} contract={runner.contract.sha256}", flush=True)
    try:
        while True:
            parts = socket.recv_multipart()
            started = time.monotonic_ns()
            try:
                request, head_jpeg, wrist_jpeg = unpack_observation(parts)
                if request.contract_sha256 != runner.contract.sha256:
                    raise ValueError("request contract hash mismatch")
                if request.task != runner.contract.task:
                    raise ValueError("request task does not match the deployed contract")
                sessions.accept(runner, request)
                images = {
                    runner.contract.camera_keys[0]: _decode_jpeg_rgb(head_jpeg),
                    runner.contract.camera_keys[1]: _decode_jpeg_rgb(wrist_jpeg),
                }
                chunk = runner.predict_chunk(
                    request.state,
                    images,
                    request.task,
                    observation_timestamp_ns=request.robot_timestamp_ns,
                    inference_delay_steps=request.rtc_inference_delay_steps,
                    request_id=request.request_id,
                    previous_accepted_request_id=request.previous_accepted_request_id,
                )
                inference_ms = (time.monotonic_ns() - started) / 1.0e6
                diagnostics = runner.last_diagnostics
                socket.send_multipart(
                    pack_action_response(
                        ActionResponse(
                            runner.contract.sha256,
                            request.session_id,
                            request.request_id,
                            inference_ms,
                            runner.contract.dataset_fps,
                            chunk,
                            bool(diagnostics.get("rtc_enabled", False)),
                            int(diagnostics.get("rtc_inference_delay_steps", 0)),
                            int(diagnostics.get("rtc_execution_horizon", 0)),
                            int(diagnostics.get("rtc_prev_leftover_steps", 0)),
                            int(diagnostics.get("raw_chunk_length", len(chunk))),
                            str(runner.checkpoint),
                            int(diagnostics.get("rtc_source_request_id", -1)),
                            int(diagnostics.get("rtc_prev_raw_remaining_steps", 0)),
                            float(diagnostics.get("rtc_elapsed_policy_position", 0.0)),
                            int(diagnostics.get("rtc_leftover_start_index", 0)),
                        )
                    )
                )
            except Exception as exc:
                socket.send_multipart(
                    [
                        encode_metadata(
                            {
                                "protocol_version": PROTOCOL_VERSION,
                                "type": "error",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                    ]
                )
    finally:
        socket.close(linger=0)
