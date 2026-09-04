from __future__ import annotations

import time

import numpy as np

from ...common.protocol import ActionResponse, pack_action_response, unpack_observation


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
    print(f"[REAL-VLA-SERVER] ready tcp://{bind}:{port} contract={runner.contract.sha256}", flush=True)
    try:
        while True:
            parts = socket.recv_multipart()
            started = time.monotonic_ns()
            request, head_jpeg, wrist_jpeg = unpack_observation(parts)
            if request.contract_sha256 != runner.contract.sha256:
                raise ValueError("request contract hash mismatch")
            images = {
                runner.contract.camera_keys[0]: _decode_jpeg_rgb(head_jpeg),
                runner.contract.camera_keys[1]: _decode_jpeg_rgb(wrist_jpeg),
            }
            chunk = runner.predict_chunk(request.state, images, request.task)
            inference_ms = (time.monotonic_ns() - started) / 1.0e6
            socket.send_multipart(
                pack_action_response(
                    ActionResponse(
                        runner.contract.sha256,
                        request.session_id,
                        request.request_id,
                        inference_ms,
                        runner.contract.dataset_fps,
                        chunk,
                    )
                )
            )
    finally:
        socket.close(linger=0)
