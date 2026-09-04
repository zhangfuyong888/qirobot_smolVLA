from __future__ import annotations

import time

from ...common.protocol import ObservationRequest, pack_observation, unpack_action_response


class PolicyClient:
    """Synchronous one-outstanding-request client; callers run it off the 30 Hz control thread."""

    def __init__(self, endpoint: str, timeout_ms: int) -> None:
        import zmq

        self._zmq = zmq
        self._context = zmq.Context.instance()
        self.endpoint = endpoint
        self.timeout_ms = int(timeout_ms)
        self._socket = self._new_socket()

    def _new_socket(self):
        socket = self._context.socket(self._zmq.REQ)
        socket.setsockopt(self._zmq.LINGER, 0)
        socket.setsockopt(self._zmq.RCVTIMEO, self.timeout_ms)
        socket.setsockopt(self._zmq.SNDTIMEO, self.timeout_ms)
        socket.connect(self.endpoint)
        return socket

    def request(self, observation: ObservationRequest, head_jpeg: bytes, wrist_jpeg: bytes):
        sent_ns = time.monotonic_ns()
        try:
            self._socket.send_multipart(pack_observation(observation, head_jpeg, wrist_jpeg))
            response = unpack_action_response(self._socket.recv_multipart())
        except self._zmq.error.Again as exc:
            self._socket.close(linger=0)
            self._socket = self._new_socket()
            raise TimeoutError(f"policy request timed out after {self.timeout_ms}ms") from exc
        if response.session_id != observation.session_id or response.request_id != observation.request_id:
            raise RuntimeError("policy response is stale or belongs to another rollout session")
        if response.contract_sha256 != observation.contract_sha256:
            raise RuntimeError("policy response contract hash mismatch")
        return response, (time.monotonic_ns() - sent_ns) / 1.0e6

    def close(self) -> None:
        self._socket.close(linger=0)
