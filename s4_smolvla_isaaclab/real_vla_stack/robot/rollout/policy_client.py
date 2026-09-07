from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable

from ...common.protocol import (
    ActionResponse,
    ObservationRequest,
    pack_observation,
    unpack_action_response,
)


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


@dataclass(frozen=True)
class PolicyResult:
    observation: ObservationRequest
    response: ActionResponse | None
    rtt_ms: float | None
    received_at_ns: int
    error: BaseException | None


class AsyncPolicyClient:
    """One persistent worker owns the ZeroMQ socket for its entire lifetime."""

    def __init__(
        self,
        endpoint: str,
        timeout_ms: int,
        *,
        client_factory: Callable[[], PolicyClient] | None = None,
    ) -> None:
        self._requests: queue.Queue[tuple[ObservationRequest, bytes, bytes] | None] = queue.Queue(maxsize=1)
        self._results: queue.Queue[PolicyResult] = queue.Queue()
        self._busy = threading.Event()
        self._closed = False
        self._client_factory = client_factory or (lambda: PolicyClient(endpoint, timeout_ms))
        self._thread = threading.Thread(target=self._run, name="policy-network", daemon=True)
        self._thread.start()

    @property
    def busy(self) -> bool:
        return self._busy.is_set()

    def submit(self, observation: ObservationRequest, head_jpeg: bytes, wrist_jpeg: bytes) -> bool:
        if self._closed:
            raise RuntimeError("policy client is closed")
        if self._busy.is_set() or not self._results.empty():
            return False
        self._busy.set()
        try:
            self._requests.put_nowait((observation, bytes(head_jpeg), bytes(wrist_jpeg)))
        except queue.Full:
            self._busy.clear()
            return False
        return True

    def poll(self) -> PolicyResult | None:
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None

    def _run(self) -> None:
        client: PolicyClient | None = None
        try:
            while True:
                item = self._requests.get()
                if item is None:
                    return
                observation, head_jpeg, wrist_jpeg = item
                response = None
                rtt_ms = None
                error = None
                try:
                    if client is None:
                        client = self._client_factory()
                    response, rtt_ms = client.request(observation, head_jpeg, wrist_jpeg)
                except BaseException as exc:
                    error = exc
                received_at_ns = time.monotonic_ns()
                self._results.put(
                    PolicyResult(observation, response, rtt_ms, received_at_ns, error)
                )
                self._busy.clear()
        finally:
            if client is not None:
                client.close()
            self._busy.clear()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._requests.put(None)
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise RuntimeError("policy network thread did not stop")
