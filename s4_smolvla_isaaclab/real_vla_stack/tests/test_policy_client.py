from __future__ import annotations

import threading
import time

import numpy as np

from real_vla_stack.common.protocol import ActionResponse, ObservationRequest
from real_vla_stack.robot.rollout.policy_client import AsyncPolicyClient


class _FakeClient:
    def __init__(self) -> None:
        self.request_thread = 0
        self.close_thread = 0

    def request(self, observation, _head, _wrist):
        self.request_thread = threading.get_ident()
        return (
            ActionResponse(
                observation.contract_sha256,
                observation.session_id,
                observation.request_id,
                1.0,
                20,
                np.zeros((50, 8), dtype=np.float32),
            ),
            2.0,
        )

    def close(self) -> None:
        self.close_thread = threading.get_ident()


def test_async_client_owns_request_and_close_on_one_worker_thread() -> None:
    fake = _FakeClient()
    client = AsyncPolicyClient("unused", 50, client_factory=lambda: fake)
    observation = ObservationRequest(
        "a" * 64,
        "session",
        1,
        time.monotonic_ns(),
        "task",
        np.zeros(8, dtype=np.float32),
        (1, 2),
    )
    try:
        assert client.submit(observation, b"head", b"wrist")
        deadline = time.monotonic() + 1.0
        result = None
        while result is None and time.monotonic() < deadline:
            result = client.poll()
            time.sleep(0.001)
        assert result is not None
        assert result.error is None
        assert result.response is not None
        assert fake.request_thread != threading.get_ident()
    finally:
        client.close()
    assert fake.close_thread == fake.request_thread
