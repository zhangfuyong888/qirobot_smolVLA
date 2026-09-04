from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...common.errors import ContractError, PolicyStaleError


@dataclass
class ActionBuffer:
    policy_hz: float
    execute_horizon: int
    max_chunk_age_ms: float
    chunk: np.ndarray | None = None
    received_at_ns: int = 0
    request_id: int = -1

    def replace(self, chunk: np.ndarray, *, request_id: int, received_at_ns: int) -> None:
        value = np.asarray(chunk, dtype=np.float32)
        if value.ndim != 2 or value.shape[1] != 8 or not np.isfinite(value).all():
            raise ContractError(f"chunk must be finite [N,8], got {value.shape}")
        if value.shape[0] < self.execute_horizon:
            raise ContractError("chunk is shorter than execute_horizon")
        if request_id <= self.request_id:
            raise ContractError(f"stale/out-of-order response request_id={request_id}, latest={self.request_id}")
        self.chunk = value[: self.execute_horizon].copy()
        self.received_at_ns = int(received_at_ns)
        self.request_id = int(request_id)

    def sample(self, now_ns: int) -> np.ndarray:
        if self.chunk is None:
            raise PolicyStaleError("no policy action chunk received")
        age_ms = (int(now_ns) - self.received_at_ns) / 1.0e6
        if age_ms < 0 or age_ms > self.max_chunk_age_ms:
            raise PolicyStaleError(f"policy chunk age {age_ms:.1f}ms exceeds {self.max_chunk_age_ms:.1f}ms")
        elapsed_s = max((int(now_ns) - self.received_at_ns) / 1.0e9, 0.0)
        position = min(elapsed_s * self.policy_hz, len(self.chunk) - 1)
        low = int(np.floor(position))
        high = min(low + 1, len(self.chunk) - 1)
        alpha = float(position - low)
        output = self.chunk[low].copy()
        output[:7] = (1.0 - alpha) * self.chunk[low, :7] + alpha * self.chunk[high, :7]
        output[7] = self.chunk[low, 7]
        return output
