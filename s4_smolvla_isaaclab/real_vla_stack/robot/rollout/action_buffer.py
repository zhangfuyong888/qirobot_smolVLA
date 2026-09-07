from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...common.errors import ContractError, PolicyStaleError


@dataclass
class ActionBuffer:
    policy_hz: float
    execute_horizon: int
    max_chunk_age_ms: float
    blend_duration_ms: float = 0.0
    chunk: np.ndarray | None = None
    received_at_ns: int = 0
    source_at_ns: int = 0
    request_id: int = -1
    blend_from_q7: np.ndarray | None = None

    def age_ms(self, now_ns: int) -> float:
        if self.chunk is None:
            raise PolicyStaleError("no policy action chunk received")
        return (int(now_ns) - self.source_at_ns) / 1.0e6

    def replace(
        self,
        chunk: np.ndarray,
        *,
        request_id: int,
        received_at_ns: int,
        source_at_ns: int | None = None,
        transition_from_q7: np.ndarray | None = None,
    ) -> None:
        value = np.asarray(chunk, dtype=np.float32)
        if value.ndim != 2 or value.shape[1] != 8 or not np.isfinite(value).all():
            raise ContractError(f"chunk must be finite [N,8], got {value.shape}")
        if value.shape[0] < self.execute_horizon:
            raise ContractError("chunk is shorter than execute_horizon")
        if request_id <= self.request_id:
            raise ContractError(f"stale/out-of-order response request_id={request_id}, latest={self.request_id}")
        source_ns = int(received_at_ns if source_at_ns is None else source_at_ns)
        if source_ns < 0 or source_ns > int(received_at_ns):
            raise ContractError("policy chunk source timestamp is invalid")
        self.chunk = value[: self.execute_horizon].copy()
        self.received_at_ns = int(received_at_ns)
        self.source_at_ns = source_ns
        self.request_id = int(request_id)
        self.blend_from_q7 = (
            None
            if transition_from_q7 is None or self.blend_duration_ms <= 0.0
            else np.asarray(transition_from_q7, dtype=np.float32).reshape(7).copy()
        )

    def sample(self, now_ns: int) -> np.ndarray:
        if self.chunk is None:
            raise PolicyStaleError("no policy action chunk received")
        age_ms = self.age_ms(now_ns)
        if age_ms < 0 or age_ms > self.max_chunk_age_ms:
            raise PolicyStaleError(f"policy chunk age {age_ms:.1f}ms exceeds {self.max_chunk_age_ms:.1f}ms")
        output = sample_policy_chunk(
            self.chunk,
            policy_hz=self.policy_hz,
            source_at_ns=self.source_at_ns,
            now_ns=now_ns,
        )
        if self.blend_from_q7 is not None:
            blend_elapsed_ms = (int(now_ns) - self.received_at_ns) / 1.0e6
            ratio = float(np.clip(blend_elapsed_ms / self.blend_duration_ms, 0.0, 1.0))
            # Smoothstep has zero slope at both ends, avoiding a velocity impulse
            # when a newly inferred chunk replaces the previous plan.
            alpha = ratio * ratio * (3.0 - 2.0 * ratio)
            output[:7] = (1.0 - alpha) * self.blend_from_q7 + alpha * output[:7]
            if ratio >= 1.0:
                self.blend_from_q7 = None
        return output


def sample_policy_chunk(
    chunk: np.ndarray,
    *,
    policy_hz: float,
    source_at_ns: int,
    now_ns: int,
) -> np.ndarray:
    """Sample an action chunk on its observation-time timeline."""
    value = np.asarray(chunk, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 8 or value.shape[0] == 0:
        raise ContractError(f"chunk must be non-empty [N,8], got {value.shape}")
    elapsed_s = max((int(now_ns) - int(source_at_ns)) / 1.0e9, 0.0)
    position = min(elapsed_s * float(policy_hz), len(value) - 1)
    low = int(np.floor(position))
    high = min(low + 1, len(value) - 1)
    alpha = float(position - low)
    output = value[low].copy()
    output[:7] = (1.0 - alpha) * value[low, :7] + alpha * value[high, :7]
    output[7] = value[low, 7]
    return output
