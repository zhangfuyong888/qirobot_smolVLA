from __future__ import annotations

import pytest

from real_vla_stack.common.errors import ContractError
import torch

from real_vla_stack.host.inference.policy_runner import (
    PolicyRunner,
    RTCSessionState,
    rtc_leftover_for_observation,
)


def test_rtc_session_promotes_only_acknowledged_raw_chunk() -> None:
    state = RTCSessionState()
    state.generated[3] = (100, "accepted-model-space-chunk")
    state.generated[4] = (200, "rejected-model-space-chunk")
    state.acknowledge(3)
    assert state.accepted_raw_chunk == "accepted-model-space-chunk"
    assert state.accepted_observation_ns == 100
    state.acknowledge(3)
    assert state.accepted_raw_chunk == "accepted-model-space-chunk"


def test_rtc_session_rejects_unknown_acknowledgement() -> None:
    state = RTCSessionState()
    with pytest.raises(ContractError, match="unknown"):
        state.acknowledge(9)


def test_runner_can_reset_only_rtc_history() -> None:
    runner = object.__new__(PolicyRunner)
    runner.rtc_state = RTCSessionState()
    runner.rtc_state.generated[4] = (100, "chunk")
    runner.reset_rtc_history()
    assert runner.rtc_state.accepted_raw_chunk is None
    assert runner.rtc_state.generated == {}


def test_rtc_leftover_is_real_length_without_zero_padding() -> None:
    chunk = torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4)
    prefix, position, start, remaining = rtc_leftover_for_observation(
        chunk,
        elapsed_ns=225_000_000,
        policy_fps=20,
        execution_horizon=10,
    )
    assert position == pytest.approx(4.5)
    assert start == 5
    assert remaining == 1
    assert prefix is not None
    assert prefix.shape == (1, 4)
    assert torch.equal(prefix, chunk[5:])


def test_rtc_leftover_uses_exact_next_unexecuted_index() -> None:
    chunk = torch.zeros((50, 32))
    prefix, position, start, remaining = rtc_leftover_for_observation(
        chunk,
        elapsed_ns=500_000_000,
        policy_fps=20,
        execution_horizon=10,
    )
    assert position == 10.0
    assert start == 10
    assert remaining == 40
    assert prefix is not None and prefix.shape == (10, 32)
