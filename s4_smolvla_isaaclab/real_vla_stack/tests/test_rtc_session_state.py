from __future__ import annotations

import pytest

from real_vla_stack.common.errors import ContractError
from real_vla_stack.host.inference.policy_runner import RTCSessionState


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
