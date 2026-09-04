from __future__ import annotations

import numpy as np
import pytest

from real_vla_stack.common.protocol import (
    ActionResponse,
    ObservationRequest,
    pack_action_response,
    pack_observation,
    unpack_action_response,
    unpack_observation,
    encode_metadata,
)


def test_multipart_protocol_round_trip() -> None:
    request = ObservationRequest("a" * 64, "session", 7, 100, "task", np.arange(8), (80, 90))
    decoded, head, wrist = unpack_observation(pack_observation(request, b"head", b"wrist"))
    assert decoded.request_id == 7
    assert np.array_equal(decoded.state, np.arange(8, dtype=np.float32))
    assert (head, wrist) == (b"head", b"wrist")
    response = ActionResponse("a" * 64, "session", 7, 12.5, 20, np.zeros((50, 8)))
    actual = unpack_action_response(pack_action_response(response))
    assert actual.action_chunk.shape == (50, 8)
    assert actual.request_id == 7


def test_server_error_frame_is_explicit() -> None:
    with pytest.raises(RuntimeError, match="bad contract"):
        unpack_action_response(
            [encode_metadata({"protocol_version": 1, "type": "error", "error": "bad contract"})]
        )
