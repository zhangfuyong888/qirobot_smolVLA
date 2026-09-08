from __future__ import annotations

import numpy as np
import pytest

from real_vla_stack.common.protocol import (
    PROTOCOL_VERSION,
    ActionResponse,
    ObservationRequest,
    pack_action_response,
    pack_observation,
    unpack_action_response,
    unpack_observation,
    encode_metadata,
)


def test_multipart_protocol_round_trip() -> None:
    request = ObservationRequest(
        "a" * 64,
        "session",
        7,
        100,
        "task",
        np.arange(8),
        (80, 90),
        4,
        6,
        True,
        0.125,
    )
    decoded, head, wrist = unpack_observation(pack_observation(request, b"head", b"wrist"))
    assert decoded.request_id == 7
    assert np.array_equal(decoded.state, np.arange(8, dtype=np.float32))
    assert decoded.rtc_inference_delay_steps == 4
    assert decoded.previous_accepted_request_id == 6
    assert decoded.rtc_reset_history
    assert decoded.execution_lag_rad == pytest.approx(0.125)
    assert (head, wrist) == (b"head", b"wrist")
    response = ActionResponse(
        "a" * 64,
        "session",
        7,
        12.5,
        20,
        np.zeros((50, 8)),
        True,
        4,
        10,
        16,
        50,
        "/models/checkpoint",
        6,
        16,
        9.5,
        10,
        True,
    )
    actual = unpack_action_response(pack_action_response(response))
    assert actual.action_chunk.shape == (50, 8)
    assert actual.request_id == 7
    assert actual.rtc_enabled
    assert actual.rtc_prev_leftover_steps == 16
    assert actual.checkpoint == "/models/checkpoint"
    assert actual.rtc_source_request_id == 6
    assert actual.rtc_prev_raw_remaining_steps == 16
    assert actual.rtc_elapsed_policy_position == 9.5
    assert actual.rtc_leftover_start_index == 10
    assert actual.rtc_history_reset


def test_server_error_frame_is_explicit() -> None:
    with pytest.raises(RuntimeError, match="bad contract"):
        unpack_action_response(
            [encode_metadata({"protocol_version": PROTOCOL_VERSION, "type": "error", "error": "bad contract"})]
        )


def test_observation_rejects_invalid_execution_lag() -> None:
    request = ObservationRequest(
        "a" * 64,
        "session",
        0,
        100,
        "task",
        np.zeros(8),
        (80, 90),
        execution_lag_rad=float("nan"),
    )
    with pytest.raises(Exception, match="execution lag"):
        pack_observation(request, b"head", b"wrist")
