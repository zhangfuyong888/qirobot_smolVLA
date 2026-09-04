from __future__ import annotations

import json

import pytest

from teleoperation.protocol import LatestFrameStore, parse_controller_frame


def frame_payload(sequence: int = 1) -> dict:
    side = {
        "valid": True,
        "position": [0.1, 1.2, -0.3],
        "orientation_xyzw": [0.0, 0.0, 0.0, 2.0],
        "trigger": 0.25,
        "squeeze": 0.75,
        "buttons": [0.25, 0.75],
        "axes": [0.0, 0.1],
        "profiles": ["oculus-touch-v3"],
    }
    return {
        "type": "controller_frame",
        "version": 1,
        "session_id": "test-session",
        "sequence": sequence,
        "client_time_ms": 123.0,
        "reference_space": "local-floor",
        "calibration_id": 3,
        "calibration_viewer_orientation_xyzw": [0.0, 0.0, 0.0, 2.0],
        "boundary_safe": True,
        "boundary_distance_m": 0.8,
        "left": side,
        "right": side,
    }


def test_protocol_parses_one_atomic_bimanual_frame() -> None:
    frame = parse_controller_frame(json.dumps(frame_payload()), received_monotonic=4.0)
    assert frame.sequence == 1
    assert frame.left.trigger == pytest.approx(0.25)
    assert frame.right.squeeze == pytest.approx(0.75)
    assert frame.left.orientation_xyzw == pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert frame.calibration_id == 3
    assert frame.calibration_viewer_orientation_xyzw == pytest.approx((0.0, 0.0, 0.0, 1.0))
    assert frame.boundary_safe
    assert frame.boundary_distance_m == pytest.approx(0.8)
    assert frame.received_monotonic == 4.0


def test_latest_frame_store_rejects_old_sequences() -> None:
    store = LatestFrameStore()
    newer = parse_controller_frame(frame_payload(2), received_monotonic=2.0)
    older = parse_controller_frame(frame_payload(1), received_monotonic=3.0)
    assert store.publish(newer)
    assert not store.publish(older)
    assert store.snapshot() == newer
    assert store.stats()["rejected"] == 1


def test_store_allows_only_one_controller_client() -> None:
    store = LatestFrameStore()
    assert store.client_connected()
    assert not store.client_connected()
    store.client_disconnected()
    assert store.client_connected()


def test_protocol_rejects_non_finite_pose() -> None:
    payload = frame_payload()
    payload["left"]["position"][0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        parse_controller_frame(payload)


def test_protocol_rejects_partial_heading_calibration() -> None:
    payload = frame_payload()
    payload["calibration_viewer_orientation_xyzw"] = None
    with pytest.raises(ValueError, match="must be provided together"):
        parse_controller_frame(payload)


def test_protocol_rejects_non_finite_boundary_distance() -> None:
    payload = frame_payload()
    payload["boundary_distance_m"] = float("nan")
    with pytest.raises(ValueError, match="boundary_distance_m must be finite"):
        parse_controller_frame(payload)


def test_protocol_rejects_non_boolean_boundary_state() -> None:
    payload = frame_payload()
    payload["boundary_safe"] = "false"
    with pytest.raises(ValueError, match="boundary_safe must be a boolean"):
        parse_controller_frame(payload)
