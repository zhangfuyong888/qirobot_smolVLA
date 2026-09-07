from __future__ import annotations

import numpy as np
import pytest

from real_vla_stack.common.errors import ContractError, PolicyStaleError
from real_vla_stack.robot.rollout.action_buffer import ActionBuffer, sample_policy_chunk
from real_vla_stack.robot.rollout.safety import validate_execution_target, validate_policy_chunk


def test_arm_interpolates_20hz_to_control_time_but_gripper_is_stepwise() -> None:
    buffer = ActionBuffer(policy_hz=20, execute_horizon=3, max_chunk_age_ms=500)
    chunk = np.zeros((3, 8), dtype=np.float32)
    chunk[1, :7] = 1.0
    chunk[1, 7] = 1.0
    buffer.replace(chunk, request_id=1, received_at_ns=1_000_000_000)
    action = buffer.sample(1_025_000_000)
    assert action[:7] == pytest.approx(np.full(7, 0.5))
    assert action[7] == 0.0


def test_buffer_rejects_old_response_and_stale_chunk() -> None:
    buffer = ActionBuffer(policy_hz=20, execute_horizon=2, max_chunk_age_ms=100)
    buffer.replace(np.zeros((2, 8)), request_id=2, received_at_ns=0)
    with pytest.raises(ContractError, match="out-of-order"):
        buffer.replace(np.zeros((2, 8)), request_id=1, received_at_ns=1)
    with pytest.raises(PolicyStaleError):
        buffer.sample(101_000_000)


def test_buffer_reports_source_age_for_degraded_hold_logic() -> None:
    buffer = ActionBuffer(policy_hz=20, execute_horizon=2, max_chunk_age_ms=900)
    buffer.replace(np.zeros((2, 8)), request_id=1, received_at_ns=100_000_000)
    assert buffer.age_ms(650_000_000) == pytest.approx(550.0)


def test_buffer_uses_observation_time_and_skips_delayed_actions() -> None:
    buffer = ActionBuffer(policy_hz=20, execute_horizon=4, max_chunk_age_ms=500)
    chunk = np.zeros((4, 8), dtype=np.float32)
    chunk[:, :7] = np.arange(4, dtype=np.float32)[:, None]
    buffer.replace(
        chunk,
        request_id=1,
        source_at_ns=1_000_000_000,
        received_at_ns=1_100_000_000,
    )
    # A 100 ms inference/transport delay means execution starts at policy step 2.
    assert buffer.sample(1_100_000_000)[:7] == pytest.approx(np.full(7, 2.0))


def test_replacement_blends_from_last_published_target() -> None:
    buffer = ActionBuffer(
        policy_hz=20,
        execute_horizon=4,
        max_chunk_age_ms=500,
        blend_duration_ms=100,
    )
    chunk = np.ones((4, 8), dtype=np.float32)
    buffer.replace(
        chunk,
        request_id=1,
        source_at_ns=1_000_000_000,
        received_at_ns=1_100_000_000,
        transition_from_q7=np.zeros(7),
    )
    assert buffer.sample(1_100_000_000)[:7] == pytest.approx(np.zeros(7))
    assert buffer.sample(1_150_000_000)[:7] == pytest.approx(np.full(7, 0.5))
    assert buffer.sample(1_200_000_000)[:7] == pytest.approx(np.ones(7))


def test_policy_chunk_sampler_uses_observation_timeline() -> None:
    chunk = np.zeros((4, 8), dtype=np.float32)
    chunk[:, :7] = np.arange(4, dtype=np.float32)[:, None]
    sampled = sample_policy_chunk(
        chunk,
        policy_hz=20,
        source_at_ns=1_000_000_000,
        now_ns=1_125_000_000,
    )
    assert sampled[:7] == pytest.approx(np.full(7, 2.5))


def test_policy_sanity_rejects_gross_jump() -> None:
    chunk = np.zeros((3, 8))
    chunk[1, 0] = 1.0
    with pytest.raises(ContractError, match="adjacent"):
        validate_policy_chunk(
            chunk, measured_q7=np.zeros(7), max_target_jump_rad=0.2, max_tracking_error_rad=0.25
        )


def test_policy_preflight_can_defer_only_initial_tracking_check() -> None:
    chunk = np.zeros((3, 8))
    measured = np.ones(7)
    with pytest.raises(ContractError, match="first policy target"):
        validate_policy_chunk(
            chunk,
            measured_q7=measured,
            max_target_jump_rad=0.2,
            max_tracking_error_rad=0.25,
        )
    assert validate_policy_chunk(
        chunk,
        measured_q7=measured,
        max_target_jump_rad=0.2,
        max_tracking_error_rad=0.25,
        enforce_initial_tracking=False,
    ).shape == (3, 8)


def test_finite_gripper_overshoot_is_safely_clipped() -> None:
    chunk = np.zeros((3, 8))
    chunk[:, 7] = [-0.2, 0.5, 1.3]
    safe = validate_policy_chunk(
        chunk,
        measured_q7=np.zeros(7),
        max_target_jump_rad=0.2,
        max_tracking_error_rad=0.25,
    )
    assert safe[:, 7] == pytest.approx([0.0, 0.5, 1.0])


def test_execution_target_is_checked_against_current_state() -> None:
    with pytest.raises(ContractError, match="delay-aligned.*joint=3"):
        validate_execution_target(
            [0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0],
            measured_q7=np.zeros(7),
            max_tracking_error_rad=0.25,
        )
