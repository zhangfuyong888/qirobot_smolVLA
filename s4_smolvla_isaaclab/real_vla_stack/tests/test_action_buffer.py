from __future__ import annotations

import numpy as np
import pytest

from real_vla_stack.common.errors import ContractError, PolicyStaleError
from real_vla_stack.robot.rollout.action_buffer import ActionBuffer
from real_vla_stack.robot.rollout.safety import validate_policy_chunk


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


def test_policy_sanity_rejects_gross_jump() -> None:
    chunk = np.zeros((3, 8))
    chunk[1, 0] = 1.0
    with pytest.raises(ContractError, match="adjacent"):
        validate_policy_chunk(
            chunk, measured_q7=np.zeros(7), max_target_jump_rad=0.2, max_tracking_error_rad=0.25
        )
