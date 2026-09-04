from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from real_vla_stack.common.config import load_pipeline_config
from real_vla_stack.common.contract import CANONICAL_RIGHT_ARM_JOINTS, PolicyContract
from real_vla_stack.common.errors import ContractError
from s4_robot.s4_robot_cfg import RIGHT_ARM_JOINTS


def test_contract_matches_robot_joint_order_and_round_trips(tmp_path) -> None:
    config = load_pipeline_config()
    assert tuple(RIGHT_ARM_JOINTS) == CANONICAL_RIGHT_ARM_JOINTS
    assert config.contract.state_names == tuple(RIGHT_ARM_JOINTS) + ("gripper",)
    path = tmp_path / "contract.json"
    config.contract.write(path)
    assert PolicyContract.read(path) == config.contract
    assert len(config.contract.sha256) == 64


def test_contract_rejects_return_home_prompt_when_phase_is_excluded() -> None:
    contract = load_pipeline_config().contract
    with pytest.raises(ContractError, match="promises return home"):
        replace(contract, task=contract.task + " Return home.").validate()


def test_contract_state_and_action_are_finite_8d() -> None:
    contract = load_pipeline_config().contract
    assert contract.make_state(np.arange(7), 1.0).shape == (8,)
    with pytest.raises(ContractError):
        contract.validate_action(np.full(8, np.nan))
