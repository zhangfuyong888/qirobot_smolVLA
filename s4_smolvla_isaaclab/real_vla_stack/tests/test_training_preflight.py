from __future__ import annotations

import json
import struct

import pytest

from real_vla_stack.common.config import load_pipeline_config
from real_vla_stack.host.training.launcher import training_command
from real_vla_stack.host.training.preflight import preflight_pretrained_policy


def _snapshot(tmp_path, *, include_expert: bool = True):
    root = tmp_path / "smolvla_base"
    root.mkdir()
    config = {
        "type": "smolvla",
        "input_features": {},
        "output_features": {},
        "chunk_size": 50,
        "n_action_steps": 50,
        "n_obs_steps": 1,
        "max_state_dim": 32,
        "max_action_dim": 32,
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        (root / name).write_text("{}", encoding="utf-8")
    (root / "policy_preprocessor_step_0_normalizer_processor.safetensors").write_bytes(b"stats")
    names = ["model.action_in_proj.weight", "model.action_out_proj.weight", "model.state_proj.weight"]
    if include_expert:
        names.append("model.vlm_with_expert.lm_expert.layers.0.weight")
    header = {name: {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]} for name in names}
    encoded = json.dumps(header).encode()
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"0" * 16)
    return root


def test_pretrained_preflight_requires_action_expert_and_projections(tmp_path) -> None:
    cfg = load_pipeline_config()
    report = preflight_pretrained_policy(
        _snapshot(tmp_path),
        cfg.contract,
        expected_chunk_size=50,
        expected_n_obs_steps=1,
        train_expert_only=True,
        train_state_proj=True,
    )
    assert report["max_state_dim"] == 32
    assert report["trainable_parameter_estimate"] == 16


def test_pretrained_preflight_fails_closed_without_expert(tmp_path) -> None:
    cfg = load_pipeline_config()
    with pytest.raises(ValueError, match="core SmolVLA"):
        preflight_pretrained_policy(
            _snapshot(tmp_path, include_expert=False),
            cfg.contract,
            expected_chunk_size=50,
            expected_n_obs_steps=1,
            train_expert_only=True,
            train_state_proj=True,
        )


def test_pretrained_training_preserves_snapshot_architecture() -> None:
    cfg = load_pipeline_config()
    command, output = training_command(cfg, profile="smoke")
    assert any(value.startswith("--policy.path=") for value in command)
    assert "--policy.strict_pretrained_loading=true" in command
    assert "--policy.input_features=null" in command
    forbidden = (
        "--policy.type=",
        "--policy.chunk_size=",
        "--policy.max_state_dim=",
        "--policy.max_action_dim=",
        "--policy.vlm_model_name=",
        "--policy.load_vlm_weights=",
    )
    assert not any(value.startswith(forbidden) for value in command)
    assert "--steps=300" in command
    assert output.name.endswith("_smolvla_base_ft_smoke")
