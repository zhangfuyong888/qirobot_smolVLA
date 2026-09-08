from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any

from ...common.contract import PolicyContract


CORE_WEIGHT_PREFIXES = (
    "model.vlm_with_expert.lm_expert.",
    "model.action_in_proj.",
    "model.action_out_proj.",
    "model.state_proj.",
)


def _safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"invalid safetensors header: {path}")
        header_length = struct.unpack("<Q", length_bytes)[0]
        if header_length <= 0 or header_length > 100_000_000:
            raise ValueError(f"invalid safetensors header length: {header_length}")
        payload = stream.read(header_length)
    header = json.loads(payload)
    if not isinstance(header, dict):
        raise ValueError(f"invalid safetensors metadata: {path}")
    return header


def _parameter_count(
    header: dict[str, Any], *, train_expert_only: bool, train_state_proj: bool
) -> tuple[int, int]:
    trainable = 0
    frozen = 0
    for name, metadata in header.items():
        if name == "__metadata__" or not isinstance(metadata, dict):
            continue
        shape = metadata.get("shape")
        if not isinstance(shape, list):
            continue
        count = math.prod(int(value) for value in shape)
        is_trainable = True
        if train_expert_only and name.startswith("model.vlm_with_expert.vlm."):
            is_trainable = False
        if not train_state_proj and name.startswith("model.state_proj."):
            is_trainable = False
        if is_trainable:
            trainable += count
        else:
            frozen += count
    return trainable, frozen


def preflight_pretrained_policy(
    path: Path,
    contract: PolicyContract,
    *,
    expected_chunk_size: int,
    expected_n_obs_steps: int,
    train_expert_only: bool,
    train_state_proj: bool,
) -> dict[str, Any]:
    """Validate a local full-policy snapshot before handing it to LeRobot training."""
    root = Path(path).expanduser()
    if not root.is_absolute():
        raise ValueError("pretrained_policy must be an absolute local path")
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"pretrained SmolVLA policy directory not found: {root}")

    required = ("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json")
    missing = [name for name in required if not (root / name).is_file()]
    normalization_files = sorted(root.glob("policy_*processor_step_*normalizer_processor.safetensors"))
    if not normalization_files:
        missing.append("policy_*normalizer_processor.safetensors")
    if missing:
        raise FileNotFoundError(f"pretrained policy snapshot is incomplete: missing={missing}")

    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config.get("type") != "smolvla":
        raise ValueError(f"pretrained policy type={config.get('type')!r}, expected 'smolvla'")
    if int(config.get("chunk_size", -1)) != int(expected_chunk_size):
        raise ValueError(
            f"pretrained chunk_size={config.get('chunk_size')}, expected={expected_chunk_size}; "
            "architecture overrides are forbidden"
        )
    if int(config.get("n_obs_steps", -1)) != int(expected_n_obs_steps):
        raise ValueError(
            f"pretrained n_obs_steps={config.get('n_obs_steps')}, expected={expected_n_obs_steps}; "
            "architecture overrides are forbidden"
        )

    expected_inputs = {"observation.state", *contract.camera_keys}
    inputs = config.get("input_features") or {}
    outputs = config.get("output_features") or {}
    if contract.state_dim > int(config["max_state_dim"]):
        raise ValueError("drawer state does not fit pretrained max_state_dim")
    if contract.action_dim > int(config["max_action_dim"]):
        raise ValueError("drawer action does not fit pretrained max_action_dim")

    header = _safetensors_header(root / "model.safetensors")
    weight_names = tuple(name for name in header if name != "__metadata__")
    missing_core = [
        prefix
        for prefix in CORE_WEIGHT_PREFIXES
        if not any(name.startswith(prefix) for name in weight_names)
    ]
    if missing_core:
        raise ValueError(f"pretrained policy is missing core SmolVLA weights: {missing_core}")
    trainable, frozen = _parameter_count(
        header,
        train_expert_only=train_expert_only,
        train_state_proj=train_state_proj,
    )
    return {
        "pretrained_path": str(root),
        "snapshot_input_features": sorted(inputs),
        "snapshot_output_features": sorted(outputs),
        "adapted_input_features": sorted(expected_inputs),
        "adapted_output_features": ["action"],
        "max_state_dim": int(config["max_state_dim"]),
        "max_action_dim": int(config["max_action_dim"]),
        "chunk_size": int(config["chunk_size"]),
        "n_obs_steps": int(config["n_obs_steps"]),
        "n_action_steps": int(config["n_action_steps"]),
        "trainable_parameter_estimate": trainable,
        "frozen_parameter_estimate": frozen,
        "core_weights_verified": list(CORE_WEIGHT_PREFIXES),
    }
