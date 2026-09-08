from __future__ import annotations

import json
import math
import struct
from gc import collect
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


def strict_load_pretrained_policy(path: Path, *, model_root: Path) -> dict[str, Any]:
    """Load the complete base policy strictly without patching LeRobot.

    This is deliberately kept in the S4 training integration rather than in the
    pinned ``lerobot`` submodule.  It proves that the snapshot's complete state
    dict, including the action expert and the action/state projections, loads
    into the upstream SmolVLA architecture with no missing or unexpected keys.

    The check runs on CPU before training begins.  Dataset feature adaptation is
    validated separately by :func:`preflight_pretrained_policy`: the real 8D
    state/action dimensions must fit SmolVLA's pretrained padded dimensions.
    """
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"pretrained SmolVLA policy directory not found: {root}")

    # These imports intentionally stay host/training-only.  ``common`` and the
    # robot rollout process remain free of Torch and LeRobot imports.
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    config = PreTrainedConfig.from_pretrained(root, local_files_only=True)
    configured_vlm = Path(config.vlm_model_name).expanduser()
    if not configured_vlm.is_dir():
        fallback = (
            Path(model_root).expanduser().resolve()
            / "HuggingFaceTB"
            / "SmolVLM2-500M-Video-Instruct"
        )
        if not fallback.is_dir():
            raise FileNotFoundError(
                f"pretrained VLM path unavailable: {configured_vlm}; fallback missing: {fallback}"
            )
        config.vlm_model_name = str(fallback)
    config.device = "cpu"
    policy = SmolVLAPolicy.from_pretrained(
        str(root), config=config, local_files_only=True, strict=True
    )
    state_names = tuple(policy.state_dict())
    missing_core = [
        prefix for prefix in CORE_WEIGHT_PREFIXES if not any(name.startswith(prefix) for name in state_names)
    ]
    if missing_core:
        raise ValueError(f"strict policy load is missing core SmolVLA modules: {missing_core}")
    report = {
        "strict_weight_load": "passed",
        "strict_weight_load_device": "cpu",
        "strict_weight_load_vlm": str(config.vlm_model_name),
        "loaded_tensor_count": len(state_names),
        "core_modules_loaded": list(CORE_WEIGHT_PREFIXES),
    }
    del policy
    collect()
    return report
