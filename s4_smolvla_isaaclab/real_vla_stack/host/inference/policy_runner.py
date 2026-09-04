from __future__ import annotations

import contextlib
import os
from pathlib import Path

import numpy as np

from ...common.contract import PolicyContract
from ...common.errors import ContractError


def resolve_checkpoint(path: Path) -> Path:
    checkpoint = Path(path).expanduser().resolve()
    if (checkpoint / "pretrained_model").is_dir():
        checkpoint = checkpoint / "pretrained_model"
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"checkpoint config.json not found: {checkpoint}")
    return checkpoint


class PolicyRunner:
    def __init__(self, checkpoint: Path, contract: PolicyContract, *, device: str = "cuda") -> None:
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.contract = contract
        self.checkpoint = resolve_checkpoint(checkpoint)
        self.device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        config = PreTrainedConfig.from_pretrained(self.checkpoint, local_files_only=True)
        configured_vlm = Path(config.vlm_model_name).expanduser()
        if not configured_vlm.is_dir():
            model_root = Path(os.environ.get("SMOLVLA_MODEL_ROOT", "models")).expanduser()
            fallback = model_root / "HuggingFaceTB" / "SmolVLM2-500M-Video-Instruct"
            if not fallback.is_dir():
                raise FileNotFoundError(f"checkpoint VLM path unavailable: {configured_vlm}; fallback missing: {fallback}")
            config.vlm_model_name = str(fallback.resolve())
        with contextlib.redirect_stdout(None):
            self.policy = SmolVLAPolicy.from_pretrained(
                str(self.checkpoint), config=config, local_files_only=True
            ).to(self.device)
        self.policy.eval()
        self.policy.reset()
        self.image_keys = tuple(
            key for key, value in self.policy.config.input_features.items() if value.type.name == "VISUAL"
        )
        if self.image_keys != contract.camera_keys:
            raise ContractError(f"checkpoint image keys={self.image_keys}, contract={contract.camera_keys}")
        state_dim = int(self.policy.config.input_features["observation.state"].shape[0])
        action_dim = int(self.policy.config.output_features["action"].shape[0])
        if (state_dim, action_dim) != (contract.state_dim, contract.action_dim):
            raise ContractError(f"checkpoint state/action dims={(state_dim, action_dim)}, expected=(8,8)")
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=str(self.checkpoint),
            preprocessor_overrides={
                "tokenizer_processor": {"tokenizer_name": self.policy.config.vlm_model_name},
                "device_processor": {"device": str(self.device)},
            },
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )

    def reset(self) -> None:
        self.policy.reset()

    def predict_chunk(self, state: np.ndarray, images: dict[str, np.ndarray], task: str) -> np.ndarray:
        import torch
        from lerobot.policies import prepare_observation_for_inference

        state = np.asarray(state, dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ContractError("inference state must be finite float32[8]")
        if set(images) != set(self.contract.camera_keys):
            raise ContractError(f"inference image keys={sorted(images)}, expected={sorted(self.contract.camera_keys)}")
        observation: dict[str, np.ndarray] = {"observation.state": state}
        for key, image in images.items():
            value = np.asarray(image, dtype=np.uint8)
            if value.ndim != 3 or value.shape[2] != 3:
                raise ContractError(f"{key} must be uint8 HWC RGB")
            observation[key] = value
        with torch.inference_mode():
            batch = prepare_observation_for_inference(
                observation, self.device, task=str(task), robot_type=self.contract.robot_type
            )
            batch = self.preprocessor(batch)
            chunk = self.postprocessor(self.policy.predict_action_chunk(batch))
        result = chunk.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if result.ndim != 2 or result.shape[1] != 8 or not np.isfinite(result).all():
            raise ContractError(f"postprocessed policy output is invalid: {result.shape}")
        return result
