from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ...common.contract import PolicyContract
from ...common.errors import ContractError


@dataclass
class RTCSessionState:
    accepted_request_id: int = -1
    accepted_raw_chunk: object | None = None
    accepted_observation_ns: int | None = None
    generated: dict[int, tuple[int, object]] = field(default_factory=dict)

    def reset(self) -> None:
        self.accepted_request_id = -1
        self.accepted_raw_chunk = None
        self.accepted_observation_ns = None
        self.generated.clear()

    def acknowledge(self, request_id: int) -> None:
        value = int(request_id)
        if value == self.accepted_request_id:
            # Any newer generated chunk was observed but not accepted by the
            # robot before this request, so it can never become executable.
            self.generated.clear()
            return
        if value < self.accepted_request_id:
            raise ContractError("RTC accepted request id moved backwards")
        if value not in self.generated:
            raise ContractError(f"RTC acknowledgement refers to unknown request_id={value}")
        observation_ns, chunk = self.generated[value]
        self.accepted_request_id = value
        self.accepted_observation_ns = observation_ns
        self.accepted_raw_chunk = chunk
        self.generated.clear()


def resolve_checkpoint(path: Path) -> Path:
    checkpoint = Path(path).expanduser().resolve()
    if (checkpoint / "pretrained_model").is_dir():
        checkpoint = checkpoint / "pretrained_model"
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"checkpoint config.json not found: {checkpoint}")
    return checkpoint


class PolicyRunner:
    def __init__(
        self,
        checkpoint: Path,
        contract: PolicyContract,
        *,
        device: str = "cuda",
        rtc_enabled: bool = False,
        rtc_execution_horizon: int = 10,
        rtc_max_guidance_weight: float = 10.0,
    ) -> None:
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.rtc.configuration_rtc import RTCConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.contract = contract
        self.checkpoint = resolve_checkpoint(checkpoint)
        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "policy server requires CUDA but torch.cuda.is_available() is false; "
                "refusing an unsafe silent CPU fallback"
            )
        self.device = requested_device
        config = PreTrainedConfig.from_pretrained(self.checkpoint, local_files_only=True)
        config.rtc_config = RTCConfig(
            enabled=bool(rtc_enabled),
            execution_horizon=int(rtc_execution_horizon),
            max_guidance_weight=float(rtc_max_guidance_weight),
        )
        configured_vlm = Path(config.vlm_model_name).expanduser()
        if not configured_vlm.is_dir():
            model_root = Path(os.environ.get("SMOLVLA_MODEL_ROOT", "models")).expanduser()
            fallback = model_root / "HuggingFaceTB" / "SmolVLM2-500M-Video-Instruct"
            if not fallback.is_dir():
                raise FileNotFoundError(
                    f"checkpoint VLM path unavailable: {configured_vlm}; fallback missing: {fallback}"
                )
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
        self.rtc_enabled = bool(rtc_enabled)
        self.rtc_execution_horizon = int(rtc_execution_horizon)
        self.rtc_state = RTCSessionState()
        self.last_diagnostics: dict[str, int | bool] = {}

    def reset(self) -> None:
        self.policy.reset()
        self.rtc_state.reset()
        self.last_diagnostics = {}

    def predict_chunk(
        self,
        state: np.ndarray,
        images: dict[str, np.ndarray],
        task: str,
        *,
        observation_timestamp_ns: int | None = None,
        inference_delay_steps: int = 0,
        request_id: int | None = None,
        previous_accepted_request_id: int = -1,
    ) -> np.ndarray:
        import torch
        from lerobot.policies import prepare_observation_for_inference

        state = np.asarray(state, dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ContractError("inference state must be finite float32[8]")
        if set(images) != set(self.contract.camera_keys):
            raise ContractError(
                f"inference image keys={sorted(images)}, "
                f"expected={sorted(self.contract.camera_keys)}"
            )
        observation: dict[str, np.ndarray] = {"observation.state": state}
        for key, image in images.items():
            value = np.asarray(image, dtype=np.uint8)
            if value.ndim != 3 or value.shape[2] != 3:
                raise ContractError(f"{key} must be uint8 HWC RGB")
            observation[key] = value
        prev_actions = None
        prev_leftover_steps = 0
        if self.rtc_enabled:
            self.rtc_state.acknowledge(previous_accepted_request_id)
        if self.rtc_enabled and self.rtc_state.accepted_raw_chunk is not None:
            if observation_timestamp_ns is None or self.rtc_state.accepted_observation_ns is None:
                raise ContractError("RTC inference requires monotonic observation timestamps")
            elapsed_ns = int(observation_timestamp_ns) - self.rtc_state.accepted_observation_ns
            if elapsed_ns < 0:
                raise ContractError("RTC observation timestamp moved backwards")
            elapsed_steps = int(elapsed_ns * self.contract.dataset_fps / 1_000_000_000)
            leftover = self.rtc_state.accepted_raw_chunk[elapsed_steps:]
            prev_leftover_steps = int(leftover.shape[0])
            horizon = self.rtc_execution_horizon
            prev_actions = torch.zeros(
                (horizon, leftover.shape[1]),
                dtype=leftover.dtype,
                device=leftover.device,
            )
            copied = min(horizon, leftover.shape[0])
            if copied:
                prev_actions[:copied] = leftover[:copied]
        delay = max(0, min(int(inference_delay_steps), self.rtc_execution_horizon))
        # RTC temporarily enables autograd inside its denoising guidance. An
        # outer inference_mode cannot be overridden, while no_grad can.
        with torch.no_grad():
            batch = prepare_observation_for_inference(
                observation, self.device, task=str(task), robot_type=self.contract.robot_type
            )
            batch = self.preprocessor(batch)
            raw_chunk = self.policy.predict_action_chunk(
                batch,
                inference_delay=delay,
                prev_chunk_left_over=prev_actions,
                execution_horizon=self.rtc_execution_horizon,
            )
            chunk = self.postprocessor(raw_chunk)
        if self.rtc_enabled:
            if request_id is None or observation_timestamp_ns is None:
                raise ContractError("RTC inference requires request_id and observation timestamp")
            self.rtc_state.generated[int(request_id)] = (
                int(observation_timestamp_ns),
                raw_chunk.squeeze(0).detach().clone(),
            )
        self.last_diagnostics = {
            "rtc_enabled": self.rtc_enabled,
            "rtc_inference_delay_steps": delay,
            "rtc_execution_horizon": self.rtc_execution_horizon if self.rtc_enabled else 0,
            "rtc_prev_leftover_steps": prev_leftover_steps,
            "raw_chunk_length": int(raw_chunk.shape[1]),
            "rtc_source_request_id": self.rtc_state.accepted_request_id,
        }
        result = chunk.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if result.ndim != 2 or result.shape[1] != 8 or not np.isfinite(result).all():
            raise ContractError(f"postprocessed policy output is invalid: {result.shape}")
        return result
