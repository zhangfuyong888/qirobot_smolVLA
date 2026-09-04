from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import ContractError
from .hashing import payload_sha256


CANONICAL_RIGHT_ARM_JOINTS = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


@dataclass(frozen=True)
class PolicyContract:
    contract_version: str
    raw_schema_version: str
    task_id: str
    task: str
    robot_type: str
    active_arm: str
    dataset_fps: int
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    camera_keys: tuple[str, ...]
    camera_sources: tuple[str, ...]
    phase_filter: tuple[str, ...]
    max_camera_age_ms: float
    max_cross_camera_skew_ms: float
    lerobot_commit: str
    state_semantics: str = "measured_joint_state"
    action_semantics: str = "absolute_joint_target"
    units: str = "rad"
    image_color_space: str = "RGB"
    gripper_semantics: str = "logical_commanded_state_not_measured"
    gripper_open_threshold: float = 0.35
    gripper_grasp_threshold: float = 0.65
    alignment: str = "causal_latest_before"

    @property
    def state_dim(self) -> int:
        return len(self.state_names)

    @property
    def action_dim(self) -> int:
        return len(self.action_names)

    @property
    def sha256(self) -> str:
        return payload_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state_dim"] = self.state_dim
        payload["action_dim"] = self.action_dim
        return payload

    def document(self) -> dict[str, Any]:
        return {**self.to_dict(), "contract_sha256": self.sha256}

    def validate(self) -> None:
        expected = CANONICAL_RIGHT_ARM_JOINTS + ("gripper",)
        if self.active_arm != "right":
            raise ContractError("v1 supports only the commissioned right-arm policy")
        if self.state_names != expected or self.action_names != expected:
            raise ContractError(f"state/action order must be the canonical right-arm 7+1 layout: {expected}")
        if self.state_dim != 8 or self.action_dim != 8:
            raise ContractError("real policy state/action dimensions must both be 8")
        if self.action_semantics != "absolute_joint_target":
            raise ContractError("only absolute_joint_target actions are safe for this stack")
        if self.alignment != "causal_latest_before":
            raise ContractError("image alignment must be causal_latest_before")
        if self.image_color_space != "RGB":
            raise ContractError("policy images must be RGB")
        if self.dataset_fps <= 0:
            raise ContractError("dataset_fps must be positive")
        if len(self.camera_keys) != 2 or len(self.camera_sources) != 2:
            raise ContractError("exactly two camera keys and sources are required")
        if len(set(self.camera_keys)) != 2 or len(set(self.camera_sources)) != 2:
            raise ContractError("camera keys and sources must be unique")
        if not self.phase_filter:
            raise ContractError("phase_filter cannot be empty")
        if "return_home" not in self.phase_filter and "return home" in self.task.lower():
            raise ContractError("task text promises return home but return_home is excluded from training")
        if not 0 <= self.gripper_open_threshold < self.gripper_grasp_threshold <= 1:
            raise ContractError("gripper thresholds must satisfy 0 <= open < grasp <= 1")

    def make_state(self, arm_q: np.ndarray, gripper: float) -> np.ndarray:
        value = np.concatenate((np.asarray(arm_q, dtype=np.float32).reshape(7), [float(gripper)]))
        if not np.isfinite(value).all():
            raise ContractError("state contains non-finite values")
        return value.astype(np.float32)

    def validate_action(self, action: np.ndarray) -> np.ndarray:
        value = np.asarray(action, dtype=np.float32)
        if value.shape != (self.action_dim,) or not np.isfinite(value).all():
            raise ContractError(f"action must be finite with shape ({self.action_dim},), got {value.shape}")
        return value

    def write(self, path: Path) -> None:
        self.validate()
        Path(path).write_text(json.dumps(self.document(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> "PolicyContract":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        claimed = str(raw.pop("contract_sha256", ""))
        raw.pop("state_dim", None)
        raw.pop("action_dim", None)
        for key in ("state_names", "action_names", "camera_keys", "camera_sources", "phase_filter"):
            raw[key] = tuple(raw[key])
        contract = cls(**raw)
        contract.validate()
        if not claimed or claimed != contract.sha256:
            raise ContractError(f"contract hash mismatch in {path}")
        return contract
