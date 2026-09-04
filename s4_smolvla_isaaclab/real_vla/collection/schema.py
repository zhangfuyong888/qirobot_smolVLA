"""Policy and raw-episode contracts for real-robot collection."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from real_vla import SCHEMA_VERSION

STATE_DIM = 8
ACTION_DIM = 8


@dataclass(frozen=True)
class PolicyState:
    timestamp_ns: int
    arm_q: np.ndarray
    gripper_state: float

    def as_8d(self) -> np.ndarray:
        return np.concatenate(
            [np.asarray(self.arm_q, dtype=np.float64).reshape(7), [float(self.gripper_state)]]
        )


@dataclass(frozen=True)
class PublishedCommand:
    timestamp_ns: int
    arm_target_q: np.ndarray
    gripper_target: float
    hand_command_6d: np.ndarray
    quest_trigger: float
    limited: bool
    motion_allowed: bool

    def as_8d(self) -> np.ndarray:
        return np.concatenate(
            [np.asarray(self.arm_target_q, dtype=np.float64).reshape(7), [float(self.gripper_target)]]
        )


@dataclass
class CameraFrame:
    timestamp_ns: int
    capture_seq: int
    image_bgr: np.ndarray
    name: str


@dataclass
class EpisodeMeta:
    schema_version: str = SCHEMA_VERSION
    episode_id: int = 0
    task: str = ""
    active_arm: str = "right"
    cameras: list[str] = field(default_factory=list)
    control_hz: float = 30.0
    result: str = "pending"
    quality_valid: bool = True
    quality_warning: bool = False
    quality_notes: list[str] = field(default_factory=list)
    git_commit: str = ""
    duration_s: float = 0.0
    t_start_ns: int = 0
    t_end_ns: int = 0
    writer_drops: dict[str, int] = field(default_factory=dict)
    alignment: dict[str, dict[str, float]] = field(default_factory=dict)
    state_spec: dict[str, int] = field(
        default_factory=lambda: {"arm_q": 7, "gripper": 1, "dim": STATE_DIM}
    )
    action_spec: dict[str, object] = field(
        default_factory=lambda: {
            "arm_target_q": 7,
            "gripper_target": 1,
            "dim": ACTION_DIM,
            "semantics": "absolute_joint_target",
        }
    )
