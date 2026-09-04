"""Map 1D policy gripper to the existing 6D HandsCmd presets."""

from __future__ import annotations

from dataclasses import dataclass

from hardware_teleop.config_loader import HardwareHandsConfig
from hardware_teleop.hand_mapping import left_hand_positions, right_hand_positions


OPEN = 0.0
GRASP = 1.0


@dataclass
class BinaryGripper:
    open_threshold: float = 0.35
    grasp_threshold: float = 0.65
    state: float = OPEN

    def reset(self, state: float = OPEN) -> None:
        self.state = OPEN if float(state) < 0.5 else GRASP

    def update(self, trigger: float) -> float:
        value = float(trigger)
        if self.state < 0.5 and value > self.grasp_threshold:
            self.state = GRASP
        elif self.state >= 0.5 and value < self.open_threshold:
            self.state = OPEN
        return self.state


def gripper_to_hand6(
    hands_cfg: HardwareHandsConfig,
    *,
    side: str,
    gripper: float,
) -> list[int]:
    trigger = 0.0 if float(gripper) < 0.5 else 1.0
    if side == "left":
        return left_hand_positions(hands_cfg, trigger)
    if side == "right":
        return right_hand_positions(hands_cfg, trigger)
    raise ValueError(f"unsupported hand side: {side}")


def bimanual_hand6(
    hands_cfg: HardwareHandsConfig,
    *,
    active_arm: str,
    gripper: float,
    idle_gripper: float = OPEN,
) -> tuple[list[int], list[int]]:
    left_gripper = gripper if active_arm == "left" else idle_gripper
    right_gripper = gripper if active_arm == "right" else idle_gripper
    return (
        gripper_to_hand6(hands_cfg, side="left", gripper=left_gripper),
        gripper_to_hand6(hands_cfg, side="right", gripper=right_gripper),
    )
