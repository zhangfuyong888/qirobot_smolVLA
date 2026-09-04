"""Thin adapter over HardwareRobotBridge. No IsaacLab imports."""

from __future__ import annotations

import time

import numpy as np

from hardware_teleop.joint_mapping import ARM_JOINT_NAMES, apply_arm_q14, bimanual_to_arm_q14
from hardware_teleop.ros.robot_bridge import HardwareRobotBridge
from s4_robot.control_mapping import ACTION_SLICES, BIMANUAL_ACTION_DIM
from s4_robot.s4_robot_cfg import LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS

from real_vla.collection.schema import PublishedCommand
from real_vla.robot.gripper_adapter import OPEN, bimanual_hand6


class S4Adapter:
    def __init__(self, bridge: HardwareRobotBridge, *, active_arm: str, hands_cfg) -> None:
        if active_arm not in {"left", "right"}:
            raise ValueError("active_arm must be left or right")
        self.bridge = bridge
        self.active_arm = active_arm
        self.hands_cfg = hands_cfg
        self._last_published: PublishedCommand | None = None

    def spin_once(self) -> None:
        self.bridge.spin_once(timeout_sec=0.0)

    def read_bimanual(self) -> np.ndarray:
        return self.bridge.read_bimanual_state()

    def read_arm_q7(self, action: np.ndarray | None = None) -> np.ndarray:
        action_np = self.read_bimanual() if action is None else np.asarray(action)
        sl = ACTION_SLICES.left_arm if self.active_arm == "left" else ACTION_SLICES.right_arm
        return np.asarray(action_np[sl], dtype=np.float64).copy()

    def measured_state_8d(self, gripper_state: float, timestamp_ns: int | None = None) -> np.ndarray:
        del timestamp_ns
        return np.concatenate([self.read_arm_q7(), [float(gripper_state)]])

    def arm_names(self) -> tuple[str, ...]:
        return tuple(LEFT_ARM_JOINTS if self.active_arm == "left" else RIGHT_ARM_JOINTS)

    def publish(
        self,
        command_action: np.ndarray,
        *,
        gripper_target: float,
        quest_trigger: float,
        allow_motion: bool,
    ) -> PublishedCommand:
        timestamp_ns = time.monotonic_ns()
        desired = bimanual_to_arm_q14(command_action)
        targets = self.bridge.publish_arm_command(
            command_action,
            allow_motion=allow_motion,
            hold_commanded=False,
        )
        published_q14 = desired.copy()
        limited = False
        if targets:
            for index, name in enumerate(ARM_JOINT_NAMES):
                if name in targets and abs(float(targets[name]) - float(desired[index])) > 1.0e-9:
                    limited = True
                if name in targets:
                    published_q14[index] = float(targets[name])
        left_6d, right_6d = bimanual_hand6(
            self.hands_cfg,
            active_arm=self.active_arm,
            gripper=gripper_target,
            idle_gripper=OPEN,
        )
        self.bridge.publish_hands(
            0.0 if self.active_arm != "left" else (0.0 if gripper_target < 0.5 else 1.0),
            0.0 if self.active_arm != "right" else (0.0 if gripper_target < 0.5 else 1.0),
        )
        hand_6d = np.asarray(left_6d if self.active_arm == "left" else right_6d, dtype=np.float64)
        sl = slice(0, 7) if self.active_arm == "left" else slice(7, 14)
        command = PublishedCommand(
            timestamp_ns=timestamp_ns,
            arm_target_q=published_q14[sl].copy(),
            gripper_target=0.0 if float(gripper_target) < 0.5 else 1.0,
            hand_command_6d=hand_6d,
            quest_trigger=float(quest_trigger),
            limited=limited,
            motion_allowed=bool(allow_motion and bool(targets)),
        )
        self._last_published = command
        return command

    def last_published(self) -> PublishedCommand | None:
        return self._last_published

    def overlay_active_arm(self, command_action: np.ndarray, arm_q7: np.ndarray) -> np.ndarray:
        q14 = bimanual_to_arm_q14(command_action)
        if self.active_arm == "left":
            q14[:7] = np.asarray(arm_q7, dtype=np.float64)
        else:
            q14[7:] = np.asarray(arm_q7, dtype=np.float64)
        result = np.asarray(command_action, dtype=np.float32).copy()
        if result.shape != (BIMANUAL_ACTION_DIM,):
            result = np.zeros(BIMANUAL_ACTION_DIM, dtype=np.float32)
        return apply_arm_q14(result, q14)
