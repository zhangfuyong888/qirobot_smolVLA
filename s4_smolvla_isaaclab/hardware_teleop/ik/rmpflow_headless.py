"""Headless Isaac RMPflow IK for hardware teleoperation."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from hardware_teleop.config_loader import HardwareTeleopConfig
from teleoperation.controllers.rmpflow_backend import BimanualRmpFlowController
from teleoperation.mapping import TcpPose


class HeadlessRmpFlowIkBackend:
    name = "rmpflow"

    def __init__(self, config: HardwareTeleopConfig, robot: Any, device: str, base_body_id: int) -> None:
        teleop = config.teleop
        rmpflow_cfg = replace(teleop.controller.rmpflow, update_every_n_steps=1)
        self._controller = BimanualRmpFlowController(robot, device, base_body_id, rmpflow_cfg)

    def set_posture_reference(self, joint_positions: np.ndarray) -> None:
        self._controller.set_posture_reference(joint_positions)

    def compute(
        self,
        joint_positions: np.ndarray,
        dt: float,
        left_target: TcpPose,
        right_target: TcpPose,
    ) -> np.ndarray:
        return self._controller.compute(joint_positions, dt, left_target, right_target)

    def diagnostics(self) -> dict[str, str | float]:
        details = self._controller.diagnostics()
        details["runtime"] = "headless_isaac"
        return details
