"""Pinocchio DLS fallback IK for hardware teleoperation."""

from __future__ import annotations

from typing import Any

import numpy as np

from hardware_teleop.config_loader import HardwareTeleopConfig
from teleoperation.controllers.pinocchio_backend import PinocchioTeleopController
from teleoperation.mapping import TcpPose


class PinocchioHardwareIkBackend:
    name = "pinocchio"

    def __init__(self, config: HardwareTeleopConfig, robot: Any, device: str) -> None:
        self._controller = PinocchioTeleopController(robot, device, config.teleop)

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
        details["runtime"] = "pinocchio"
        return details
