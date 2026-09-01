"""Isaac-independent Pink IK backend for real-robot teleoperation."""

from __future__ import annotations

import numpy as np

from hardware_teleop.config_loader import HardwareTeleopConfig
from teleoperation.controllers.pink_solver import PinkBimanualSolver
from teleoperation.mapping import TcpPose


class PinkHardwareIkBackend:
    """Expose the vendored Pink solver with a strict LA7+RA7 contract."""

    name = "pink"

    def __init__(self, config: HardwareTeleopConfig) -> None:
        self._controller = PinkBimanualSolver(config.teleop.controller.pink)

    def forward(self, arm_q14: np.ndarray) -> tuple[TcpPose, TcpPose]:
        return self._controller.forward(arm_q14)

    def set_posture_reference(self, arm_q14: np.ndarray) -> None:
        self._controller.set_posture_reference(arm_q14)

    def compute(
        self,
        arm_q14: np.ndarray,
        dt: float,
        left_target: TcpPose,
        right_target: TcpPose,
    ) -> np.ndarray:
        return self._controller.compute(arm_q14, dt, left_target, right_target)

    def diagnostics(self) -> dict[str, str | float]:
        details = self._controller.diagnostics()
        details["runtime"] = "hardware_no_isaac"
        return details
