"""Simulation adapter for the repository-vendored Pink bimanual solver."""

from __future__ import annotations

from typing import Any

import numpy as np

from s4_robot.s4_robot_cfg import LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS
from teleoperation.config import TeleopConfig
from teleoperation.controllers.pink_solver import PinkBimanualSolver
from teleoperation.mapping import TcpPose


class PinkTeleopController:
    name = "pink"

    def __init__(self, robot: Any, config: TeleopConfig) -> None:
        self._joint_ids = tuple(
            robot.joint_names.index(name) for name in LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
        )
        self._solver = PinkBimanualSolver(config.controller.pink)

    def _arm_state(self, joint_positions: np.ndarray) -> np.ndarray:
        state = np.asarray(joint_positions, dtype=np.float64)
        return state[list(self._joint_ids)]

    def set_posture_reference(self, joint_positions: np.ndarray) -> None:
        self._solver.set_posture_reference(self._arm_state(joint_positions))

    def forward(self, joint_positions: np.ndarray) -> tuple[TcpPose, TcpPose]:
        """Return Pink FK for runtime parity checks against Isaac articulation FK."""
        return self._solver.forward(self._arm_state(joint_positions))

    def compute(
        self,
        joint_positions: np.ndarray,
        dt: float,
        left_target: TcpPose,
        right_target: TcpPose,
    ) -> np.ndarray:
        return self._solver.compute(
            self._arm_state(joint_positions), dt, left_target, right_target
        )

    def diagnostics(self) -> dict[str, str | float]:
        return self._solver.diagnostics()
