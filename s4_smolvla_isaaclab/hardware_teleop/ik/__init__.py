"""Hardware IK backend protocol and factory."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from hardware_teleop.config_loader import HardwareTeleopConfig
from teleoperation.mapping import TcpPose


@runtime_checkable
class HardwareIkBackend(Protocol):
    name: str

    def set_posture_reference(self, joint_positions: np.ndarray) -> None: ...

    def compute(
        self,
        joint_positions: np.ndarray,
        dt: float,
        left_target: TcpPose,
        right_target: TcpPose,
    ) -> np.ndarray: ...

    def diagnostics(self) -> dict[str, str | float]: ...


def create_hardware_ik_backend(
    config: HardwareTeleopConfig,
    robot: Any,
    device: str,
    base_body_id: int,
) -> HardwareIkBackend:
    backend = config.ik.backend.lower()
    if backend == "rmpflow":
        from hardware_teleop.ik.rmpflow_headless import HeadlessRmpFlowIkBackend

        return HeadlessRmpFlowIkBackend(config, robot, device, base_body_id)
    if backend == "pinocchio":
        from hardware_teleop.ik.pinocchio_backend import PinocchioHardwareIkBackend

        return PinocchioHardwareIkBackend(config, robot, device)
    raise ValueError(f"unsupported hardware IK backend: {backend!r}")
