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


@runtime_checkable
class PureHardwareIkBackend(HardwareIkBackend, Protocol):
    def forward(self, arm_q14: np.ndarray) -> tuple[TcpPose, TcpPose]: ...


def create_pure_hardware_ik_backend(config: HardwareTeleopConfig) -> PureHardwareIkBackend:
    """Create an IK backend that has no Isaac articulation dependency."""
    backend = config.ik.backend.lower()
    if backend == "pink":
        from hardware_teleop.ik.pink_backend import PinkHardwareIkBackend

        return PinkHardwareIkBackend(config)
    raise ValueError(
        f"hardware IK backend {backend!r} requires the legacy Isaac entry; "
        "use backend='pink' for the pure hardware runtime"
    )


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
