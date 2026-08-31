"""Arm-controller backends used only by interactive teleoperation."""

from __future__ import annotations

from typing import Any

from teleoperation.config import TeleopConfig


def create_arm_controller(
    backend: str,
    config: TeleopConfig,
    robot: Any,
    device: str,
    base_body_id: int,
):
    """Create the requested teleoperation controller without changing pipeline controllers."""
    selected = backend.lower()
    if selected == "pinocchio":
        from teleoperation.controllers.pinocchio_backend import PinocchioTeleopController

        return PinocchioTeleopController(robot, device, config)
    if selected == "pink":
        from teleoperation.controllers.pink_backend import PinkTeleopController

        return PinkTeleopController(robot, config)
    if selected == "rmpflow":
        from teleoperation.controllers.rmpflow_backend import BimanualRmpFlowController

        return BimanualRmpFlowController(robot, device, base_body_id, config.controller.rmpflow)
    raise ValueError(f"Unsupported teleoperation controller backend: {backend!r}")


__all__ = ["create_arm_controller"]
