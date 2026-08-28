"""Reserved entry point for real-robot Quest teleoperation."""

from __future__ import annotations

from typing import Any

from teleoperation.config import TeleopConfig


class HardwareTeleopNotImplementedError(NotImplementedError):
    """Raised until a real-robot command/state bridge is wired."""


def run_hardware_teleop(config: TeleopConfig, args_cli: Any) -> None:
    """Launch hardware teleoperation (not implemented).

    Future implementations should:
    - Reuse QuestWebServer + BimanualTeleopMapper from the simulation path
    - Implement TeleopRuntime against the robot SDK / ROS bridge
    - Read hardware.* settings from configs/teleoperation/meta_quest3.yaml
    """
    hardware_cfg = config.runtime.hardware
    interface = hardware_cfg.get("interface")
    raise HardwareTeleopNotImplementedError(
        "Real-robot teleoperation is not implemented yet. "
        "Set runtime.mode: simulation in configs/teleoperation/meta_quest3.yaml. "
        f"Reserved hardware.interface={interface!r}."
    )
