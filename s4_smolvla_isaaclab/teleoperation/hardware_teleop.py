"""Reserved entry point for real-robot Quest teleoperation."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from hardware_teleop.config_loader import HardwareIkConfig
from teleoperation.config import TeleopConfig


class HardwareTeleopNotImplementedError(NotImplementedError):
    """Raised until a real-robot command/state bridge is wired."""


def run_hardware_teleop(config: TeleopConfig, args_cli: Any) -> None:
    """Launch real-robot Quest teleoperation via hardware_teleop/."""
    del config  # hardware entry loads its own merged config file.
    from hardware_teleop.config_loader import load_hardware_teleop_config
    from hardware_teleop.main import run_hardware_teleop as run_main

    hardware_config_path = getattr(args_cli, "hardware_config", None)
    if hardware_config_path is None:
        from pathlib import Path

        hardware_config_path = Path(__file__).resolve().parents[1] / "hardware_teleop/config/quest_hardware.yaml"
    hw_config = load_hardware_teleop_config(hardware_config_path)
    if getattr(args_cli, "ik_backend", None):
        hw_config = replace(hw_config, ik=HardwareIkConfig(backend=str(args_cli.ik_backend)))
    run_main(hw_config, args_cli)
