"""Teleoperation runtime selection (simulation today, hardware reserved)."""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol, runtime_checkable

import numpy as np

from teleoperation.config import TeleopConfig


class TeleopRuntimeMode(str, Enum):
    SIMULATION = "simulation"
    HARDWARE = "hardware"


@runtime_checkable
class TeleopRuntime(Protocol):
    """Bridge Quest controller input to robot commands.

  Implementations:
  - IsaacLab simulation: `teleoperation.isaaclab_teleop.run`
  - Real robot (reserved): `teleoperation.hardware_teleop.run_hardware_teleop`
    """

    def read_bimanual_state(self) -> np.ndarray:
        """Return the current 26D bimanual joint state."""

    def apply_bimanual_command(self, command_action: np.ndarray) -> None:
        """Apply a 26D absolute joint target."""

    def step(self, *, render: bool) -> None:
        """Advance one control cycle (physics/render for simulation)."""

    def close(self) -> None:
        """Release resources."""


def resolve_runtime_mode(config: TeleopConfig) -> TeleopRuntimeMode:
    mode = str(config.runtime.mode).lower()
    if mode == TeleopRuntimeMode.SIMULATION.value:
        return TeleopRuntimeMode.SIMULATION
    if mode == TeleopRuntimeMode.HARDWARE.value:
        return TeleopRuntimeMode.HARDWARE
    raise ValueError(f"runtime.mode must be 'simulation' or 'hardware', got {mode!r}")


def launch_teleop(config: TeleopConfig, args_cli: Any) -> None:
    """Dispatch to the configured teleoperation runtime."""
    mode = resolve_runtime_mode(config)
    if mode == TeleopRuntimeMode.HARDWARE:
        from teleoperation.hardware_teleop import run_hardware_teleop

        run_hardware_teleop(config, args_cli)
        return
    from teleoperation.isaaclab_teleop import run_simulation_teleop

    run_simulation_teleop(config, args_cli)
