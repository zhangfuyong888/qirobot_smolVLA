#!/usr/bin/env python
"""Validate FK determinism of a recorded Pink hardware shadow session."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from hardware_teleop.config_loader import load_hardware_teleop_config  # noqa: E402
from hardware_teleop.replay import validate_fk_replay  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("recording", type=Path)
    parser.add_argument(
        "--hardware-config",
        type=Path,
        default=PROJECT_ROOT / "hardware_teleop/config/quest_hardware.yaml",
    )
    args = parser.parse_args()
    config = load_hardware_teleop_config(args.hardware_config)
    result = validate_fk_replay(config, args.recording)
    print(
        f"[HW-PINK][REPLAY] frames={result.frames} "
        f"max_fk_position_error_m={result.max_fk_position_error_m:.9f} "
        f"max_fk_rotation_component_error={result.max_fk_rotation_component_error:.9f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
