from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_common_and_robot_primitives_do_not_import_ml_or_ros() -> None:
    code = """
import sys
import real_vla_stack.common.config
import real_vla_stack.common.contract
import real_vla_stack.common.protocol
import real_vla_stack.robot.rollout.action_buffer
import real_vla_stack.robot.rollout.safety
forbidden = sorted({name.split('.')[0] for name in sys.modules} & {'torch', 'lerobot', 'rclpy'})
if forbidden:
    raise SystemExit('forbidden imports: ' + ','.join(forbidden))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT), env.get("PYTHONPATH", "")))
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
