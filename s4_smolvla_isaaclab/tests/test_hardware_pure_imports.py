from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pink_hardware_entry_imports_without_isaac_torch_or_omni() -> None:
    code = """
import sys
import hardware_teleop.pink_main
forbidden = sorted({name.split('.')[0] for name in sys.modules} & {'torch', 'isaaclab', 'isaacsim', 'omni'})
if forbidden:
    raise SystemExit('forbidden imports: ' + ','.join(forbidden))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(ROOT), env.get("PYTHONPATH", "")) if value
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=10.0,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
