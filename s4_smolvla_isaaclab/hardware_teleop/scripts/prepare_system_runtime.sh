#!/usr/bin/env bash
# Prepare project-local wheels for the ROS2 Humble system Python runtime.
#
# Nothing is installed unless --install is passed. Packages are written only
# to the ignored project directory .local/hardware_python, never to /usr or
# ~/.local.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${S4_HW_TELEOP_SYSTEM_PYTHON:-/usr/bin/python3}"
TARGET="${S4_HW_TELEOP_SITE_PACKAGES:-$PROJECT_ROOT/.local/hardware_python}"
MODE="check"

usage() {
    echo "Usage: bash run.sh teleop-hardware-system-prepare [--check|--install]" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check) MODE="check" ;;
        --install) MODE="install" ;;
        -h|--help) usage; exit 0 ;;
        *) usage; echo "Unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[HW-PINK][SYSTEM] Python is missing: $PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -f /opt/ros/humble/setup.bash ]]; then
    echo "[HW-PINK][SYSTEM] ROS2 Humble is missing: /opt/ros/humble/setup.bash" >&2
    exit 1
fi

set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
set -u

"$PYTHON_BIN" - <<'PY'
import platform
import sys

if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"ROS2 Humble requires Python 3.10, got {sys.version.split()[0]}")
if platform.machine() != "x86_64":
    raise SystemExit(
        f"prebuilt system-runtime wheels are validated only on x86_64, got {platform.machine()}"
    )
import numpy
import pinocchio

pin_path = str(pinocchio.__file__)
if not pin_path.startswith("/opt/ros/humble/"):
    raise SystemExit(
        "system runtime must load ROS Pinocchio from /opt/ros/humble, got " + pin_path
    )
print(
    f"[HW-PINK][SYSTEM] base_ok python={sys.version.split()[0]} "
    f"numpy={numpy.__version__} pinocchio={pinocchio.__version__} path={pin_path}"
)
PY

echo "[HW-PINK][SYSTEM] target=$TARGET"
echo "[HW-PINK][SYSTEM] packages: scipy=1.15.2 aiohttp=3.14.3 qpsolvers=4.12.0 daqp=0.8.7 quadprog=0.1.13"

if [[ "$MODE" == "install" ]]; then
    mkdir -p "$TARGET"
    # aiohttp has pure networking dependencies that are also kept in TARGET.
    "$PYTHON_BIN" -m pip install \
        --only-binary=:all: --upgrade --target "$TARGET" \
        "aiohttp==3.14.3"
    # Keep the robot's NumPy and ROS Pinocchio. --no-deps prevents pip from
    # copying or upgrading either package in this project-local directory.
    "$PYTHON_BIN" -m pip install \
        --only-binary=:all: --upgrade --no-deps --target "$TARGET" \
        "scipy==1.15.2" \
        "qpsolvers==4.12.0" \
        "daqp==0.8.7" \
        "quadprog==0.1.13"
fi

if [[ ! -d "$TARGET" ]]; then
    echo "[HW-PINK][SYSTEM] local packages are not installed yet." >&2
    echo "[HW-PINK][SYSTEM] after approval run: bash run.sh teleop-hardware-system-prepare --install" >&2
    exit 1
fi

PYTHONPATH="$TARGET:${PYTHONPATH:-}" "$PYTHON_BIN" - <<'PY'
import aiohttp
import daqp
import importlib.metadata
import numpy as np
import pinocchio
import qpsolvers
import quadprog
import scipy

expected = {
    "aiohttp": (aiohttp.__version__, "3.14.3"),
    "scipy": (scipy.__version__, "1.15.2"),
    "qpsolvers": (qpsolvers.__version__, "4.12.0"),
    "daqp": (importlib.metadata.version("daqp"), "0.8.7"),
    "quadprog": (importlib.metadata.version("quadprog"), "0.1.13"),
}
for name, (actual, wanted) in expected.items():
    if actual != wanted:
        raise SystemExit(f"{name} version mismatch: expected {wanted}, got {actual}")
if not str(pinocchio.__file__).startswith("/opt/ros/humble/"):
    raise SystemExit(f"wrong Pinocchio selected: {pinocchio.__file__}")
for solver in ("quadprog", "daqp"):
    if solver not in qpsolvers.available_solvers:
        raise SystemExit(f"QP solver is unavailable: {solver}")
result = qpsolvers.solve_qp(
    np.eye(2), -np.ones(2), G=np.eye(2), h=np.ones(2), solver="quadprog"
)
if result is None or not np.isfinite(result).all():
    raise SystemExit(f"quadprog probe failed: {result}")
print(
    f"[HW-PINK][SYSTEM] PASS numpy={np.__version__} scipy={scipy.__version__} "
    f"pinocchio={pinocchio.__version__} qpsolvers={qpsolvers.__version__} "
    f"solvers={qpsolvers.available_solvers}"
)
PY
