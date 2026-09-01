#!/usr/bin/env bash
# Source hardware teleop ROS2 / DDS environment for the current shell.
set -eo pipefail

# ROS2 Humble's generated setup scripts probe optional variables without
# nounset-safe expansion. run.sh uses `set -u`, so suspend it only while the
# ROS environment is being sourced, then restore the caller's setting.
_HW_TELEOP_RESTORE_NOUNSET=0
if [[ $- == *u* ]]; then
    _HW_TELEOP_RESTORE_NOUNSET=1
    set +u
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="$PROJECT_ROOT/hardware_teleop/config/ros_env.sh"
ENV_EXAMPLE="$PROJECT_ROOT/hardware_teleop/config/ros_env.example.sh"

if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1091
    source "$ENV_FILE"
elif [[ -f "$ENV_EXAMPLE" ]]; then
    echo "[HW-TELEOP][ENV] ros_env.sh missing; using example defaults." >&2
    echo "[HW-TELEOP][ENV] copy: cp hardware_teleop/config/ros_env.example.sh hardware_teleop/config/ros_env.sh" >&2
    # shellcheck disable=SC1091
    source "$ENV_EXAMPLE"
else
    echo "[HW-TELEOP][ENV] missing $ENV_FILE" >&2
    exit 1
fi

if [[ "$_HW_TELEOP_RESTORE_NOUNSET" -eq 1 ]]; then
    set -u
fi
unset _HW_TELEOP_RESTORE_NOUNSET
