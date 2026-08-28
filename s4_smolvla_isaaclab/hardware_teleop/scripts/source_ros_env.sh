#!/usr/bin/env bash
# Source hardware teleop ROS2 / DDS environment for the current shell.
set -eo pipefail

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
