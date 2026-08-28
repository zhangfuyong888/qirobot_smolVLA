#!/usr/bin/env bash
# Build vendored qi ROS2 messages inside this repository.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WS="$PROJECT_ROOT/hardware_teleop/ros_ws"

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
    echo "ROS2 Humble is required at /opt/ros/humble/setup.bash" >&2
    exit 2
fi

# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash

cd "$WS"
colcon build --packages-select qi --symlink-install
echo "[OK] built local qi messages: $WS/install/setup.bash"
