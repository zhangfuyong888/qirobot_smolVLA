#!/usr/bin/env bash
# Hardware teleop ROS2 / CycloneDDS environment (copy once, edit rarely).
#
# Setup:
#   cp hardware_teleop/config/ros_env.example.sh hardware_teleop/config/ros_env.sh
#   # edit HW_TELEOP_NETWORK_INTERFACE in ros_env.sh
#
# Manual use (e.g. ros2 topic echo in another terminal):
#   source hardware_teleop/scripts/source_ros_env.sh
#
# `bash run.sh teleop-hardware` sources this automatically.

: "${HW_TELEOP_ROS_DISTRO:=/opt/ros/humble}"
: "${HW_TELEOP_NETWORK_INTERFACE:=enp47s0}"
: "${HW_TELEOP_RMW_IMPLEMENTATION:=rmw_cyclonedds_cpp}"

_HW_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_HW_ROOT="$(cd "$_HW_CONFIG_DIR/.." && pwd)"

if [[ ! -f "$HW_TELEOP_ROS_DISTRO/setup.bash" ]]; then
    echo "[HW-TELEOP][ENV] ROS distro not found: $HW_TELEOP_ROS_DISTRO" >&2
    return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "$HW_TELEOP_ROS_DISTRO/setup.bash"

_HW_QI_INSTALL="$_HW_ROOT/ros_ws/install/setup.bash"
if [[ ! -f "$_HW_QI_INSTALL" ]]; then
    echo "[HW-TELEOP][ENV] qi messages not built. Run: bash run.sh teleop-hardware-build" >&2
    return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "$_HW_QI_INSTALL"

export RMW_IMPLEMENTATION="$HW_TELEOP_RMW_IMPLEMENTATION"
export CYCLONEDDS_URI="<CycloneDDS><Domain><General><Interfaces><NetworkInterface name=\"${HW_TELEOP_NETWORK_INTERFACE}\" priority=\"default\" multicast=\"default\" /></Interfaces></General></Domain></CycloneDDS>"

if [[ -f "$_HW_CONFIG_DIR/ros_env.local.sh" ]]; then
    # shellcheck disable=SC1091
    source "$_HW_CONFIG_DIR/ros_env.local.sh"
fi

export HW_TELEOP_ROS_ENV_READY=1
echo "[HW-TELEOP][ENV] ready ros=$HW_TELEOP_ROS_DISTRO interface=$HW_TELEOP_NETWORK_INTERFACE rmw=$RMW_IMPLEMENTATION"
