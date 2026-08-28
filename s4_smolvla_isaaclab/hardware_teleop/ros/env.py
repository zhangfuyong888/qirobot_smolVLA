"""Resolve local ROS2 message workspace for hardware teleoperation."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_ROS_INSTALL = PROJECT_ROOT / "hardware_teleop" / "ros_ws" / "install" / "setup.bash"
BUILD_SCRIPT = PROJECT_ROOT / "hardware_teleop" / "scripts" / "build_ros_msgs.sh"


def local_ros_install_exists() -> bool:
    return LOCAL_ROS_INSTALL.is_file()


def local_ros_install_hint() -> str:
    return (
        "Build vendored qi ROS messages inside this repository:\n"
        f"  bash {BUILD_SCRIPT}\n"
        "Then source ROS2 Humble and the local overlay before teleop:\n"
        f"  source /opt/ros/humble/setup.bash\n"
        f"  source {LOCAL_ROS_INSTALL}"
    )
