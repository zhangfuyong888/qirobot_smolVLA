#!/usr/bin/env bash
# Copy this file to ros_env.local.sh on the S4 robot computer. The destination
# is gitignored and does not modify qiling_s4 or the SDK configuration.

# The inspected robot runs the SDK and Pink process on the same computer. Its
# installed SDK DDS config uses loopback and ROS domain 16.
export HW_TELEOP_NETWORK_INTERFACE=lo
export ROS_DOMAIN_ID=16
export HW_TELEOP_RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# Direct /usr/bin/python3.10 runtime; no Conda or virtual environment.
export S4_HW_TELEOP_RUNTIME=system
export S4_HW_TELEOP_SYSTEM_PYTHON=/usr/bin/python3
# The inspected robot currently has the exact lightweight packages in
# ~/.local. Runtime version checks and the ROS Pinocchio path check remain
# mandatory. Set this to 0 after using the project-local installer.
export S4_HW_TELEOP_ALLOW_USER_SITE=1

# Strict doctor expectations for this robot profile.
export HW_TELEOP_EXPECT_PIN_PREFIX=/opt/ros/humble
export HW_TELEOP_EXPECT_ROS_DOMAIN_ID=16
export HW_TELEOP_EXPECT_DDS_INTERFACE=lo
