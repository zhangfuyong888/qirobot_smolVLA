#!/bin/bash

set -e

DEPLOY_DIR="$HOME/project/qi_deploy"

echo "[INFO] Entering qi_deploy..."
cd "$DEPLOY_DIR"

echo "[INFO] Sourcing ROS environment..."
source install/setup.bash

echo "[INFO] Starting S2_homie_controller..."
ros2 launch rl_deploy_python S2_homie_controller
