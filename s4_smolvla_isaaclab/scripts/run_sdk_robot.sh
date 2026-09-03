#!/bin/bash

set -e

SDK_DIR="$HOME/nanshan_south/qi_sdk_internal"
SERVER_DIR="$SDK_DIR/build/examples/sn_loco_server"
NET_IF="wlp44s0"

echo "[INFO] Installing drivers..."
cd "$SDK_DIR"
sudo ./install_drivers.sh

echo "[INFO] Starting sn_loco_server..."
cd "$SERVER_DIR"
sudo ./sn_loco_server "$NET_IF"
