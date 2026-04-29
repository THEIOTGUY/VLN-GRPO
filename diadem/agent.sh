#!/usr/bin/env bash
# Start the micro-ROS agent — bridges the Diadem ESP32 to ROS2.
# Run this in a dedicated terminal and keep it open.
#
# Usage:
#   bash diadem/agent.sh

unset PYTHONPATH PYTHONHOME
source /home/chitti/uros_ws/install/setup.bash
export ROS_DOMAIN_ID=101

pkill -9 -f micro_ros_agent 2>/dev/null || true
sleep 1

echo "[agent] Starting micro_ros_agent on /dev/ttyUSB0 @ 921600 baud..."
echo "[agent] ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "[agent] Waiting for ESP32 'session established'..."
echo ""

exec /home/chitti/uros_ws/install/micro_ros_agent/lib/micro_ros_agent/micro_ros_agent \
  serial --dev /dev/ttyUSB0 -b 921600
