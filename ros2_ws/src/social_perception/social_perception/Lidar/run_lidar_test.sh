#!/bin/bash
# Quick simulation test: lidar detector + person mover, both with sim_params.
# Run from anywhere inside the container.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARAMS="$SCRIPT_DIR/sim_params.yaml"

source /opt/ros/jazzy/setup.bash
source /root/thesis_social_navigation_ws/install/setup.bash

echo "[lidar_test] Starting lidar_person_detector..."
python3 "$SCRIPT_DIR/lidar_person_detector.py" \
  --ros-args \
  --params-file "$PARAMS" \
  -p use_sim_time:=true &
LIDAR_PID=$!

sleep 2

echo "[lidar_test] Starting move_person_oneway..."
python3 /root/thesis_social_navigation_ws/src/social_perception/social_perception/move_person_oneway.py \
  --ros-args \
  -p use_sim_time:=true &
MOVER_PID=$!

echo "[lidar_test] Running — Ctrl+C to stop both."
trap "kill $LIDAR_PID $MOVER_PID 2>/dev/null; echo '[lidar_test] Stopped.'" INT TERM
wait $MOVER_PID
kill $LIDAR_PID 2>/dev/null
echo "[lidar_test] Person walk complete."
