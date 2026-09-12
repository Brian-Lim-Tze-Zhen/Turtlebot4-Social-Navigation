#!/bin/bash
# Full Lidar pipeline test in simulation.
# Launches all six nodes then the person mover, waits for the walk to finish,
# then kills everything cleanly.
#
# Prerequisites: run_sim.sh must already be running (Gazebo + Nav2 + RViz up,
# 2D Pose Estimate set in RViz so /odom and /map TF are connected).
#
# Usage:
#   bash run_lidar_test.sh
#
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS=/root/thesis_social_navigation_ws
PARAMS="$SCRIPT_DIR/sim_params.yaml"

source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash"

PIDS=()
cleanup() {
  echo "[lidar_test] Stopping all nodes..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  echo "[lidar_test] Done."
}
trap cleanup INT TERM EXIT

echo "[lidar_test] 1/5 lidar_person_detector..."
python3 "$SCRIPT_DIR/lidar_person_detector.py" \
  --ros-args --params-file "$PARAMS" -p use_sim_time:=true &
PIDS+=($!)
sleep 1

echo "[lidar_test] 2/5 yolo_detector_lidar..."
python3 "$SCRIPT_DIR/yolo_detector_lidar.py" \
  --ros-args -p use_sim_time:=true &
PIDS+=($!)
sleep 1

echo "[lidar_test] 3/5 identity_fusion_node..."
python3 "$SCRIPT_DIR/identity_fusion_node_lidar.py" \
  --ros-args --params-file "$PARAMS" -p use_sim_time:=true &
PIDS+=($!)
sleep 1

echo "[lidar_test] 4/5 human_kf_predictor..."
python3 "$SCRIPT_DIR/human_kf_predictor_lidar.py" \
  --ros-args -p use_sim_time:=true &
PIDS+=($!)
sleep 1

echo "[lidar_test] 5/5 predicted_person_cloud_node..."
python3 "$SCRIPT_DIR/predicted_person_cloud_node_lidar.py" \
  --ros-args -p use_sim_time:=true &
PIDS+=($!)
sleep 2

echo "[lidar_test] Starting person mover..."
python3 "$WS/src/social_perception/social_perception/move_person_oneway.py" \
  --ros-args -p use_sim_time:=true &
MOVER_PID=$!
PIDS+=($MOVER_PID)

echo "[lidar_test] Pipeline running — Ctrl+C to stop."
echo "[lidar_test] Watch topics:"
echo "  /lidar_person_clusters     (lidar detections)"
echo "  /person_positions_map      (yolo detections)"
echo "  /person_positions_fused    (fused output)"
echo "  /predicted_person_positions (KF predictions)"
echo "  /predicted_person_cloud    (costmap input)"

wait $MOVER_PID
echo "[lidar_test] Person walk complete."
