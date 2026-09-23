#!/bin/bash
# run_sim_lidar_occupancy.sh
#
# Combined launcher: Gazebo/Nav2 bringup (run_sim.sh) + the lidar-vision
# fusion perception pipeline (run_lidar_test.sh) + the new gradient-
# ellipse social occupancy grid node (social_occupancy_grid_node_lidar.py),
# all wired to social_nav2_occupancy_sim.yaml instead of the ablation
# configs.
#
# This replaces run_sim.sh's own camera-only pipeline (yolo_detector,
# human_kf_predictor, predicted_person_cloud_node, corridor_yield_node)
# with the lidar+camera fusion pipeline, and feeds Nav2's social_layer
# from the occupancy grid node instead of a PointCloud2 voxel layer.
#
# Run this from inside the container, e.g.:
#   docker exec -it thesis_social_nav bash
#   ./run_sim_lidar_occupancy.sh
#
set -e

# ----------------------------------------------------------------
# ROS2 environment
# ----------------------------------------------------------------
source /opt/ros/jazzy/setup.bash
source /root/thesis_social_navigation_ws/install/setup.bash

# See run_sim.sh: CycloneDDS hits a participant-limit crash with this
# many nodes registered on domain 0 at once. Stick with the default RMW.
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0

# ----------------------------------------------------------------
# Sync custom worlds/models before launching anything (see run_sim.sh
# for why this is called explicitly rather than left to .bashrc).
# ----------------------------------------------------------------
/usr/local/bin/setup_gazebo_worlds.sh

WS=/root/thesis_social_navigation_ws
LIDAR_DIR="$WS/src/social_perception/social_perception/Lidar"
LIDAR_PARAMS="$LIDAR_DIR/sim_params.yaml"
NAV2_PARAMS="$WS/config/social_nav2_occupancy_sim.yaml"

cd "$WS"

# ----------------------------------------------------------------
# World / map selection. Override at launch time: WORLD=corridor_headon ./run_sim_lidar_occupancy.sh
# ----------------------------------------------------------------
WORLD="${WORLD:-corridor_headon}"
MAP_FILE="$WS/maps/${WORLD}.yaml"
if [ ! -f "$MAP_FILE" ]; then
  echo "[run_sim_lidar_occupancy] WARNING: no map for world '$WORLD', falling back to map_name.yaml"
  MAP_FILE="$WS/maps/map_name.yaml"
fi
BRIDGE_WORLD="$WORLD"

if [ "$WORLD" = "corridor_headon" ]; then
  SPAWN_X=-3.0; SPAWN_Y=0.6; SPAWN_YAW=3.14159
else
  SPAWN_X=0.0; SPAWN_Y=0.0; SPAWN_YAW=0.0
fi

PIDS=()
cleanup() {
  echo "[run_sim_lidar_occupancy] Stopping all nodes..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM EXIT

# ----------------------------------------------------------------
# Gazebo + TurtleBot4 + Nav2 + localization + RViz
# ----------------------------------------------------------------
ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py \
  world:="$WORLD" \
  x:="$SPAWN_X" \
  y:="$SPAWN_Y" \
  yaw:="$SPAWN_YAW" \
  slam:=false \
  nav2:=true \
  localization:=true \
  rviz:=true \
  map:="$MAP_FILE" \
  params_file:="$NAV2_PARAMS" \
  gz_args:="-r" &
PIDS+=($!)

# ----------------------------------------------------------------
# Wait for map_server, then for /map to actually have data
# (see run_sim.sh for the transient-local race this works around).
# ----------------------------------------------------------------
echo "[run_sim_lidar_occupancy] Waiting for /map_server to become active..."
for i in $(seq 1 60); do
  state=$(ros2 lifecycle get /map_server 2>/dev/null | awk '{print $1}')
  if [ "$state" = "active" ]; then
    echo "[run_sim_lidar_occupancy] /map_server is active."
    break
  fi
  sleep 1
done

echo "[run_sim_lidar_occupancy] Waiting for /map to have data..."
for i in $(seq 1 30); do
  if timeout 2 ros2 topic echo /map --once > /dev/null 2>&1; then
    echo "[run_sim_lidar_occupancy] /map has data."
    break
  fi
  sleep 1
done

echo "[run_sim_lidar_occupancy] Giving Nav2 lifecycle managers time to finish bringup..."
sleep 8

# NOTE: no auto /initialpose publish here - set the pose manually in RViz
# (2D Pose Estimate) before the perception pipeline starts below, otherwise
# AMCL's map->odom transform stays wrong and lidar_person_detector's
# static-map subtraction will misread the corridor walls as a person.

# ----------------------------------------------------------------
# ros_gz_bridge: expose Gazebo's set_pose service so move_person_*
# nodes can reposition simulated people.
# ----------------------------------------------------------------
ros2 run ros_gz_bridge parameter_bridge \
  "/world/${BRIDGE_WORLD}/set_pose@ros_gz_interfaces/srv/SetEntityPose" &
PIDS+=($!)
sleep 2

# Settle time before the perception pipeline starts querying TF.
sleep 5

# ----------------------------------------------------------------
# Person mover
# ----------------------------------------------------------------
LOG_DIR="$WS/logs"
mkdir -p "$LOG_DIR"
echo "[run_sim_lidar_occupancy] Node logs: $LOG_DIR/"

ros2 run social_perception move_person_gazebo2 --ros-args -p use_sim_time:=true -p world_name:="$WORLD" \
  2>&1 | tee "$LOG_DIR/mover.log" | sed 's/^/[mover] /' &
PIDS+=($!)
sleep 2

# ----------------------------------------------------------------
# Lidar-vision fusion perception pipeline (from run_lidar_test.sh)
# ----------------------------------------------------------------
echo "[run_sim_lidar_occupancy] 1/5 lidar_person_detector..."
python3 "$LIDAR_DIR/lidar_person_detector.py" \
  --ros-args --params-file "$LIDAR_PARAMS" -p use_sim_time:=true \
  2>&1 | tee "$LOG_DIR/lidar_detector.log" | sed 's/^/[lidar_detector] /' &
PIDS+=($!)
sleep 1

echo "[run_sim_lidar_occupancy] 2/5 yolo_detector_lidar..."
python3 "$LIDAR_DIR/yolo_detector_lidar.py" \
  --ros-args -p use_sim_time:=true \
  2>&1 | tee "$LOG_DIR/yolo_lidar.log" | sed 's/^/[yolo_lidar] /' &
PIDS+=($!)
sleep 1

echo "[run_sim_lidar_occupancy] 3/5 identity_fusion_node..."
python3 "$LIDAR_DIR/identity_fusion_node_lidar.py" \
  --ros-args --params-file "$LIDAR_PARAMS" -p use_sim_time:=true \
  2>&1 | tee "$LOG_DIR/fusion.log" | sed 's/^/[fusion] /' &
PIDS+=($!)
sleep 1

echo "[run_sim_lidar_occupancy] 4/5 human_kf_predictor..."
python3 "$LIDAR_DIR/human_kf_predictor_lidar.py" \
  --ros-args -p use_sim_time:=true \
  2>&1 | tee "$LOG_DIR/kf.log" | sed 's/^/[kf] /' &
PIDS+=($!)
sleep 1

echo "[run_sim_lidar_occupancy] 5/5 social_occupancy_grid_node..."
python3 "$LIDAR_DIR/social_occupancy_grid_node_lidar.py" \
  --ros-args -p use_sim_time:=true \
  2>&1 | tee "$LOG_DIR/occupancy.log" | sed 's/^/[occupancy] /' &
PIDS+=($!)

echo "[run_sim_lidar_occupancy] Pipeline running — Ctrl+C to stop."
echo "[run_sim_lidar_occupancy] Watch topics:"
echo "  /lidar_person_clusters      (lidar detections)"
echo "  /person_positions_map       (yolo detections)"
echo "  /person_positions_fused     (fused output -> SocialCritic)"
echo "  /predicted_person_positions (KF predictions)"
echo "  /social_cost_grid           (occupancy grid -> nav2 social_layer)"

wait
