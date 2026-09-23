#!/bin/bash
# run_sim_perception.sh
#
# Part 2/2 of the split run_sim_lidar_occupancy.sh: the person mover,
# the lidar-vision fusion perception pipeline, and the social occupancy
# grid node. Assumes run_sim_bringup.sh is already running in another
# shell (Gazebo + Nav2 + RViz up) and that you have set the initial
# pose in RViz (2D Pose Estimate) - this script does NOT auto-publish
# one; see run_sim_lidar_occupancy.sh's history for why that was
# removed (fragile YAML-over-CLI message construction).
#
# Run this from inside the container, in a SEPARATE shell from
# run_sim_bringup.sh:
#   docker exec -it thesis_social_nav bash
#   ./run_sim_perception.sh
#
set -e

# ----------------------------------------------------------------
# ROS2 environment
# ----------------------------------------------------------------
source /opt/ros/jazzy/setup.bash
source /root/thesis_social_navigation_ws/install/setup.bash

export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0

WS=/root/thesis_social_navigation_ws
LIDAR_DIR="$WS/src/social_perception/social_perception/Lidar"
LIDAR_PARAMS="$LIDAR_DIR/sim_params.yaml"

cd "$WS"

WORLD="${WORLD:-corridor_headon}"
BRIDGE_WORLD="$WORLD"

if [ "$WORLD" = "corridor_headon" ]; then
  SPAWN_X=-3.0; SPAWN_Y=0.6; SPAWN_YAW=3.14159
else
  SPAWN_X=0.0; SPAWN_Y=0.0; SPAWN_YAW=0.0
fi

PIDS=()
cleanup() {
  echo "[run_sim_perception] Stopping perception pipeline..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM EXIT

# ----------------------------------------------------------------
# Wait for map_server to be active and /map to have data, in case
# this script is started before run_sim_bringup.sh has finished.
# ----------------------------------------------------------------
echo "[run_sim_perception] Waiting for /map_server to become active..."
for i in $(seq 1 60); do
  state=$(ros2 lifecycle get /map_server 2>/dev/null | awk '{print $1}')
  if [ "$state" = "active" ]; then
    echo "[run_sim_perception] /map_server is active."
    break
  fi
  sleep 1
done

echo "[run_sim_perception] Waiting for /map to have data..."
for i in $(seq 1 30); do
  if timeout 2 ros2 topic echo /map --once > /dev/null 2>&1; then
    echo "[run_sim_perception] /map has data."
    break
  fi
  sleep 1
done

# ----------------------------------------------------------------
# THESIS FIX (silent localization failure): a bag-recorded debug run
# showed map->odom NEVER appeared on /tf or /tf_static for the entire
# 4-minute run (only odom->base_link did) - AMCL was never given an
# initial pose, so it never localized or broadcast the transform at
# all. Everything downstream kept running without erroring loudly:
# lidar_person_detector's "map" frame TF lookups were operating on a
# stale/absent transform the whole time, silently producing garbage
# clusters instead of failing outright. This publishes /initialpose
# automatically at the known spawn point, removing the dependency on
# someone remembering to click 2D Pose Estimate in RViz.
#
# Message field order matters here: `covariance` must come BEFORE the
# nested `pose` field in the flow-style YAML, or `ros2 topic pub`
# fails with "The passed value needs to be in YAML string or a
# dictionary" - confirmed by direct testing, not documented anywhere.
# ----------------------------------------------------------------
# THESIS FIX (race: AMCL only subscribes to /initialpose once ACTIVE,
# not while Configuring - a one-shot `-1` publish sent before that
# transition finishes goes out with zero subscribers and is lost with
# no retry). Wait for AMCL's own lifecycle state, not a fixed sleep.
echo "[run_sim_perception] Waiting for /amcl to become active..."
for i in $(seq 1 30); do
  state=$(ros2 lifecycle get /amcl 2>/dev/null | awk '{print $1}')
  if [ "$state" = "active" ]; then
    echo "[run_sim_perception] /amcl is active."
    break
  fi
  sleep 1
done

# THESIS FIX: "active" only means the lifecycle STATE transitioned, not
# that AMCL's own odom subscription has accumulated enough TF history
# yet. Every automated publish attempt logged "Failed to transform
# initial pose in time (extrapolation into the future...)" - the pose
# message's timestamp was consistently a few ms ahead of what AMCL's TF
# buffer had seen so far. A human clicking 2D Pose Estimate in RViz
# naturally happens later/slower than an immediate scripted publish;
# give AMCL's buffer time to actually warm up before publishing.
echo "[run_sim_perception] Letting AMCL's TF buffer warm up..."
sleep 5

echo "[run_sim_perception] Publishing initial pose to AMCL..."
INITIALPOSE_MSG=$(python3 -c "
import math
yaw = $SPAWN_YAW
qz = math.sin(yaw / 2.0)
qw = math.cos(yaw / 2.0)
print(f'{{header: {{frame_id: \"map\"}}, pose: {{covariance: [0.25,0,0,0,0,0, 0,0.25,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0.06853891945200942], pose: {{position: {{x: $SPAWN_X, y: $SPAWN_Y, z: 0.0}}, orientation: {{z: {qz}, w: {qw}}}}}}}}}')")
# THESIS FIX: publish a few times over ~1s instead of a single -1 shot,
# in case the subscriber connection is still settling right after the
# lifecycle transition (discovery handshake lag, not just activation).
ros2 topic pub -1 /initialpose geometry_msgs/msg/PoseWithCovarianceStamped "$INITIALPOSE_MSG" > /dev/null
sleep 2

# Verify it actually took: AMCL should now be broadcasting map->odom.
# If not, warn loudly instead of silently continuing into a broken run.
echo "[run_sim_perception] Verifying AMCL localized (map->odom transform present)..."
LOCALIZED=false
for i in $(seq 1 10); do
  if timeout 2 ros2 run tf2_ros tf2_echo map odom > /dev/null 2>&1; then
    LOCALIZED=true
    break
  fi
  sleep 1
done
if [ "$LOCALIZED" = true ]; then
  echo "[run_sim_perception] AMCL localized - map->odom transform confirmed."
else
  echo "[run_sim_perception] WARNING: map->odom transform NOT found after 10s."
  echo "[run_sim_perception] WARNING: AMCL did not localize - detections will be garbage."
  echo "[run_sim_perception] WARNING: set the pose manually in RViz (2D Pose Estimate) now."
fi

# ----------------------------------------------------------------
# ros_gz_bridge: expose Gazebo's set_pose service so move_person_*
# nodes can reposition simulated people.
# ----------------------------------------------------------------
ros2 run ros_gz_bridge parameter_bridge \
  "/world/${BRIDGE_WORLD}/set_pose@ros_gz_interfaces/srv/SetEntityPose" &
PIDS+=($!)
sleep 2

LOG_DIR="$WS/logs"
mkdir -p "$LOG_DIR"
echo "[run_sim_perception] Node logs: $LOG_DIR/"

# ----------------------------------------------------------------
# Person mover
# ----------------------------------------------------------------
ros2 run social_perception move_person_gazebo2 --ros-args -p use_sim_time:=true -p world_name:="$WORLD" \
  2>&1 | tee "$LOG_DIR/mover.log" | sed 's/^/[mover] /' &
PIDS+=($!)
sleep 2

# ----------------------------------------------------------------
# Lidar-vision fusion perception pipeline
# ----------------------------------------------------------------
echo "[run_sim_perception] 1/5 lidar_person_detector..."
python3 "$LIDAR_DIR/lidar_person_detector.py" \
  --ros-args --params-file "$LIDAR_PARAMS" -p use_sim_time:=true \
  2>&1 | tee "$LOG_DIR/lidar_detector.log" | sed 's/^/[lidar_detector] /' &
PIDS+=($!)
sleep 1

echo "[run_sim_perception] 2/5 yolo_detector_lidar..."
python3 "$LIDAR_DIR/yolo_detector_lidar.py" \
  --ros-args -p use_sim_time:=true \
  2>&1 | tee "$LOG_DIR/yolo_lidar.log" | sed 's/^/[yolo_lidar] /' &
PIDS+=($!)
sleep 1

echo "[run_sim_perception] 3/5 identity_fusion_node..."
python3 "$LIDAR_DIR/identity_fusion_node_lidar.py" \
  --ros-args --params-file "$LIDAR_PARAMS" -p use_sim_time:=true \
  2>&1 | tee "$LOG_DIR/fusion.log" | sed 's/^/[fusion] /' &
PIDS+=($!)
sleep 1

echo "[run_sim_perception] 4/5 human_kf_predictor..."
python3 "$LIDAR_DIR/human_kf_predictor_lidar.py" \
  --ros-args -p use_sim_time:=true \
  2>&1 | tee "$LOG_DIR/kf.log" | sed 's/^/[kf] /' &
PIDS+=($!)
sleep 1

echo "[run_sim_perception] 5/5 social_occupancy_grid_node..."
python3 "$LIDAR_DIR/social_occupancy_grid_node_lidar.py" \
  --ros-args -p use_sim_time:=true \
  2>&1 | tee "$LOG_DIR/occupancy.log" | sed 's/^/[occupancy] /' &
PIDS+=($!)

echo "[run_sim_perception] Pipeline running — Ctrl+C to stop."
echo "[run_sim_perception] Watch topics:"
echo "  /lidar_person_clusters      (lidar detections)"
echo "  /person_positions_map       (yolo detections)"
echo "  /person_positions_fused     (fused output -> SocialCritic)"
echo "  /predicted_person_positions (KF predictions)"
echo "  /social_cost_grid           (occupancy grid -> nav2 social_layer)"

wait
