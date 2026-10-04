#!/bin/bash
# hw_pipeline_replay.sh <bag name under bags/hw_pipeline_replay/in>
#
# Replays the SENSOR topics of a hardware bag (camera, LiDAR, TF, odometry)
# and runs the robot's current perception scripts on them: YOLO pose
# detector -> camera-ray person node -> camera-ray identity node. The scripts
# are COPIES of the hardware repo's scripts/ (bags/hw_pipeline_replay/scripts),
# started with the arguments of the hardware launch file. Their outputs go to
# bags/hw_pipeline_replay/out/<bag>.csv. Nothing is written to the hardware repo.
#
# Run inside the sim container, with nothing else running.
set -e
NAME=$1
WS=/root/thesis_social_navigation_ws
R=$WS/bags/hw_pipeline_replay
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=0
pgrep -f "^gz sim" > /dev/null && { echo "ERROR: a simulation is running" >&2; exit 1; }
mkdir -p "$R/out" "$R/log/$NAME"
PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill -TERM -- "-$p" 2>/dev/null || true; done; sleep 2
            for p in "${PIDS[@]}"; do kill -KILL -- "-$p" 2>/dev/null || true; done; }
trap cleanup EXIT
start() { local log=$1; shift; setsid "$@" > "$R/log/$NAME/$log.log" 2>&1 & PIDS+=($!); }
TF="-r /tf:=/turtlebot4/tf -r /tf_static:=/turtlebot4/tf_static"

start yolo python3 "$R/scripts/yolo_leg_detector_lidar.py" --ros-args -p use_sim_time:=true \
  -r /oakd/rgb/preview/image_raw/compressed:=/turtlebot4/oakd/rgb/preview/image_raw/compressed
start ray_person python3 "$R/scripts/camera_ray_person_node.py" --ros-args -p use_sim_time:=true \
  -p scan_topic:=/turtlebot4/scan -p camera_yaw_offset_deg:=2.25 -p ray_half_width_deg:=6.0 $TF
start ray_identity python3 "$R/scripts/camera_ray_identity_node.py" --ros-args -p use_sim_time:=true \
  -p scan_topic:=/turtlebot4/scan $TF
start logger python3 "$WS/analysis/hw_pipeline_logger.py" "$R/out/$NAME.csv" --ros-args \
  -p use_sim_time:=true $TF
sleep 15    # YOLO model load

ros2 bag play "$R/in/$NAME" --clock 50 --read-ahead-queue-size 2000 --topics \
  /turtlebot4/oakd/rgb/preview/image_raw/compressed /turtlebot4/oakd/rgb/preview/camera_info \
  /turtlebot4/scan /turtlebot4/tf /turtlebot4/tf_static /turtlebot4/odom /turtlebot4/map \
  > "$R/log/$NAME/play.log" 2>&1
sleep 2
echo "$NAME: $(grep -c . "$R/out/$NAME.csv") output messages"
