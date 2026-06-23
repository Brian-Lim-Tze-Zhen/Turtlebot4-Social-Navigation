#!/bin/bash
# run_sim.sh
#
# Launches the full thesis social-navigation simulation stack:
#   1. Gazebo + TurtleBot4 + Nav2 + localization + RViz, via
#      turtlebot4_gz_bringup's combined launch file.
#   2. The social_perception pipeline nodes (person mover,
#      YOLO detector, Kalman-filter predictor, predicted-cloud
#      publisher), launched separately afterward.
#
# Run this from inside the container, e.g.:
#   docker exec -it thesis_social_nav bash
#   ./run_sim.sh
#
set -e

# ----------------------------------------------------------------
# ROS2 environment
# ----------------------------------------------------------------
source /opt/ros/jazzy/setup.bash
source /root/thesis_social_navigation_ws/install/setup.bash

# THESIS FIX: removed RMW_IMPLEMENTATION=rmw_cyclonedds_cpp.
# CycloneDDS hit "Failed to find a free participant index for domain 0"
# once Gazebo + Nav2 + RViz + the 5 perception nodes all register as
# DDS participants simultaneously on domain 0, causing rviz2 to crash
# on startup (exit code -6). Reverting to the default RMW (FastDDS)
# avoids this participant-limit issue entirely.
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0

# ----------------------------------------------------------------
# THESIS FIX: sync custom worlds/models into turtlebot4_gz_bringup
# before launching anything. Called explicitly here (rather than
# relying solely on .bashrc) so this script works correctly even
# when invoked non-interactively, e.g. via:
#   docker exec thesis_social_nav bash -c "./run_sim.sh"
# .bashrc is NOT sourced for non-interactive `bash -c` invocations,
# so without this explicit call the custom world/person model can
# silently fail to resolve in that case.
# ----------------------------------------------------------------
/usr/local/bin/setup_gazebo_worlds.sh

cd /root/thesis_social_navigation_ws

# ----------------------------------------------------------------
# Gazebo + TurtleBot4 + Nav2 + localization + RViz
# (turtlebot4_gz_bringup handles Gazebo spawn, robot spawn, and
# the ros_gz bridge internally via this single launch file)
# ----------------------------------------------------------------
ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py \
  world:=two_human \
  slam:=false \
  nav2:=true \
  localization:=true \
  rviz:=true \
  map:=/root/thesis_social_navigation_ws/maps/map_name.yaml \
  params_file:=/root/thesis_social_navigation_ws/config/social_nav2.yaml \
  gz_args:="-r" &

# ----------------------------------------------------------------
# THESIS FIX: wait for /map_server to actually be ACTIVE, then for
# /map to actually have data, instead of a fixed `sleep 15` guess.
#
# Root cause this works around: when Gazebo + Nav2 + RViz all start
# together (vs. typed one-by-one manually with natural gaps), RViz's
# /map subscription can connect to map_server's publisher before
# map_server has published its first (transient-local/latched)
# message. TRANSIENT_LOCAL only helps late joiners if at least one
# publish has already happened - if RViz subscribes a moment too
# early, there is nothing latched yet to deliver, and RViz's Map
# display is left showing "No map received" with no further retries.
#
# Fix: poll map_server's lifecycle state until ACTIVE, then poll
# /map until a message can actually be echoed, then simply wait
# (see note below) rather than touching any node's lifecycle state
# ourselves.
# ----------------------------------------------------------------
echo "[run_sim] Waiting for /map_server to become active..."
for i in $(seq 1 60); do
  state=$(ros2 lifecycle get /map_server 2>/dev/null | awk '{print $1}')
  if [ "$state" = "active" ]; then
    echo "[run_sim] /map_server is active."
    break
  fi
  sleep 1
done

echo "[run_sim] Waiting for /map to have data..."
for i in $(seq 1 30); do
  if timeout 2 ros2 topic echo /map --once > /dev/null 2>&1; then
    echo "[run_sim] /map has data."
    break
  fi
  sleep 1
done

# ----------------------------------------------------------------
# THESIS NOTE: a previous version of this script manually cycled
# map_server via `ros2 lifecycle set /map_server deactivate/activate`
# here, to force a fresh publish for any subscriber (e.g. RViz) that
# might have subscribed just before the first publish.
#
# REMOVED: this directly raced against Nav2's own lifecycle_manager,
# which is simultaneously driving map_server (and every other Nav2
# node) through its own startup transitions as part of normal
# bringup. Two actors changing the same node's lifecycle state at
# once caused lifecycle_manager to lose the race ("Failed to change
# state for node: map_server"), which made it abort bringing up
# every other node in that manager's list (controller_server,
# planner_server, bt_navigator, behavior_server, etc.) - explaining
# why Nav2 goals were being rejected even though the map itself
# displayed correctly afterward.
#
# Instead: just wait for the map data check above to succeed, then
# give the lifecycle managers extra uninterrupted time to finish
# their own bringup before anything else touches the graph.
# ----------------------------------------------------------------
echo "[run_sim] Giving Nav2 lifecycle managers time to finish bringup..."
sleep 8

# A little extra settle time before starting the perception pipeline.
# This also helps avoid "Lookup would require extrapolation into the
# past" TF warnings seen when perception nodes start querying
# map->camera transforms before TF has buffered enough history yet.
sleep 5

# ----------------------------------------------------------------
# Social perception pipeline (separate nodes, run after sim is up)
#
# All nodes pass use_sim_time:=true so they use Gazebo's simulated
# clock instead of the wall clock - required for correct timestamp
# sync with /odom, /clock, and the simulated camera/detections.
# ----------------------------------------------------------------

# Moves the simulated person(s) in Gazebo along their motion pattern
ros2 run social_perception move_person_gazebo2 --ros-args -p use_sim_time:=true &
sleep 2

# YOLO + ByteTrack person detection from the robot's camera
ros2 run social_perception yolo_detector --ros-args -p use_sim_time:=true &
sleep 2

# Kalman-filter velocity/position predictor
ros2 run social_perception human_kf_predictor --ros-args -p use_sim_time:=true &
sleep 2

# RViz marker visualizer for the predicted position (the "red sphere")
ros2 run social_perception prediction_marker_node --ros-args -p use_sim_time:=true &
sleep 2

# Publishes the predicted-position obstacle/risk-zone point cloud
# for the Nav2 local costmap
ros2 run social_perception predicted_person_cloud_node --ros-args -p use_sim_time:=true &

wait