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

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
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

# Give Gazebo + Nav2 + RViz time to fully come up before starting
# the perception pipeline, which expects the simulation and TF
# tree to already be live.
sleep 15

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