#!/bin/bash
# run_sim_bringup.sh
#
# Part 1/2 of the split run_sim_lidar_occupancy.sh: Gazebo + TurtleBot4 +
# Nav2 + localization + RViz only. No perception pipeline, no person
# mover - run run_sim_perception.sh separately once this is up and you
# have set the initial pose in RViz (2D Pose Estimate).
#
# Run this from inside the container, e.g.:
#   docker exec -it thesis_social_nav bash
#   ./run_sim_bringup.sh
#
set -e

# ----------------------------------------------------------------
# ROS2 environment
# ----------------------------------------------------------------
source /opt/ros/jazzy/setup.bash
source /root/thesis_social_navigation_ws/install/setup.bash

export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0

# ----------------------------------------------------------------
# Sync custom worlds/models before launching anything.
# ----------------------------------------------------------------
/usr/local/bin/setup_gazebo_worlds.sh

WS=/root/thesis_social_navigation_ws
NAV2_PARAMS="$WS/config/social_nav2_occupancy_sim.yaml"

cd "$WS"

# ----------------------------------------------------------------
# World / map selection. Override at launch time: WORLD=corridor_headon ./run_sim_bringup.sh
# ----------------------------------------------------------------
WORLD="${WORLD:-corridor_headon}"
MAP_FILE="$WS/maps/${WORLD}.yaml"
if [ ! -f "$MAP_FILE" ]; then
  echo "[run_sim_bringup] WARNING: no map for world '$WORLD', falling back to map_name.yaml"
  MAP_FILE="$WS/maps/map_name.yaml"
fi

if [ "$WORLD" = "corridor_headon" ]; then
  SPAWN_X=-3.0; SPAWN_Y=0.6; SPAWN_YAW=3.14159
else
  SPAWN_X=0.0; SPAWN_Y=0.0; SPAWN_YAW=0.0
fi

echo "[run_sim_bringup] World: $WORLD, spawn=($SPAWN_X, $SPAWN_Y, $SPAWN_YAW)"
echo "[run_sim_bringup] Once RViz is up, set the initial pose (2D Pose Estimate) at the spawn point above,"
echo "[run_sim_bringup] then run ./run_sim_perception.sh WORLD=$WORLD in another shell."

# ----------------------------------------------------------------
# Gazebo + TurtleBot4 + Nav2 + localization + RViz
# Runs in the foreground - Ctrl+C here stops the whole bringup.
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
  gz_args:="-r"
