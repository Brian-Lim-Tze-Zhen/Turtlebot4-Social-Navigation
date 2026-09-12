#!/bin/bash
# setup_gazebo_worlds.sh
#
# THESIS FIX: turtlebot4_gz_bringup's launch file resolves
# world:=<name> as a bare filename relative to its own
# installed worlds/ directory, and does not reliably honor
# GZ_SIM_RESOURCE_PATH for nested model:// URIs when Gazebo
# is spawned via `ros2 launch` (confirmed by testing: the
# same env var works for a standalone `gz sim` call, but
# fails for the exact same world/model under `ros2 launch`).
#
# This script copies our custom world file and any custom
# models into turtlebot4_gz_bringup's own worlds/ directory,
# where lookup is known to work reliably. It must run at
# CONTAINER STARTUP (not image build time), because the
# source files only exist via the docker-compose bind mount
# (../ros2_ws:/root/thesis_social_navigation_ws), which is
# not present during `docker build`.
#
# Safe to re-run: uses cp -f, just overwrites with the latest
# version from the workspace every time the container starts.

set -e

WORKSPACE_WORLDS="/root/thesis_social_navigation_ws/simulation_models/worlds"
WORKSPACE_MODELS="/root/thesis_social_navigation_ws/simulation_models"
TARGET_WORLDS="/opt/ros/jazzy/share/turtlebot4_gz_bringup/worlds"

echo "[setup_gazebo_worlds] Syncing custom worlds/models into turtlebot4_gz_bringup..."

if [ -d "$WORKSPACE_WORLDS" ]; then
    for world_file in "$WORKSPACE_WORLDS"/*.sdf; do
        [ -e "$world_file" ] || continue
        cp -f "$world_file" "$TARGET_WORLDS/"
        echo "[setup_gazebo_worlds]   copied $(basename "$world_file")"
    done
else
    echo "[setup_gazebo_worlds]   WARNING: $WORKSPACE_WORLDS not found, skipping world copy"
fi

# Copy known custom models referenced by model:// URIs in our
# worlds (add more names here if new models are introduced).
CUSTOM_MODELS=("person_standing")

for model_name in "${CUSTOM_MODELS[@]}"; do
    src="$WORKSPACE_MODELS/$model_name"
    if [ -d "$src" ]; then
        rm -rf "$TARGET_WORLDS/$model_name"
        cp -r "$src" "$TARGET_WORLDS/$model_name"
        echo "[setup_gazebo_worlds]   copied model: $model_name"
    else
        echo "[setup_gazebo_worlds]   WARNING: model '$model_name' not found at $src, skipping"
    fi
done

echo "[setup_gazebo_worlds] Done."
