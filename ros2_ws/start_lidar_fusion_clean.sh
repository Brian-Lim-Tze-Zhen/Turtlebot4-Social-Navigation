#!/bin/bash
# start_lidar_fusion_clean.sh
#
# Kills any leftover lidar_person_fusion.py process before starting a
# fresh one, so a forgotten Ctrl+C between trials can't leak
# self.lidar_xy / self.lidar_vx state from the previous trial into the
# next one's SocialCritic scoring.
#
# Usage: same as running the node directly, just call this instead:
#   ./start_lidar_fusion_clean.sh

set -e

NODE_PATH="/root/thesis_social_navigation_ws/src/social_perception/social_perception/lidar_person_fusion.py"

# Kill any stale instance and wait for it to actually exit before
# starting a new one — a race here (starting before the old one has
# released its subscriptions) can also cause weird transient behavior.
if pgrep -f "lidar_person_fusion.py" > /dev/null; then
    echo "[start_lidar_fusion_clean] Found a running lidar_person_fusion.py — killing it first."
    pkill -f "lidar_person_fusion.py"
    # Wait up to 5s for it to actually die
    for i in $(seq 1 50); do
        if ! pgrep -f "lidar_person_fusion.py" > /dev/null; then
            break
        fi
        sleep 0.1
    done
    if pgrep -f "lidar_person_fusion.py" > /dev/null; then
        echo "[start_lidar_fusion_clean] WARNING: old process did not exit within 5s, forcing kill -9"
        pkill -9 -f "lidar_person_fusion.py"
        sleep 0.5
    fi
fi

echo "[start_lidar_fusion_clean] Starting fresh lidar_person_fusion.py instance."
exec python3 -u "$NODE_PATH" --ros-args -p use_sim_time:=true
