#!/bin/bash
# replay_detector_test.sh
#
# Replays a bag through two versions of lidar_person_detector and compares
# how many confirmed tracks (travel > 0.5m) get dropped.
#
# Usage:
#   bash replay_detector_test.sh <bag_dir> <hw_detector_path>
#
# Example:
#   bash replay_detector_test.sh /tmp/test_bag /tmp/lidar_person_detector_hw.py

BAG="${1:-/tmp/test_bag}"
HW_DETECTOR="${2:-/tmp/lidar_person_detector_hw.py}"
FIXED_DETECTOR="/root/thesis_social_navigation_ws/src/social_perception/social_perception/Lidar/lidar_person_detector.py"
LOG_DIR="/root/thesis_social_navigation_ws/logs/streak_test"
mkdir -p "$LOG_DIR"

source /opt/ros/jazzy/setup.bash
source /root/thesis_social_navigation_ws/install/setup.bash 2>/dev/null || true

echo "======================================================"
echo " Replay detector comparison"
echo " Bag: $BAG"
echo "======================================================"

run_detector_on_bag() {
    local label="$1"
    local detector="$2"
    local out="$LOG_DIR/${label}_clusters.log"

    echo ""
    echo "--- $label ---"

    # Subscribe to /lidar_person_clusters and log it
    ros2 topic echo /lidar_person_clusters std_msgs/msg/String > "$LOG_DIR/${label}_echo.log" 2>/dev/null &
    ECHO_PID=$!

    # Start detector
    python3 "$detector" --ros-args \
        -p scan_topic:=/turtlebot4/scan \
        -p use_sim_time:=true \
        -p min_points_per_cluster:=3 \
        -p confirm_min_span:=0.5 \
        -p static_occupancy_threshold:=50 \
        > "$out" 2>&1 &
    DETECTOR_PID=$!

    sleep 3

    # Play bag — only topics detector needs
    echo "  Playing bag..."
    ros2 bag play "$BAG" \
        --topics /turtlebot4/scan /tf /tf_static /turtlebot4/map /map \
        --clock \
        --rate 1.0 2>/dev/null

    echo "  Bag finished, waiting 3s..."
    sleep 3

    kill "$DETECTOR_PID" 2>/dev/null
    kill "$ECHO_PID" 2>/dev/null
    wait "$DETECTOR_PID" 2>/dev/null || true
    wait "$ECHO_PID" 2>/dev/null || true

    echo "  Detector log: $out"

    # Count key events
    CONFIRMED=$(grep -c "PERSON DETECTED\|New person id" "$out" 2>/dev/null || echo 0)
    DROPPED=$(grep -c "stopped publishing" "$out" 2>/dev/null || echo 0)
    EVICTED=$(grep -c "static_streak" "$out" 2>/dev/null || echo 0)

    echo ""
    echo "  Confirmed detections : $CONFIRMED"
    echo "  Tracks dropped       : $DROPPED"
    echo "  Static evictions     : $EVICTED"
    echo ""
    echo "  Drop details:"
    grep "stopped publishing" "$out" | head -20
}

run_detector_on_bag "A_hardware_bugged" "$HW_DETECTOR"
run_detector_on_bag "B_fixed" "$FIXED_DETECTOR"

echo ""
echo "======================================================"
echo " FINAL COMPARISON"
echo "======================================================"
echo "Hardware (bugged) drops: $(grep -c 'stopped publishing' $LOG_DIR/A_hardware_bugged_clusters.log 2>/dev/null || echo 0)"
echo "Fixed drops            : $(grep -c 'stopped publishing' $LOG_DIR/B_fixed_clusters.log 2>/dev/null || echo 0)"
echo ""
echo "Full logs in $LOG_DIR/"
