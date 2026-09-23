#!/bin/bash
# test_static_streak.sh
#
# Replays a ROS bag through lidar_person_detector.py twice:
#   Pass A: hardware version (bug — evicts confirmed tracks)
#   Pass B: sim/fixed version (confirmed tracks exempt from eviction)
#
# Compares confirmed-track drop events between the two runs.
#
# Usage:
#   bash test_static_streak.sh <bag_dir>
#   bash test_static_streak.sh /home/brian/Desktop/Turtlebot4/humble_client/workspace/bags/berth06_20260910_111228
#
set -e

BAG="${1:-/tmp/test_bag}"
HW_DETECTOR="${2:-/tmp/lidar_person_detector_hw.py}"
SIM_DETECTOR=/root/thesis_social_navigation_ws/src/social_perception/social_perception/Lidar/lidar_person_detector.py
LOG_DIR=/tmp/static_streak_test
mkdir -p "$LOG_DIR"

source /opt/ros/jazzy/setup.bash
source /home/brian/thesis_social_navigation_ws/install/setup.bash 2>/dev/null || true

echo "======================================================"
echo " Static-streak eviction bug test"
echo " Bag: $BAG"
echo "======================================================"

run_pass() {
    local label="$1"
    local detector="$2"
    local log="$LOG_DIR/${label}.log"

    echo ""
    echo "--- Pass $label: $detector ---"

    # Launch detector in background
    python3 "$detector" --ros-args \
        -p scan_topic:=/turtlebot4/scan \
        -p use_sim_time:=true \
        -p min_points_per_cluster:=3 \
        -p confirm_min_span:=0.5 \
        > "$log" 2>&1 &
    DETECTOR_PID=$!

    sleep 2

    # Play bag — only the topics the detector needs
    ros2 bag play "$BAG" \
        --topics /turtlebot4/scan /tf /tf_static /map \
        --clock \
        --rate 1.0 \
        > /dev/null 2>&1

    sleep 2
    kill "$DETECTOR_PID" 2>/dev/null || true
    wait "$DETECTOR_PID" 2>/dev/null || true

    echo "Log saved: $log"

    # Extract key events
    echo ""
    echo "  Confirmed tracks:"
    grep "confirmed\|PERSON DETECTED\|New person\|stopped publishing\|evict\|static_streak" "$log" | grep -i "confirmed\|PERSON DETECTED\|New person\|stopped" | head -30

    echo ""
    echo "  Drop events (stopped publishing):"
    grep "stopped publishing" "$log" | head -20

    DETECTED=$(grep -c "PERSON DETECTED\|New person" "$log" 2>/dev/null || echo 0)
    DROPPED=$(grep -c "stopped publishing" "$log" 2>/dev/null || echo 0)
    echo ""
    echo "  SUMMARY: $DETECTED confirmed detections, $DROPPED dropped"
}

run_pass "A_hardware_bugged" "$HW_DETECTOR"
run_pass "B_sim_fixed"       "$SIM_DETECTOR"

echo ""
echo "======================================================"
echo " COMPARISON"
echo "======================================================"
echo "Hardware (bugged):"
grep "stopped publishing" "$LOG_DIR/A_hardware_bugged.log" | head -10

echo ""
echo "Fixed:"
grep "stopped publishing" "$LOG_DIR/B_sim_fixed.log" | head -10

echo ""
echo "Full logs in $LOG_DIR/"
