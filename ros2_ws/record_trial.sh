#!/bin/bash
set -e
[ -z "$1" ] && { echo "usage: $0 <bag_name>"; exit 1; }
WS=/root/thesis_social_navigation_ws
BAG=$WS/bags/$1
[ -e "$BAG" ] && { echo "ERROR: $BAG exists"; exit 1; }

N=$(ps aux | grep -c "[y]olo_detector" || true)
[ "$N" -eq 1 ] || { echo "ERROR: yolo_detector count = $N (need 1)"; exit 1; }

mkdir -p "$BAG"
cp $WS/config/social_nav2_fast_person_test.yaml "$BAG/config_used.yaml"
cp $WS/src/social_perception/social_perception/predicted_person_cloud_node.py "$BAG/cloud_node_used.py"
cp $WS/src/social_perception/social_perception/human_kf_predictor.py "$BAG/kf_used.py"
cp $WS/src/social_perception/social_perception/move_person_oneway.py "$BAG/mover_used.py"

ros2 bag record \
  --qos-profile-overrides-path $WS/eval_scripts/tf_qos_override.yaml \
  --topics /person_ground_truth /odom /amcl_pose /tf /tf_static /plan \
           /predicted_person_positions /person_positions_map \
	   /fused_person_positions \
           /cmd_vel /cmd_vel_smoothed /cmd_vel_nav /collision_monitor_state \
           /optimal_trajectory \
  -o "$BAG/data"
