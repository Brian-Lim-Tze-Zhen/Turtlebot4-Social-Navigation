#!/bin/bash
set -e
[ -z "$1" ] && { echo "usage: $0 <bag_name>"; exit 1; }
WS=/root/thesis_social_navigation_ws
BAG=$WS/bags/$1

# Refuse to overwrite an existing trial. Re-running with a name that is
# already taken silently replaced the bag AND its provenance snapshots,
# so the earlier run vanished with no warning - discovered only because
# two analyses of the "same" trial disagreed.
if [ -e "$BAG" ]; then
  echo "ERROR: $BAG already exists. Choose another name, or remove it first." >&2
  exit 1
fi
[ -e "$BAG" ] && { echo "ERROR: $BAG exists"; exit 1; }

N=$(ps aux | grep -c "[y]olo_detector" || true)
[ "$N" -eq 1 ] || { echo "ERROR: yolo_detector count = $N (need 1)"; exit 1; }

mkdir -p "$BAG"
cp $WS/config/social_nav2_ablation_E_critweight20_socialcritic_on.yaml "$BAG/config_used.yaml"
cp $WS/src/social_perception/social_perception/predicted_person_cloud_node.py "$BAG/cloud_node_used.py"
cp $WS/src/social_perception/social_perception/human_kf_predictor.py "$BAG/kf_used.py"
cp $WS/src/social_perception/social_perception/move_person_oneway.py "$BAG/mover_used.py"
# Perception/group-detection provenance (added for queue scenario).
# yolo_detector.py: perception v2 boundary (track_id=-1 filter).
# group_formation_detector.py: _detect_queue + queue hold-open params.
# social_group_cloud_node.py: queue costmap zone geometry.
cp $WS/src/social_perception/social_perception/yolo_detector.py "$BAG/yolo_used.py"
cp $WS/src/social_perception/social_perception/group_formation_detector.py "$BAG/group_detector_used.py"
cp $WS/src/social_perception/social_perception/social_group_cloud_node.py "$BAG/group_cloud_used.py"
# Queue scenario additions.
# leg_detector: lidar clusters anchor the queue hold-open when the camera
#   goes blind, so its parameters affect the recorded run.
# queue_test.sdf: the scenario's people are static, so the world file -
#   not a mover script - defines the queue geometry.
cp $WS/src/social_perception/social_perception/leg_detector_node.py "$BAG/leg_detector_used.py"
cp $WS/simulation_models/worlds/queue_test.sdf "$BAG/world_used.sdf"
cp $WS/src/social_perception/social_perception/queue_ground_truth_node.py "$BAG/ground_truth_used.py"
cp $WS/launch/queue_perception.launch.py "$BAG/launch_used.py"

# Configuration NOT captured by a file snapshot - record it explicitly.
cat > "$BAG/run_notes.txt" <<'NOTES'
Scenario: queue_test, 4 static pedestrians, x=3, y=1/-0.2/-1.4/-2.6
Robot spawn: x=-2.0 y=-1.0 yaw=0.0   Goal: (6.5, 0.0)

identity_fusion_node: NOT RUNNING.
  Lidar (leg_detector_node) was used ONLY as a position anchor for the
  queue hold-open, never for camera-lidar re-identification. The KF ran
  WITHOUT the -p input_topic:=/person_positions_fused remap, i.e. on raw
  /person_positions_map. Do not read this run as a full fusion chain.

Perception v2: yolo_detector filters track_id=-1.
BT replan rate: unchanged from the head-on scenario (no override applied).
NOTES

ros2 bag record \
  --qos-profile-overrides-path $WS/eval_scripts/tf_qos_override.yaml \
  --topics /person_ground_truth /odom /amcl_pose /tf /tf_static /plan \
           /predicted_person_positions /person_positions_map \
	   /fused_person_positions \
           /cmd_vel /cmd_vel_smoothed /cmd_vel_nav /collision_monitor_state \
           /optimal_trajectory \
           /social_groups /social_group_cloud \
           /predicted_person_cloud /lidar_person_clusters \
  -o "$BAG/data"
