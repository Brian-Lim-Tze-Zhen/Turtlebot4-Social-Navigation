#!/bin/bash
# Head-on A/B recorder. Snapshots exactly what runs: the config file passed
# in, the code folder passed in, live Nav2 params, and running processes.
set -e
if [ $# -ne 4 ]; then
  echo "usage: $0 <bag_name> <nav2_params_file> <code_ws_root> <cloud_enabled:true|false>" >&2
  exit 1
fi
NAME=$1; CFG=$2; CODE_ROOT=$3; CLOUD=$4
WS=/root/thesis_social_navigation_ws
BAG=$WS/bags/$NAME
PKG=$CODE_ROOT/src/social_perception/social_perception

[ -e "$BAG" ] && { echo "ERROR: $BAG already exists" >&2; exit 1; }
[ -f "$CFG" ] || { echo "ERROR: config $CFG not found" >&2; exit 1; }
case "$CLOUD" in true|false) ;; *) echo "ERROR: cloud_enabled must be true or false" >&2; exit 1;; esac
for f in yolo_detector.py human_kf_predictor.py predicted_person_cloud_node.py move_person_oneway.py; do
  [ -f "$PKG/$f" ] || { echo "ERROR: $PKG/$f missing" >&2; exit 1; }
done

# Which perception processes are running, and from which folder.
PROCS=$(ps -eo args | grep -E "[y]olo_detector.py|[h]uman_kf_predictor.py|[p]redicted_person_cloud_node.py|[m]ove_person_oneway.py" || true)
N_YOLO=$(echo "$PROCS" | grep -c "yolo_detector.py" || true)
[ "$N_YOLO" -eq 1 ] || { echo "ERROR: yolo_detector count = $N_YOLO (need 1)" >&2; exit 1; }
echo "$PROCS" | grep "yolo_detector.py" | grep -q "$PKG/" || { echo "ERROR: yolo_detector not running from $PKG" >&2; exit 1; }
echo "$PROCS" | grep "human_kf_predictor.py" | grep -q "$PKG/" || { echo "ERROR: human_kf_predictor not running from $PKG" >&2; exit 1; }
N_CLOUD=$(echo "$PROCS" | grep -c "predicted_person_cloud_node.py" || true)
if [ "$CLOUD" = "true" ]; then
  [ "$N_CLOUD" -eq 1 ] || { echo "ERROR: cloud node count = $N_CLOUD (need 1 for cloud ON)" >&2; exit 1; }
  echo "$PROCS" | grep "predicted_person_cloud_node.py" | grep -q "$PKG/" || { echo "ERROR: cloud node not running from $PKG" >&2; exit 1; }
else
  [ "$N_CLOUD" -eq 0 ] || { echo "ERROR: cloud node is running but cloud_enabled=false" >&2; exit 1; }
fi

mkdir -p "$BAG"
cp "$CFG" "$BAG/config_used.yaml"
echo "$CFG" > "$BAG/config_path.txt"
cp "$PKG/predicted_person_cloud_node.py" "$BAG/cloud_node_used.py"
cp "$PKG/human_kf_predictor.py" "$BAG/kf_used.py"
cp "$PKG/move_person_oneway.py" "$BAG/mover_used.py"
cp "$PKG/yolo_detector.py" "$BAG/yolo_used.py"
cp "$WS/launch/headon_scenario.py" "$BAG/launch_used.py"
cp /opt/ros/jazzy/share/turtlebot4_gz_bringup/worlds/empty_human.sdf "$BAG/world_used.sdf"
echo "$PROCS" > "$BAG/processes_running.txt"

# Live parameters from the running Nav2 nodes (what actually launched).
for n in /controller_server /planner_server /bt_navigator /local_costmap/local_costmap /global_costmap/global_costmap; do
  ros2 param dump "$n" > "$BAG/params_live$(echo "$n" | tr '/' '_').yaml" || echo "WARN: param dump failed for $n" >&2
done
find "$WS/install" -name "libsocial_critic.so" -exec md5sum {} + > "$BAG/critic_so_md5.txt" || true
md5sum "$BAG"/*_used.* > "$BAG/provenance_md5.txt"

cat > "$BAG/run_notes.txt" <<NOTES
Condition: $NAME
Nav2 params file: $CFG
Code root: $CODE_ROOT
Predicted cloud enabled: $CLOUD
Scenario: head-on, world empty_human, A-E August protocol.
yolo_detector: as found in code root (replay_AB = pre-0451ced, no track_id=-1 filter).
NOTES

ros2 bag record \
  --qos-profile-overrides-path $WS/analysis/tf_qos_override.yaml \
  --topics /person_ground_truth /odom /amcl_pose /tf /tf_static /plan \
           /predicted_person_positions /person_positions_map /predicted_person_cloud \
           /cmd_vel /cmd_vel_smoothed /cmd_vel_nav /collision_monitor_state \
           /optimal_trajectory /rosout \
  -o "$BAG/data"
