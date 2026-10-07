#!/bin/bash
# run_conv_F_trial.sh <bag_name>
#
# One complete conversation trial (ablation F, two people standing at
# (3.0, +/-0.75)) from a fresh Gazebo launch to a recorded bag. Same protocol
# as the hand-run trials of 24 Sep (README, "Running a narrow trial"), with
# the bringup, AMCL, undock and goal handling of run_headon_F_trial.sh:
#   Gazebo + Nav2 -> /initialpose -> undock -> zone + perception + group
#   detector -> wait for the expected wide/narrow class -> bag record ->
#   goal (6, 0) -> goal result -> teardown.
#
# Run inside the container:
#   docker exec -it thesis_social_nav bash
#   ./run_conv_F_trial.sh conv_F_wide_rerun_trial1
#   CASE=narrow ./run_conv_F_trial.sh conv_F_narrow_rerun_trial1
#
# Overridable through the environment:
#   CASE=wide|narrow   world conversation_test / conversation_test_narrow (default wide)
#   CASE=queue         world queue_test (four people in a line at x = 3, 1.2 m apart).
#                      Starts the group detector with queue_detection:=true and
#                      waits for a "queue" group. QUEUE_DETECTION=false runs it
#                      without (pairs only) and does not wait for a group.
#   CFG=<nav2 params file>   (default: the F social-zone sim config)
#   SHOW_RVIZ=true  HEADLESS=false   RViz (config/headon_view.rviz) / the Gazebo GUI
#   GOAL_X, GOAL_Y, GOAL_TIMEOUT, CLASS_TIMEOUT
set -e

if [ $# -ne 1 ]; then
  echo "usage: $0 <bag_name>" >&2
  exit 1
fi
NAME=$1

source /opt/ros/jazzy/setup.bash
source /root/thesis_social_navigation_ws/install/setup.bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0

WS=/root/thesis_social_navigation_ws
CL="$WS/src/social_perception/social_perception/camera_lidar"
CFG="${CFG:-$WS/config/social_nav2_ablation_F_socialzone_sim.yaml}"
CASE="${CASE:-wide}"
case "$CASE" in
  wide)   WORLD=conversation_test;        MAP="${MAP:-$WS/maps/map_name.yaml}" ;;
  narrow) WORLD=conversation_test_narrow; MAP="${MAP:-$WS/maps/conversation_test_narrow.yaml}" ;;
  queue)  WORLD=queue_test;               MAP="${MAP:-$WS/maps/map_name.yaml}" ;;
  *) echo "ERROR: CASE must be wide, narrow or queue" >&2; exit 1 ;;
esac
# Queue detection in the group detector: on for the queue case only.
QUEUE_DETECTION="${QUEUE_DETECTION:-$([ "$CASE" = queue ] && echo true || echo false)}"
BAG="$WS/bags/$NAME"
LOG_DIR="$WS/logs/$NAME"
PROV="$LOG_DIR/provenance"

# ----------------------------------------------------------------
# Conversation protocol (world frame == map frame)
# ----------------------------------------------------------------
# The robot spawns ON THE DOCK at (-1, 0) facing away from the pair and
# undocks: it backs off the dock and turns 180 deg, which leaves it at about
# (-0.65, 0) facing the pair. That is where all ten trials of 24 Sep started
# (first ground-truth pose (-0.648, -0.001, yaw 0.004), wide and narrow).
SPAWN_X=-1.0; SPAWN_Y=0.0; SPAWN_YAW=3.14159
GOAL_X="${GOAL_X:-6.0}"; GOAL_Y="${GOAL_Y:-0.0}"
GOAL_TIMEOUT="${GOAL_TIMEOUT:-180}"              # wall seconds
CLASS_TIMEOUT="${CLASS_TIMEOUT:-120}"            # wall seconds to see the expected class
PAIR_MAX_SEP="${PAIR_MAX_SEP:-1.8}"              # m; the detector's CONV_MAX_DIST (a wider estimated pair is never a group)
SEP_GIVEUP="${SEP_GIVEUP:-20}"                   # wall seconds of a wider estimated pair before the wait is given up
RVIZ="${RVIZ:-false}"; HEADLESS="${HEADLESS:-true}"
SHOW_RVIZ="${SHOW_RVIZ:-false}"

[ -e "$BAG" ] && { echo "ERROR: $BAG already exists" >&2; exit 1; }
[ -f "$MAP" ] || { echo "ERROR: map $MAP missing" >&2; exit 1; }
if pgrep -f "^gz sim" > /dev/null; then
  echo "ERROR: a Gazebo instance is already running - every trial needs a fresh launch" >&2
  exit 1
fi

mkdir -p "$LOG_DIR" "$PROV"
cd "$WS"

PIDS=()
BAG_PID=""
# SIGTERM, not SIGINT: a non-interactive shell starts background jobs with
# SIGINT ignored, so an INT here left the recorder running.
cleanup() {
  trap - INT TERM EXIT
  echo "[conv_F] Tearing down..."
  if [ -n "$BAG_PID" ]; then
    kill -TERM -- "-$BAG_PID" 2>/dev/null || true   # rosbag2 closes the file on SIGTERM
    sleep 3
  fi
  for pid in "${PIDS[@]}"; do
    kill -TERM -- "-$pid" 2>/dev/null || true
  done
  sleep 5
  for pid in "${PIDS[@]}"; do
    kill -KILL -- "-$pid" 2>/dev/null || true
  done
  pkill -KILL -f "^gz sim" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# Each process gets its own session so teardown can signal the whole group.
start() {   # start <log name> <command...>
  local log=$1; shift
  setsid "$@" > "$LOG_DIR/$log.log" 2>&1 &
  PIDS+=($!)
}

wait_active() {   # wait_active <lifecycle node> <seconds>
  local deadline=$((SECONDS + $2))
  while [ "$SECONDS" -lt "$deadline" ]; do
    state=$(timeout -k 2 10 ros2 lifecycle get "$1" 2>/dev/null | awk '{print $1}')
    [ "$state" = "active" ] && { echo "[conv_F] $1 is active."; return 0; }
    sleep 1
  done
  echo "ERROR: $1 not active after $2 s" >&2
  return 1
}

# ----------------------------------------------------------------
# 1. Gazebo + TurtleBot4 + Nav2 + localization
#    (stale-DDS cleanup and patched headless launch files: see
#    run_headon_F_trial.sh for why each is needed)
# ----------------------------------------------------------------
[ -x /usr/local/bin/setup_gazebo_worlds.sh ] && /usr/local/bin/setup_gazebo_worlds.sh

timeout 10 ros2 daemon stop > /dev/null 2>&1 || true
pkill -KILL -f "ros2cli.daemon" 2>/dev/null || true
rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* /dev/shm/fast_datasharing_* 2>/dev/null || true

SHARE_LAUNCH=/opt/ros/jazzy/share/turtlebot4_gz_bringup/launch
for f in sim.launch.py turtlebot4_gz.launch.py; do
  if ! cmp -s "$WS/launch/$f" "$SHARE_LAUNCH/$f"; then
    [ -f "$SHARE_LAUNCH/$f.orig" ] || cp "$SHARE_LAUNCH/$f" "$SHARE_LAUNCH/$f.orig"
    cp "$WS/launch/$f" "$SHARE_LAUNCH/$f"
    echo "[conv_F] Installed patched $f (headless support)."
  fi
done

echo "[conv_F] Launching Gazebo + Nav2 (world $WORLD, headless=$HEADLESS)..."
start nav2 ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py \
  world:="$WORLD" slam:=false nav2:=true localization:=true rviz:="$RVIZ" \
  map:="$MAP" params_file:="$CFG" headless:="$HEADLESS" \
  x:="$SPAWN_X" y:="$SPAWN_Y" yaw:="$SPAWN_YAW"

if [ "$SHOW_RVIZ" = "true" ]; then
  start rviz rviz2 -d "$WS/config/headon_view.rviz" --ros-args -p use_sim_time:=true
fi

wait_active /map_server 120
wait_active /amcl 60
wait_active /controller_server 120

# ----------------------------------------------------------------
# 2. Initial pose, then undock
# ----------------------------------------------------------------
echo "[conv_F] Letting AMCL's TF buffer warm up..."
sleep 5
INITIALPOSE_MSG=$(python3 -c "
import math
yaw = $SPAWN_YAW
qz = math.sin(yaw / 2.0)
qw = math.cos(yaw / 2.0)
print(f'{{header: {{frame_id: \"map\"}}, pose: {{covariance: [0.25,0,0,0,0,0, 0,0.25,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0.06853891945200942], pose: {{position: {{x: $SPAWN_X, y: $SPAWN_Y, z: 0.0}}, orientation: {{z: {qz}, w: {qw}}}}}}}}}')")
LOCALIZED=false
for attempt in 1 2 3 4; do
  ros2 topic pub -1 /initialpose geometry_msgs/msg/PoseWithCovarianceStamped "$INITIALPOSE_MSG" > /dev/null
  sleep 2
  for i in 1 2 3; do
    if timeout 4 ros2 run tf2_ros tf2_echo map odom 2>/dev/null | grep -q "Translation"; then
      LOCALIZED=true
      break
    fi
    sleep 1
  done
  [ "$LOCALIZED" = true ] && break
  echo "[conv_F] No map->odom after initial pose attempt $attempt, publishing again."
done
[ "$LOCALIZED" = true ] || { echo "ERROR: AMCL did not localize (no map->odom)" >&2; exit 1; }
echo "[conv_F] AMCL localized."

echo "[conv_F] Undocking..."
UNDOCKED=false
for attempt in 1 2; do
  timeout 60 ros2 action send_goal /undock irobot_create_msgs/action/Undock "{}" \
    > "$LOG_DIR/undock.log" 2>&1 || true
  if grep -q "SUCCEEDED" "$LOG_DIR/undock.log"; then
    UNDOCKED=true
    break
  fi
  echo "[conv_F] Undock attempt $attempt did not succeed."
done
[ "$UNDOCKED" = true ] || { echo "ERROR: undock failed - see $LOG_DIR/undock.log" >&2; exit 1; }
sleep 3
echo "[conv_F] Undocked."

# ----------------------------------------------------------------
# 3. Zone stack + perception (launch defaults = hardware values) +
#    group detector probing the static /map
# ----------------------------------------------------------------
start zone_node python3 "$CL/social_zone_costmap_node_sim.py" --ros-args -p use_sim_time:=true
start filter_info ros2 launch "$WS/launch/costmap_filter_info_sim.launch.py" params_file:="$CFG"
start perception ros2 launch "$WS/launch/perception_camray_bringup_sim.launch.py"
# The filter info server's lifecycle bringup sometimes times out at start
# ("failed to send response to .../change_state"). The keepout mask then never
# reaches the global costmap and the planner routes straight through the
# group for the whole run (bag queue_F_port_low_1), so check it came up.
wait_active /costmap_filter_info_server 60
start group_detector python3 "$CL/social_group_detector_node_lidarhold_sim.py" --ros-args \
  -p use_sim_time:=true -p costmap_topic:=/map -p show_debug_image:=false \
  -p queue_detection:="$QUEUE_DETECTION"

# ----------------------------------------------------------------
# 4. Readiness: the pair must be detected as a group of the expected
#    class before the goal is sent (field 10 of /social_groups:
#    0.400 = wide, below 0.39 = narrow), as in the 24 Sep protocol.
# ----------------------------------------------------------------
echo "[conv_F] Waiting for /social_groups to report a $CASE group..."
CLASS_OK=false
SEP_BAD=0
deadline=$((SECONDS + CLASS_TIMEOUT))
while [ "$SECONDS" -lt "$deadline" ]; do
  # NB: no `ros2 topic echo --field data`: it prints the string YAML-quoted.
  line=$(timeout -k 2 5 ros2 topic echo /social_groups std_msgs/msg/String --once 2>/dev/null \
           | sed -n 's/^data: *//p' | tr -d "'\"" | head -1)
  buf=$(echo "$line" | awk -F, 'NF >= 11 {print $11}')
  # Only a facing pair counts: the detector publishes it as "conversation"
  # (a side_by_side group has a buffer too and must not pass this wait).
  gtype=$(echo "$line" | awk -F, '{print $2}')
  want=conversation
  [ "$CASE" = queue ] && want=$([ "$QUEUE_DETECTION" = true ] && echo queue || echo "$gtype")
  if [ -n "$buf" ] && [ "$gtype" = "$want" ]; then
    echo "$SECONDS s: $line" >> "$LOG_DIR/social_groups_wait.log"
    if [ "$CASE" = queue ] \
       || { [ "$CASE" = wide ] && awk "BEGIN{exit !($buf >= 0.39)}"; } \
       || { [ "$CASE" = narrow ] && awk "BEGIN{exit !($buf < 0.39)}"; }; then
      CLASS_OK=true
      break
    fi
  fi
  # Guard: with the two tracks estimated farther apart than PAIR_MAX_SEP the
  # detector never forms a group and this wait would run to CLASS_TIMEOUT. It
  # happened with RViz on and an overloaded machine (AMCL dropping scans): one
  # track was fixed 2.0 m from the other for the whole run.
  if [ "$CASE" != queue ]; then
    sep=$(tail -n 400 "$LOG_DIR/perception.log" 2>/dev/null | grep 'human_kf_predictor\]: id:' \
      | sed -E 's/.*id:([0-9]+) pos=\(([-0-9.]+),([-0-9.]+)\).*/\1 \2 \3/' \
      | awk '{x[$1]=$2; y[$1]=$3; l[$1]=NR}
             END{a=-1;b=-1;for(i in l){if(a<0||l[i]>l[a]){b=a;a=i}else if(b<0||l[i]>l[b]){b=i}}
                 if(a>=0&&b>=0){dx=x[a]-x[b];dy=y[a]-y[b];printf "%.2f",sqrt(dx*dx+dy*dy)}}')
    if [ -n "$sep" ] && awk "BEGIN{exit !($sep > $PAIR_MAX_SEP)}"; then
      SEP_BAD=$((SEP_BAD + 1))
    else
      SEP_BAD=0
    fi
    if [ "$SEP_BAD" -ge "$SEP_GIVEUP" ]; then
      echo "ERROR: the two tracks are estimated $sep m apart (more than $PAIR_MAX_SEP m) for $SEP_GIVEUP s, so no group can form. Localisation or tracking problem (machine overloaded? RViz on?) - see $LOG_DIR/perception.log" >&2
      exit 1
    fi
  fi
  sleep 1
done
if [ "$CASE" = queue ] && [ "$QUEUE_DETECTION" != true ] && [ "$CLASS_OK" != true ]; then
  echo "[conv_F] No group reported for the queue within $CLASS_TIMEOUT s - going on without one."
  CLASS_OK=true; buf=none
fi
[ "$CLASS_OK" = true ] || { echo "ERROR: no $CASE group on /social_groups within $CLASS_TIMEOUT s (last buffer: ${buf:-none}) - see $LOG_DIR" >&2; exit 1; }
echo "[conv_F] Group state at goal time: buffer $buf ($CASE)."

# ----------------------------------------------------------------
# 5. Provenance + bag. The bag is recorded straight into $BAG (not
#    $BAG/data) because analyse_F_narrow.py expects metadata.yaml and
#    world_used.sdf side by side; rosbag2 must create the folder, so the
#    provenance is collected first and copied in once recording runs.
# ----------------------------------------------------------------
cp "$CFG" "$PROV/config_used.yaml"
cp "$CL/yolo_leg_detector_lidar_sim.py" "$PROV/yolo_used.py"
cp "$CL/camera_ray_person_node_sim.py" "$PROV/ray_person_used.py"
cp "$CL/camera_ray_identity_node_sim.py" "$PROV/ray_identity_used.py"
cp "$CL/human_kf_predictor_lidar.py" "$PROV/kf_used.py"
cp "$CL/predicted_person_cloud_node_lidar.py" "$PROV/cloud_node_used.py"
cp "$CL/social_zone_costmap_node_sim.py" "$PROV/zone_node_used.py"
cp "$CL/social_group_detector_node_lidarhold_sim.py" "$PROV/detector_used.py"
cp "$WS/launch/perception_camray_bringup_sim.launch.py" "$PROV/launch_used.py"
cp "$WS/src/social_critic/src/social_critic.cpp" "$PROV/social_critic_used.cpp"
cp "/opt/ros/jazzy/share/turtlebot4_gz_bringup/worlds/$WORLD.sdf" "$PROV/world_used.sdf"
cp "$MAP" "$PROV/map_used.yaml"
cp "$0" "$PROV/trial_script_used.sh"
cp "$WS/headon_send_goal.py" "$PROV/send_goal_used.py"
for n in /controller_server /planner_server /bt_navigator /local_costmap/local_costmap /global_costmap/global_costmap; do
  timeout 20 ros2 param dump "$n" > "$PROV/params_live$(echo "$n" | tr '/' '_').yaml" || echo "WARN: param dump failed for $n" >&2
done
find "$WS/install" -name "libsocial_critic.so" -exec md5sum {} + > "$PROV/critic_so_md5.txt" || true
grep -h "SocialCritic" "$LOG_DIR/nav2.log" > "$PROV/socialcritic_line.txt" || true
md5sum "$PROV"/*_used.* > "$PROV/provenance_md5.txt"
cat > "$PROV/run_notes.txt" <<NOTES
Condition: $NAME
Scenario: conversation ($CASE), world $WORLD, ablation F, hardware camera-ray pipeline (sim port).
Nav2 params file: $CFG
Map: $MAP
Robot spawn: ($SPAWN_X, $SPAWN_Y, yaw $SPAWN_YAW) on the dock, then undock   Goal: ($GOAL_X, $GOAL_Y)
Perception: perception_camray_bringup_sim.launch.py with its defaults
Group at goal time: type ${gtype:-none}, buffer $buf   Queue detection: $QUEUE_DETECTION
NOTES

echo "[conv_F] Recording to $BAG ..."
setsid ros2 bag record \
  --qos-profile-overrides-path "$WS/analysis/tf_qos_override.yaml" \
  --topics /clock /sim_ground_truth_pose /odom /amcl_pose /tf /tf_static \
           /plan /scan /cmd_vel /cmd_vel_nav /cmd_vel_smoothed /collision_monitor_state \
           /optimal_trajectory /person_positions_map /camera_ray_clusters \
           /person_positions_fused /predicted_person_positions /predicted_person_cloud \
           /social_groups /social_zone_map /rosout \
  -o "$BAG" > "$LOG_DIR/bag.log" 2>&1 &
BAG_PID=$!
sleep 3
[ -d "$BAG" ] || { echo "ERROR: the recorder did not create $BAG - see $LOG_DIR/bag.log" >&2; exit 1; }
cp "$PROV"/* "$BAG/"

# ----------------------------------------------------------------
# 6. Goal (headon_send_goal.py retries the handshake, see its docstring)
# ----------------------------------------------------------------
echo "[conv_F] Sending the goal ($GOAL_X, $GOAL_Y)..."
set +e
python3 "$WS/headon_send_goal.py" "$GOAL_X" "$GOAL_Y" "$GOAL_TIMEOUT" "$LOG_DIR/goal_accepted" \
  > "$BAG/goal_result.txt" 2>&1
set -e
cat "$BAG/goal_result.txt"

# ----------------------------------------------------------------
# 7. Result
# ----------------------------------------------------------------
if grep -q "status: SUCCEEDED" "$BAG/goal_result.txt"; then
  echo "[conv_F] RESULT: goal succeeded."
elif grep -q "TIMEOUT" "$BAG/goal_result.txt"; then
  echo "[conv_F] RESULT: goal TIMED OUT after $GOAL_TIMEOUT s (wall)."
elif grep -q "NOT_ACCEPTED" "$BAG/goal_result.txt"; then
  mv "$BAG/goal_result.txt" "$BAG/goal_result_invalid.txt"
  echo "[conv_F] RESULT: goal never accepted - trial INVALID."
elif grep -q "EARLY_ABORT" "$BAG/goal_result.txt"; then
  mv "$BAG/goal_result.txt" "$BAG/goal_result_invalid.txt"
  echo "[conv_F] RESULT: Nav2 aborted right after the goal was accepted (stack, not encounter) - trial INVALID."
else
  echo "[conv_F] RESULT: goal did NOT succeed (see $BAG/goal_result.txt)."
fi
sleep 2
echo "[conv_F] Analyse with: python3 $WS/analysis/analyse_F_narrow.py $BAG"
