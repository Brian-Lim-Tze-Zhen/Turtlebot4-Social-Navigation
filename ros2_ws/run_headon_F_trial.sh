#!/bin/bash
# run_headon_F_trial.sh <bag_name>
#
# One complete head-on trial in the 2.5 m corridor with the HARDWARE pipeline
# (camera-ray perception + ablation F config), from a fresh Gazebo launch to
# a recorded bag:
#   Gazebo + Nav2 -> /initialpose -> perception -> bag record ->
#   person starts walking + goal sent -> goal result -> teardown.
#
# Run inside the container:
#   docker exec -it thesis_social_nav bash
#   ./run_headon_F_trial.sh headon_F_corridor_trial1
#
# Overridable through the environment:
#   RVIZ=true  HEADLESS=false   show RViz / the Gazebo GUI (pilot runs)
#   WORLD=empty_human   open-space head-on world (default corridor_headon)
#   CFG=<nav2 params file>   (default: the hardware-synced F sim config)
#   YOLO_IMGSZ, YOLO_MIN_CONF, MAX_PERSON_RANGE, LANE_SLOPE, LANE_MAX, PASS_BLOCK,
#   COAST_TIMEOUT, RAY_COAST, RAY_COAST_S
#   LANE_B, DISK_R  person cloud lane half-width / body disk radius
#   TRACK_DROPOUT_AT=<m> [TRACK_DROPOUT_S]   test: critic loses the track, new id
#   SHOW_RVIZ=true   open RViz (config/headon_view.rviz) for the trial
#                                                 (defaults: hardware values)
#   PERSON_SPEED, PERSON_Y, PERSON_X0, PERSON_X1, GOAL_X, GOAL_Y, GOAL_TIMEOUT
#   PERSON2_Y=<m> [PERSON2_X0 PERSON2_X1 PERSON2_SPEED]   second walker (person_2);
#                                  needs a world that has it: WORLD=two_human
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
MOVER="$WS/src/social_perception/social_perception/move_person_oneway.py"
CFG="${CFG:-$WS/config/social_nav2_ablation_F_socialzone_sim.yaml}"
# World and map. Default: the 2.5 m corridor with its world-aligned map.
# WORLD=empty_human is the open-space ("wide") head-on world, a 15 x 10 m box
# with the same person_1 model; its map (maps/map_name.yaml) is already in the
# Gazebo frame. The spawn, goal and person line below fit both.
WORLD="${WORLD:-corridor_headon}"
if [ "$WORLD" = "corridor_headon" ]; then
  MAP="${MAP:-$WS/maps/corridor_headon_aligned.yaml}"
else
  MAP="${MAP:-$WS/maps/map_name.yaml}"
fi
BAG="$WS/bags/$NAME"
LOG_DIR="$WS/logs/$NAME"

# ----------------------------------------------------------------
# Head-on protocol (world frame == map frame with the aligned map)
# ----------------------------------------------------------------
# The robot spawns ON THE DOCK facing away from the person (yaw 180 deg),
# like the real robot, and undocks: it backs off the dock and turns 180 deg,
# ending up facing the person along +x.
SPAWN_X=-3.0; SPAWN_Y=0.0; SPAWN_YAW=3.14159
GOAL_X="${GOAL_X:-8.0}"; GOAL_Y="${GOAL_Y:-0.0}"
PERSON_X0="${PERSON_X0:-9.5}"; PERSON_X1="${PERSON_X1:--2.0}"   # stops short of the dock
PERSON_Y="${PERSON_Y:-0.0}"
PERSON_SPEED="${PERSON_SPEED:-1.2}"
# Second walker: off unless PERSON2_Y is set.
PERSON2_Y="${PERSON2_Y:-}"
PERSON2_X0="${PERSON2_X0:-$PERSON_X0}"; PERSON2_X1="${PERSON2_X1:-$PERSON_X1}"
PERSON2_SPEED="${PERSON2_SPEED:-$PERSON_SPEED}"
GOAL_TIMEOUT="${GOAL_TIMEOUT:-180}"              # wall seconds
RVIZ="${RVIZ:-false}"; HEADLESS="${HEADLESS:-true}"
# Perception range settings; defaults are the hardware values.
YOLO_IMGSZ="${YOLO_IMGSZ:-320}"; YOLO_MIN_CONF="${YOLO_MIN_CONF:-0.45}"
MAX_PERSON_RANGE="${MAX_PERSON_RANGE:-8.0}"
LANE_SLOPE="${LANE_SLOPE:-0.75}"; LANE_MAX="${LANE_MAX:-3.0}"   # person cloud forward lane
PASS_BLOCK="${PASS_BLOCK:-0.0}"   # keep-right strip width beside the lane, 0 = off
# Track persistence once the camera loses the person; defaults = hardware launch.
COAST_TIMEOUT="${COAST_TIMEOUT:-1.5}"; RAY_COAST="${RAY_COAST:-false}"; RAY_COAST_S="${RAY_COAST_S:-1.5}"
# SHOW_RVIZ=true opens RViz with config/headon_view.rviz (map, costmaps, person
# cloud, MPPI path) for this trial. It must be started by the trial, after the
# /dev/shm cleanup below: an RViz left open from before loses its DDS
# shared-memory segments in that cleanup and shows nothing.
SHOW_RVIZ="${SHOW_RVIZ:-false}"
# Person cloud lane half-width and body disk radius; defaults = hardware.
LANE_B="${LANE_B:-0.4}"; DISK_R="${DISK_R:-0.4}"

# TEST TOOL: TRACK_DROPOUT_AT=<m> makes the SocialCritic lose the person's
# track at that distance for TRACK_DROPOUT_S seconds and get it back under a
# new id (camera_lidar/track_dropout_relay_sim.py), as happened on the real
# robot. Empty = off.
TRACK_DROPOUT_AT="${TRACK_DROPOUT_AT:-}"; TRACK_DROPOUT_S="${TRACK_DROPOUT_S:-1.0}"

[ -e "$BAG" ] && { echo "ERROR: $BAG already exists" >&2; exit 1; }
[ -f "$MAP" ] || { echo "ERROR: map $MAP missing (the corridor map is made by simulation_models/worlds/make_corridor_headon_map.py, run from maps/)" >&2; exit 1; }
if pgrep -f "^gz sim" > /dev/null; then
  echo "ERROR: a Gazebo instance is already running - every trial needs a fresh launch" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
cd "$WS"

if [ -n "$TRACK_DROPOUT_AT" ]; then
  # Same config, with the critic reading the relay's output topic.
  sed 's|^\(  *\)topic: /predicted_person_positions$|\1topic: /predicted_person_positions_dropout|' \
    "$CFG" > "$LOG_DIR/config_dropout.yaml"
  grep -q "topic: /predicted_person_positions_dropout" "$LOG_DIR/config_dropout.yaml" \
    || { echo "ERROR: SocialCritic topic line not found in $CFG" >&2; exit 1; }
  CFG="$LOG_DIR/config_dropout.yaml"
fi

PIDS=()
BAG_PID=""
# SIGTERM, not SIGINT: a non-interactive shell starts background jobs with
# SIGINT ignored, so an INT here left the recorder running.
cleanup() {
  trap - INT TERM EXIT
  echo "[headon_F] Tearing down..."
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
# On a signal the script must also STOP: with a bare cleanup handler it
# carried on after tearing everything down and recorded an empty bag.
trap cleanup EXIT
trap 'exit 130' INT TERM

# Each process gets its own session so teardown can signal the whole group.
start() {   # start <log name> <command...>
  local log=$1; shift
  setsid "$@" > "$LOG_DIR/$log.log" 2>&1 &
  PIDS+=($!)
}

wait_active() {   # wait_active <lifecycle node> <seconds>
  # Deadline in wall seconds, not a loop count: each `ros2 lifecycle get`
  # can itself take up to its 10 s timeout when the node does not answer, so
  # a count of 120 once stretched a failed bringup to over 6 minutes.
  local deadline=$((SECONDS + $2))
  while [ "$SECONDS" -lt "$deadline" ]; do
    state=$(timeout 10 ros2 lifecycle get "$1" 2>/dev/null | awk '{print $1}')
    [ "$state" = "active" ] && { echo "[headon_F] $1 is active."; return 0; }
    sleep 1
  done
  echo "ERROR: $1 not active after $2 s" >&2
  return 1
}

# ----------------------------------------------------------------
# 1. Gazebo + TurtleBot4 + Nav2 + localization
# ----------------------------------------------------------------
[ -x /usr/local/bin/setup_gazebo_worlds.sh ] && /usr/local/bin/setup_gazebo_worlds.sh

# Processes killed at the end of the previous trial leave FastDDS
# shared-memory segments and locks behind in /dev/shm. With hundreds of
# stale ones, service replies got lost during the next bringup: Nav2
# lifecycle transitions hung (map_server stuck in "Configuring"), goals
# and follow_path were not acknowledged. Nothing of ours is running at
# this point (checked above), so clear them, daemon included.
# timeout: `ros2 daemon stop` itself was seen to hang for over 5 minutes.
timeout 10 ros2 daemon stop > /dev/null 2>&1 || true
pkill -KILL -f "ros2cli.daemon" 2>/dev/null || true
rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_* /dev/shm/fast_datasharing_* 2>/dev/null || true

# The installed turtlebot4_gz_bringup launch files have no `headless`
# argument: they silently ignore it and open the Gazebo GUI, and when that
# window closes or fails the server is shut down with it (seen mid-trial).
# The workspace carries patched copies (launch/sim.launch.py,
# launch/turtlebot4_gz.launch.py); install them, keeping the originals.
SHARE_LAUNCH=/opt/ros/jazzy/share/turtlebot4_gz_bringup/launch
for f in sim.launch.py turtlebot4_gz.launch.py; do
  if ! cmp -s "$WS/launch/$f" "$SHARE_LAUNCH/$f"; then
    [ -f "$SHARE_LAUNCH/$f.orig" ] || cp "$SHARE_LAUNCH/$f" "$SHARE_LAUNCH/$f.orig"
    cp "$WS/launch/$f" "$SHARE_LAUNCH/$f"
    echo "[headon_F] Installed patched $f (headless support)."
  fi
done

echo "[headon_F] Launching Gazebo + Nav2 (world $WORLD, headless=$HEADLESS)..."
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
# 2. Initial pose (see run_sim_perception.sh for why the warm-up sleep
#    and the covariance-before-pose field order are needed)
# ----------------------------------------------------------------
echo "[headon_F] Letting AMCL's TF buffer warm up..."
sleep 5
INITIALPOSE_MSG=$(python3 -c "
import math
yaw = $SPAWN_YAW
qz = math.sin(yaw / 2.0)
qw = math.cos(yaw / 2.0)
print(f'{{header: {{frame_id: \"map\"}}, pose: {{covariance: [0.25,0,0,0,0,0, 0,0.25,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0.06853891945200942], pose: {{position: {{x: $SPAWN_X, y: $SPAWN_Y, z: 0.0}}, orientation: {{z: {qz}, w: {qw}}}}}}}}}')")
# AMCL sometimes rejects the pose ("Failed to transform initial pose in
# time", its TF buffer is still behind the message stamp) and then never
# publishes map->odom, so publish again until the transform appears.
LOCALIZED=false
for attempt in 1 2 3 4; do
  ros2 topic pub -1 /initialpose geometry_msgs/msg/PoseWithCovarianceStamped "$INITIALPOSE_MSG" > /dev/null
  sleep 2
  for i in 1 2 3; do
    # tf2_echo never exits by itself, so judge by its output, not its status.
    if timeout 4 ros2 run tf2_ros tf2_echo map odom 2>/dev/null | grep -q "Translation"; then
      LOCALIZED=true
      break
    fi
    sleep 1
  done
  [ "$LOCALIZED" = true ] && break
  echo "[headon_F] No map->odom after initial pose attempt $attempt, publishing again."
done
[ "$LOCALIZED" = true ] || { echo "ERROR: AMCL did not localize (no map->odom)" >&2; exit 1; }
echo "[headon_F] AMCL localized."

# ----------------------------------------------------------------
# 2b. Undock (AMCL is already localized, so it tracks the manoeuvre)
# ----------------------------------------------------------------
echo "[headon_F] Undocking..."
# The goal is sometimes accepted by nobody right after startup (sent, no
# reply); one retry covers it.
UNDOCKED=false
for attempt in 1 2; do
  timeout 60 ros2 action send_goal /undock irobot_create_msgs/action/Undock "{}" \
    > "$LOG_DIR/undock.log" 2>&1 || true
  if grep -q "SUCCEEDED" "$LOG_DIR/undock.log"; then
    UNDOCKED=true
    break
  fi
  echo "[headon_F] Undock attempt $attempt did not succeed."
done
[ "$UNDOCKED" = true ] || { echo "ERROR: undock failed - see $LOG_DIR/undock.log" >&2; exit 1; }
sleep 3
echo "[headon_F] Undocked."

# ----------------------------------------------------------------
# 3. Person at the start point, then perception (same stack as hardware:
#    camera-ray launch with coast 1.5 s and ray coast off, plus the F
#    zone stack that the KeepoutFilter in the config expects)
# ----------------------------------------------------------------
# The SDF places person_1 at (9.0, 0.6). Move it to the start point now,
# so the mover's first step is not a 0.8 m jump seen by the tracker.
# Yaw -pi/2 = heading -x plus the mover's +pi/2 mesh offset.
gz service -s "/world/$WORLD/set_pose" --reqtype gz.msgs.Pose --reptype gz.msgs.Boolean \
  --timeout 3000 --req "name: \"person_1\", position: {x: $PERSON_X0, y: $PERSON_Y, z: 0.0}, orientation: {z: -0.70710678, w: 0.70710678}" \
  > /dev/null || { echo "ERROR: could not place person_1 at the start point" >&2; exit 1; }
if [ -n "$PERSON2_Y" ]; then
  gz service -s "/world/$WORLD/set_pose" --reqtype gz.msgs.Pose --reptype gz.msgs.Boolean \
    --timeout 3000 --req "name: \"person_2\", position: {x: $PERSON2_X0, y: $PERSON2_Y, z: 0.0}, orientation: {z: -0.70710678, w: 0.70710678}" \
    > /dev/null || { echo "ERROR: could not place person_2 (use WORLD=two_human)" >&2; exit 1; }
fi

start bridge ros2 run ros_gz_bridge parameter_bridge \
  "/world/${WORLD}/set_pose@ros_gz_interfaces/srv/SetEntityPose"
start zone_node python3 "$CL/social_zone_costmap_node_sim.py" --ros-args -p use_sim_time:=true
start filter_info ros2 launch "$WS/launch/costmap_filter_info_sim.launch.py" params_file:="$CFG"
start perception ros2 launch "$WS/launch/perception_camray_bringup_sim.launch.py" \
  coast_timeout:="$COAST_TIMEOUT" ray_coast:="$RAY_COAST" ray_coast_s:="$RAY_COAST_S" \
  yolo_imgsz:="$YOLO_IMGSZ" yolo_min_conf:="$YOLO_MIN_CONF" max_person_range:="$MAX_PERSON_RANGE" \
  ellipse_a_slope:="$LANE_SLOPE" ellipse_a_max:="$LANE_MAX" pass_side_block_width:="$PASS_BLOCK" \
  ellipse_b:="$LANE_B" person_disk_radius:="$DISK_R"
if [ -n "$TRACK_DROPOUT_AT" ]; then
  start dropout_relay python3 "$CL/track_dropout_relay_sim.py" --ros-args \
    -p use_sim_time:=true -p dropout_at_m:="$TRACK_DROPOUT_AT" -p dropout_s:="$TRACK_DROPOUT_S"
fi
start group_detector python3 "$CL/social_group_detector_node_lidarhold_sim.py" --ros-args \
  -p use_sim_time:=true -p costmap_topic:=/map -p show_debug_image:=false

# ----------------------------------------------------------------
# 4. Readiness. The person starts 12.5 m away, beyond the 8 m camera-ray
#    range, so there is no track to wait for; wait for the pipeline's own
#    outputs instead.
# ----------------------------------------------------------------
echo "[headon_F] Waiting for the perception pipeline..."
READY=false
for i in $(seq 1 90); do
  if timeout 3 ros2 topic echo /predicted_person_cloud --once > /dev/null 2>&1 \
     && [ "$(ros2 topic info /person_positions_map 2>/dev/null | awk '/Publisher count/ {print $3}')" = "1" ] \
     && [ "$(ros2 topic info /camera_ray_clusters 2>/dev/null | awk '/Publisher count/ {print $3}')" = "1" ] \
     && [ "$(ros2 topic info /person_positions_fused 2>/dev/null | awk '/Publisher count/ {print $3}')" = "1" ] \
     && [ "$(ros2 topic info /predicted_person_positions 2>/dev/null | awk '/Publisher count/ {print $3}')" = "1" ]; then
    READY=true
    break
  fi
  sleep 1
done
[ "$READY" = true ] || { echo "ERROR: perception pipeline not ready - see $LOG_DIR/perception.log" >&2; exit 1; }
sleep 5   # YOLO model warm-up
echo "[headon_F] Perception ready."

# ----------------------------------------------------------------
# 5. Provenance + bag
# ----------------------------------------------------------------
mkdir -p "$BAG"
cp "$CFG" "$BAG/config_used.yaml"
cp "$CL/yolo_leg_detector_lidar_sim.py" "$BAG/yolo_used.py"
cp "$CL/camera_ray_person_node_sim.py" "$BAG/ray_person_used.py"
cp "$CL/camera_ray_identity_node_sim.py" "$BAG/ray_identity_used.py"
cp "$CL/human_kf_predictor_lidar.py" "$BAG/kf_used.py"
cp "$CL/predicted_person_cloud_node_lidar.py" "$BAG/cloud_node_used.py"
cp "$CL/social_zone_costmap_node_sim.py" "$BAG/zone_node_used.py"
cp "$CL/social_group_detector_node_lidarhold_sim.py" "$BAG/detector_used.py"
cp "$MOVER" "$BAG/mover_used.py"
cp "$WS/launch/perception_camray_bringup_sim.launch.py" "$BAG/launch_used.py"
cp "$WS/src/social_critic/src/social_critic.cpp" "$BAG/social_critic_used.cpp"
cp "/opt/ros/jazzy/share/turtlebot4_gz_bringup/worlds/$WORLD.sdf" "$BAG/world_used.sdf"
cp "$MAP" "$BAG/map_used.yaml"
cp "$0" "$BAG/trial_script_used.sh"
cp "$WS/headon_send_goal.py" "$BAG/send_goal_used.py"
for n in /controller_server /planner_server /bt_navigator /local_costmap/local_costmap /global_costmap/global_costmap; do
  # timeout: a param dump was seen to hang indefinitely on /controller_server.
  timeout 20 ros2 param dump "$n" > "$BAG/params_live$(echo "$n" | tr '/' '_').yaml" || echo "WARN: param dump failed for $n" >&2
done
find "$WS/install" -name "libsocial_critic.so" -exec md5sum {} + > "$BAG/critic_so_md5.txt" || true
grep -h "SocialCritic" "$LOG_DIR/nav2.log" > "$BAG/socialcritic_line.txt" || true
md5sum "$BAG"/*_used.* > "$BAG/provenance_md5.txt"
cat > "$BAG/run_notes.txt" <<NOTES
Condition: $NAME
Scenario: head-on, world $WORLD, ablation F, hardware camera-ray pipeline (sim port).
Nav2 params file: $CFG
Map: $MAP
Robot spawn: ($SPAWN_X, $SPAWN_Y, yaw $SPAWN_YAW) on the dock, then undock   Goal: ($GOAL_X, $GOAL_Y)
Person: ($PERSON_X0, $PERSON_Y) -> ($PERSON_X1, $PERSON_Y) at $PERSON_SPEED m/s, one-way
Person 2: ${PERSON2_Y:+($PERSON2_X0, $PERSON2_Y) -> ($PERSON2_X1, $PERSON2_Y) at $PERSON2_SPEED m/s}${PERSON2_Y:+ }$([ -z "$PERSON2_Y" ] && echo none)
Perception: coast_timeout $COAST_TIMEOUT, ray_coast $RAY_COAST ($RAY_COAST_S s), yolo imgsz $YOLO_IMGSZ, min conf $YOLO_MIN_CONF, max person range $MAX_PERSON_RANGE, lane slope $LANE_SLOPE max $LANE_MAX, pass-side block $PASS_BLOCK, lane half-width $LANE_B, disk radius $DISK_R
Track dropout test: at "${TRACK_DROPOUT_AT:-off}" m for $TRACK_DROPOUT_S s
NOTES

echo "[headon_F] Recording to $BAG/data ..."
setsid ros2 bag record \
  --qos-profile-overrides-path "$WS/analysis/tf_qos_override.yaml" \
  --topics /clock /person_ground_truth /person2_ground_truth /sim_ground_truth_pose /odom /amcl_pose /tf /tf_static \
           /plan /scan /cmd_vel /cmd_vel_nav /cmd_vel_smoothed /collision_monitor_state \
           /optimal_trajectory /person_positions_map /camera_ray_clusters \
           /person_positions_fused /predicted_person_positions /predicted_person_cloud \
           /social_groups /social_zone_map /social_critic/lane_markers /social_critic/lane_block \
           /predicted_person_positions_dropout /rosout \
  -o "$BAG/data" > "$LOG_DIR/bag.log" 2>&1 &
BAG_PID=$!
sleep 3

# ----------------------------------------------------------------
# 6. Goal first; the person starts the moment the goal is ACCEPTED
#    (headon_send_goal.py retries the handshake, see its docstring)
# ----------------------------------------------------------------
echo "[headon_F] Sending the goal ($GOAL_X, $GOAL_Y)..."
FLAG="$LOG_DIR/goal_accepted"
rm -f "$FLAG"
python3 "$WS/headon_send_goal.py" "$GOAL_X" "$GOAL_Y" "$GOAL_TIMEOUT" "$FLAG" \
  > "$BAG/goal_result.txt" 2>&1 &
GOAL_PID=$!
for i in $(seq 1 300); do
  [ -f "$FLAG" ] && break
  kill -0 "$GOAL_PID" 2>/dev/null || break
  sleep 0.1
done
if [ -f "$FLAG" ]; then
  echo "[headon_F] Goal accepted - starting the person."
  # Two walkers: both hold until one shared sim time, so they start together.
  START_AT=0.0
  if [ -n "$PERSON2_Y" ]; then
    START_AT=$(timeout 20 python3 -c "
import rclpy
from rosgraph_msgs.msg import Clock
from rclpy.qos import qos_profile_sensor_data
rclpy.init(); n = rclpy.create_node('headon_clock_probe'); got = []
n.create_subscription(Clock, '/clock', lambda m: got.append(m.clock.sec + m.clock.nanosec * 1e-9), qos_profile_sensor_data)
while not got: rclpy.spin_once(n, timeout_sec=0.2)
print(f'{got[0] + 3.0:.2f}')
" 2>/dev/null || echo 0.0)
    echo "[headon_F] Both walkers start at sim time $START_AT s."
  fi
  start mover python3 "$MOVER" --ros-args -p use_sim_time:=true \
    -p world_name:="$WORLD" -p model_name:=person_1 \
    -p point_a:="[$PERSON_X0, $PERSON_Y]" -p point_b:="[$PERSON_X1, $PERSON_Y]" \
    -p speed:="$PERSON_SPEED" -p start_sim_time:="$START_AT"
  if [ -n "$PERSON2_Y" ]; then
    start mover2 python3 "$MOVER" --ros-args -p use_sim_time:=true \
      -p world_name:="$WORLD" -p model_name:=person_2 \
      -p point_a:="[$PERSON2_X0, $PERSON2_Y]" -p point_b:="[$PERSON2_X1, $PERSON2_Y]" \
      -p speed:="$PERSON2_SPEED" -p ground_truth_topic:=/person2_ground_truth \
      -p start_sim_time:="$START_AT"
  fi
fi
set +e
wait "$GOAL_PID"
set -e
cat "$BAG/goal_result.txt"

# ----------------------------------------------------------------
# 7. Result
# ----------------------------------------------------------------
if grep -q "status: SUCCEEDED" "$BAG/goal_result.txt"; then
  echo "[headon_F] RESULT: goal succeeded."
elif grep -q "TIMEOUT" "$BAG/goal_result.txt"; then
  echo "[headon_F] RESULT: goal TIMED OUT after $GOAL_TIMEOUT s (wall)."
elif grep -q "NOT_ACCEPTED" "$BAG/goal_result.txt"; then
  echo "[headon_F] RESULT: goal never accepted - person not started, trial INVALID."
elif grep -q "EARLY_ABORT" "$BAG/goal_result.txt"; then
  # Renamed so a batch runner that keys on goal_result.txt reruns the trial.
  mv "$BAG/goal_result.txt" "$BAG/goal_result_invalid.txt"
  echo "[headon_F] RESULT: Nav2 aborted right after the goal was accepted (stack, not encounter) - trial INVALID."
else
  echo "[headon_F] RESULT: goal did NOT succeed (see $BAG/goal_result.txt)."
fi
sleep 2
echo "[headon_F] Analyse with: python3 $WS/analysis/analyse_headon_F.py $BAG"
