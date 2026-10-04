#!/bin/bash
# run_scenario.sh <scenario> [bag_name]
#
# One command per demo scenario: fresh Gazebo + Nav2 + perception, the goal,
# a recorded bag, teardown, then a short result summary. Picks the trial
# script and settings; the work is done by run_headon_F_trial.sh and
# run_conv_F_trial.sh.
#
# Scenarios:
#   headon               one person walks toward the robot at 1.2 m/s in a
#                        2.5 m corridor; the robot moves aside and passes
#   headon_blocked       two people stand across the corridor; the robot
#                        stops and beeps until they walk away, then goes on
#   conversation_wide    two people talking in open space; the robot goes
#                        around the pair
#   conversation_narrow  the same pair in a 2.0 m corridor; the robot passes
#                        between them
#   queue                four people stand in a line across the robot's way; the
#                        robot goes around the end of the line, not through it
#
# Run inside the container:
#   ./run_scenario.sh headon
#   ./run_scenario.sh conversation_narrow my_bag_name
#
# Environment:
#   SHOW_RVIZ=false HEADLESS=true   no windows (default: RViz and Gazebo shown)
#   CFG=<nav2 params file>          override the scenario's config
#   STAND_S=<sim s>                 headon_blocked: how long the pair stands (default 20)
#   TRIES=<n>                       attempts if the stack fails to start (default 3)
set -e

WS=/root/thesis_social_navigation_ws
SCENARIOS="headon headon_blocked conversation_wide conversation_narrow queue"

SCEN=$1
if [ -z "$SCEN" ] || ! echo " $SCENARIOS " | grep -q " $SCEN "; then
  echo "usage: $0 <scenario> [bag_name]" >&2
  echo "scenarios: $SCENARIOS" >&2
  exit 1
fi
NAME=${2:-demo_${SCEN}_$(date +%Y%m%d_%H%M%S)}

export SHOW_RVIZ="${SHOW_RVIZ:-true}"
export HEADLESS="${HEADLESS:-false}"
# Head-on scenarios: the avoidance config with the stop-and-beep hold and no
# swerve for standing people.
HEADON_CFG="$WS/config/social_nav2_headon_F_hwreq_block_blockedhold_sim.yaml"

cd "$WS"
echo "[scenario] $SCEN -> bag $NAME (RViz $SHOW_RVIZ, Gazebo window $([ "$HEADLESS" = true ] && echo off || echo on))"

run_trial() {
  case "$SCEN" in
    headon)
      CFG="${CFG:-$HEADON_CFG}" ./run_headon_F_trial.sh "$NAME"
      ;;
    headon_blocked)
      # The pair stands at x = 2.0, 0.5 m either side of the centre line, then
      # walks away to x = 9.5, beyond the goal at x = 8.0.
      CFG="${CFG:-$HEADON_CFG}" WORLD=corridor_two_human \
        MAP="$WS/maps/corridor_headon_aligned.yaml" \
        PERSON_X0=2.0 PERSON_X1=9.5 PERSON_Y=0.5 PERSON2_Y=-0.5 \
        PERSON_START_DELAY="${STAND_S:-20}" GOAL_TIMEOUT="${GOAL_TIMEOUT:-300}" \
        ./run_headon_F_trial.sh "$NAME"
      ;;
    conversation_wide)
      CASE=wide ./run_conv_F_trial.sh "$NAME"
      ;;
    conversation_narrow)
      CASE=narrow ./run_conv_F_trial.sh "$NAME"
      ;;
    queue)
      # Goal (6, -3): the straight line to it crosses the queue between the
      # third and fourth person.
      export GOAL_Y="${GOAL_Y:--3.0}"
      CASE=queue ./run_conv_F_trial.sh "$NAME"
      ;;
  esac
}

# The stack does not always come up (Nav2 lifecycle hang, filter server, goal
# never acknowledged). A trial that produced no goal result is set aside as
# <bag>_invalid<k> and started again, up to TRIES times in all.
TRIES="${TRIES:-3}"
trap 'exit 130' INT TERM      # Ctrl+C stops the scenario, it is not a failed attempt
for k in $(seq 1 "$TRIES"); do
  set +e
  run_trial
  rc=$?
  set -e
  [ "$rc" -ge 128 ] && exit "$rc"
  [ -f "bags/$NAME/goal_result.txt" ] && break
  echo "[scenario] Attempt $k of $TRIES did not produce a result (start-up failure)."
  [ -e "bags/$NAME" ] && mv "bags/$NAME" "bags/${NAME}_invalid$k"
  [ -e "logs/$NAME" ] && mv "logs/$NAME" "logs/${NAME}_invalid$k"
  [ "$k" = "$TRIES" ] && { echo "[scenario] Giving up after $TRIES attempts." >&2; exit 1; }
  echo "[scenario] Starting again..."
  sleep 8
done

# ----------------------------------------------------------------
# Result summary
# ----------------------------------------------------------------
source /opt/ros/jazzy/setup.bash
echo
echo "================ $SCEN : $NAME ================"
case "$SCEN" in
  headon*)
    python3 analysis/analyse_headon_F.py "bags/$NAME" 2>&1 \
      | grep -E "goal reached|time to goal|min centre|person 2 min|stopped / spin" || true
    if [ "$SCEN" = headon_blocked ]; then
      echo "Beeps and block events (sim seconds after the robot started):"
      python3 analysis/blocked_beep_timeline.py "bags/$NAME/data" | grep -F "**" || echo "  none"
    fi
    ;;
  queue)
    python3 analysis/queue_run_summary.py "bags/$NAME" "${GOAL_X:-6.0}" "$GOAL_Y" 2>&1 \
      | grep -vE "^bag:|zone mask|^    [+-]" || true
    ;;
  conversation*)
    python3 analysis/analyse_F_narrow.py "bags/$NAME" 2>&1 \
      | grep -E "goal reached|time start|stopped time|spin-in-place|min centre|msgs .* narrow|through the gap" || true
    ;;
esac
echo "Bag and provenance: $WS/bags/$NAME    Logs: $WS/logs/$NAME"
