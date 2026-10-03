#!/bin/bash
# run_headon_F_batch.sh <prefix> <n> [ENV=val ...]
#
# n head-on trials (<prefix>_trial1..n) with the head-on AVOIDANCE config,
# each retried once if it did not produce a valid result, then mean +/- SD.
# Extra ENV=val arguments are passed to run_headon_F_trial.sh, e.g.
#   ./run_headon_F_batch.sh headon_avoid_v9 3 YOLO_IMGSZ=640 YOLO_MIN_CONF=0.35 \
#       MAX_PERSON_RANGE=11.5 PASS_BLOCK=2.0
# CFG=... overrides the config.
cd /root/thesis_social_navigation_ws
p=$1; N=$2; shift 2
for i in $(seq 1 "$N"); do
  n=${p}_trial$i
  for a in 1 2; do
    env CFG=$PWD/config/social_nav2_headon_F_avoid_sim.yaml "$@" ./run_headon_F_trial.sh "$n" 2>&1 | grep "RESULT\|ERROR"
    sleep 8
    [ -f "bags/$n/goal_result.txt" ] && break
    rm -rf "bags/$n"
  done
done
source /opt/ros/jazzy/setup.bash
python3 analysis/analyse_headon_F.py "bags/${p}_trial*" 2>&1 | grep -v "^\[" > "analysis/${p}_report.txt"
grep "===\|min centre\|goal reached" "analysis/${p}_report.txt" | grep -v Aggregate | paste - - - | awk '{print $2, $6, $10}'
sed -n '/Aggregate/,$p' "analysis/${p}_report.txt"
