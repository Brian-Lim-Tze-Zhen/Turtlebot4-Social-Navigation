#!/bin/bash
# run_hwsync_split.sh - separates the two ported hardware measures (8 Oct 2026). Run inside the container.
# Both start from the v4 approach config (baseline headon_v4_corr_trial1-5), head-on corridor, 5 trials each:
#   A  headon_sidegoal_corr_trialN : ..._approach_sidegoal_sim.yaml (side_ref_goal true), old predictor
#   B  headon_noghost_corr_trialN  : ..._approach_sim.yaml unchanged, KF_SCRIPT=human_kf_predictor_lidar_noghost.py
# A trial counts if the goal succeeded, the bag carries the current critic hash and the log shows the
# intended variant (critic side reference, predictor); otherwise it is repeated (up to 3 attempts, _a2, _a3).
WS=/root/thesis_social_navigation_ws; cd "$WS" || exit 1
source /opt/ros/jazzy/setup.bash; source install/setup.bash
S="$WS/logs/hwsync_split_summary.txt"
H=$(md5sum install/social_critic/lib/libsocial_critic.so | cut -c1-8)
APPROACH="$WS/config/social_nav2_headon_F_hwreq_block_blockedhold_approach_sim.yaml"
SIDEGOAL="$WS/config/social_nav2_headon_F_hwreq_block_blockedhold_approach_sidegoal_sim.yaml"
echo "# split started $(date '+%F %T'), critic $H" >> "$S"
run() {   # run <base name> <cfg> <expected side line> <expected noghost count> [env...]
  local base=$1 cfg=$2 side=$3 ng=$4; shift 4
  local tries=0 ok=false name
  while [ $tries -lt 3 ] && [ "$ok" != true ]; do
    tries=$((tries + 1)); name=$base; [ $tries -gt 1 ] && name="${base}_a${tries}"
    env CFG="$cfg" "$@" ./run_headon_F_trial.sh "$name"; rc=$?
    goal=$(grep -i "finished" "bags/$name/goal_result.txt" 2>/dev/null | tail -1 | grep -o '[A-Z]*$')
    md5=$(cut -c1-8 "bags/$name/critic_so_md5.txt" 2>/dev/null | head -1)
    sideok=$(grep -c "pass side measured against the $side" "logs/$name/nav2.log" 2>/dev/null)
    noghost=$(grep -c "NOGHOST: a track" "logs/$name/perception.log" 2>/dev/null)
    drops=$(grep -c "NOGHOST: dropped" "logs/$name/perception.log" 2>/dev/null)
    if [ "$rc" -eq 0 ] && [ "$goal" = SUCCEEDED ] && [ "$md5" = "$H" ] && [ "${sideok:-0}" -gt 0 ] \
       && [ "$(( ${noghost:-0} > 0 ))" -eq "$ng" ]; then ok=true; v=OK; else v="REPEAT (rc=$rc)"; fi
    echo "$name | $v | goal ${goal:-none} | critic ${md5:-none} | side ref ok ${sideok:-0} | noghost start ${noghost:-0} drops ${drops:-0}" >> "$S"
    [ "$ok" = true ] || { pkill -f "gz sim"; sleep 5; }
  done
}
for i in 1 2 3 4 5; do run headon_sidegoal_corr_trial$i "$SIDEGOAL" "robot-goal line" 0; done
for i in 1 2 3 4 5; do run headon_noghost_corr_trial$i "$APPROACH" "start of the global path" 1 KF_SCRIPT=human_kf_predictor_lidar_noghost.py; done
echo "# split finished $(date '+%F %T')" >> "$S"
