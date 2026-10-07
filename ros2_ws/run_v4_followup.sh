#!/bin/bash
# run_v4_followup.sh - steps 1 to 3 of the sustained-approach test (7 Oct 2026). Run inside the container,
# after the version 4 source (social_critic.cpp/.hpp with lane_approach_window_s) is in src/social_critic.
#   1. build the critic, record its hash
#   2. narrow pair, 6 trials, approach config            -> bags conv_F_v4_narrow_trialN
#   3. head-on corridor, 5 trials with the final config   -> bags headon_final_corr_trialN
#      and 5 trials with the approach config              -> bags headon_v4_corr_trialN
# A trial counts if the goal succeeded and the bag carries the hash of step 1; otherwise it is repeated
# (up to 3 attempts, suffix _a2, _a3). One line per attempt goes to logs/v4_followup_summary.txt.
WS=/root/thesis_social_navigation_ws; cd "$WS" || exit 1
source /opt/ros/jazzy/setup.bash
S="$WS/logs/v4_followup_summary.txt"
colcon build --packages-select social_critic > logs/v4_build.log 2>&1 || { echo "build failed" >> "$S"; exit 1; }
source install/setup.bash
H=$(md5sum install/social_critic/lib/libsocial_critic.so | cut -c1-8)
grep -q approachConfirmed src/social_critic/src/social_critic.cpp || { echo "source is not version 4" >> "$S"; exit 1; }
echo "# follow-up started $(date '+%F %T'), critic $H" >> "$S"
APPROACH="$WS/config/social_nav2_headon_F_hwreq_block_blockedhold_approach_sim.yaml"
FINAL="$WS/config/social_nav2_headon_F_hwreq_block_blockedhold_sim.yaml"
run() {   # run <script> <name> <cfg> [env...]
  local script=$1 base=$2 cfg=$3; shift 3
  local tries=0 ok=false name
  while [ $tries -lt 3 ] && [ "$ok" != true ]; do
    tries=$((tries + 1)); name=$base; [ $tries -gt 1 ] && name="${base}_a${tries}"
    env CFG="$cfg" "$@" ./$script "$name"; rc=$?
    goal=$(grep -i "finished" "bags/$name/goal_result.txt" 2>/dev/null | tail -1 | grep -o '[A-Z]*$')
    md5=$(cut -c1-8 "bags/$name/critic_so_md5.txt" 2>/dev/null | head -1)
    lanes=$(grep -c "lane rule now follows" "logs/$name/nav2.log" 2>/dev/null)
    zone=$(ros2 bag info "bags/$name" 2>/dev/null | grep -o 'social_zone_map.*Count: [0-9]*' | grep -o '[0-9]*$')
    if [ "$rc" -eq 0 ] && [ "$goal" = SUCCEEDED ] && [ "$md5" = "$H" ]; then ok=true; v=OK; else v="REPEAT (rc=$rc)"; fi
    if [ "$script" = run_conv_F_trial.sh ] && [ "${zone:-0}" -lt 20 ]; then ok=false; v="REPEAT (zone ${zone:-0})"; fi
    echo "$name | $v | goal ${goal:-none} | critic ${md5:-none} | lane lines ${lanes:-0} | zone ${zone:--}" >> "$S"
    [ "$ok" = true ] || { pkill -f "gz sim"; sleep 5; }
  done
}
for i in 1 2 3 4 5 6; do run run_conv_F_trial.sh conv_F_v4_narrow_trial$i "$APPROACH" CASE=narrow; done
for i in 1 2 3 4 5; do run run_headon_F_trial.sh headon_final_corr_trial$i "$FINAL"; done
for i in 1 2 3 4 5; do run run_headon_F_trial.sh headon_v4_corr_trial$i "$APPROACH"; done
echo "# follow-up finished $(date '+%F %T')" >> "$S"
