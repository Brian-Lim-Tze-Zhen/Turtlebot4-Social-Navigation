#!/bin/bash
# run_hwsync_headon.sh - head-on corridor, 5 trials with the HW-SYNC config (8 Oct 2026). Run inside the container.
# Config social_nav2_headon_F_hwreq_block_blockedhold_approach_hwsync_sim.yaml = the v4 approach config plus
# what the robot changed after the port: transform tolerances, GoalAngleCritic 0.25, SocialCritic side_ref_goal,
# and (chosen by run_headon_F_trial.sh for *hwsync*) human_kf_predictor_lidar_noghost.py.
# Compare with headon_v4_corr_trial1-5 (same scenario, approach config, critic ee5b22ce).
# A trial counts if the goal succeeded, the bag carries the current critic hash, the critic logged the
# robot-goal line and the noghost predictor started; otherwise it is repeated (up to 3 attempts, _a2, _a3).
WS=/root/thesis_social_navigation_ws; cd "$WS" || exit 1
source /opt/ros/jazzy/setup.bash; source install/setup.bash
S="$WS/logs/hwsync_headon_summary.txt"
H=$(md5sum install/social_critic/lib/libsocial_critic.so | cut -c1-8)
grep -q side_ref_goal_ src/social_critic/src/social_critic.cpp || { echo "source has no side_ref_goal" >> "$S"; exit 1; }
CFG_HS="$WS/config/social_nav2_headon_F_hwreq_block_blockedhold_approach_hwsync_sim.yaml"
echo "# hwsync head-on started $(date '+%F %T'), critic $H" >> "$S"
for i in 1 2 3 4 5; do
  tries=0; ok=false
  while [ $tries -lt 3 ] && [ "$ok" != true ]; do
    tries=$((tries + 1)); name=headon_hwsync_corr_trial$i; [ $tries -gt 1 ] && name="${name}_a${tries}"
    env CFG="$CFG_HS" ./run_headon_F_trial.sh "$name"; rc=$?
    goal=$(grep -i "finished" "bags/$name/goal_result.txt" 2>/dev/null | tail -1 | grep -o '[A-Z]*$')
    md5=$(cut -c1-8 "bags/$name/critic_so_md5.txt" 2>/dev/null | head -1)
    lanes=$(grep -c "lane rule now follows" "logs/$name/nav2.log" 2>/dev/null)
    goalline=$(grep -c "robot-goal line" "logs/$name/nav2.log" 2>/dev/null)
    noghost=$(grep -c "NOGHOST: a track" "logs/$name/perception.log" 2>/dev/null)
    drops=$(grep -c "NOGHOST: dropped" "logs/$name/perception.log" 2>/dev/null)
    if [ "$rc" -eq 0 ] && [ "$goal" = SUCCEEDED ] && [ "$md5" = "$H" ] && [ "${goalline:-0}" -gt 0 ] && [ "${noghost:-0}" -gt 0 ]; then
      ok=true; v=OK; else v="REPEAT (rc=$rc)"; fi
    echo "$name | $v | goal ${goal:-none} | critic ${md5:-none} | goal-line log ${goalline:-0} | noghost start ${noghost:-0} drops ${drops:-0} | lane lines ${lanes:-0}" >> "$S"
    [ "$ok" = true ] || { pkill -f "gz sim"; sleep 5; }
  done
done
echo "# hwsync head-on finished $(date '+%F %T')" >> "$S"
