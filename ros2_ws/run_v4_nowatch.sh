#!/bin/bash
# run_v4_nowatch.sh - next test of the sustained-approach critic (version 4), run inside the container:
#   docker exec -it thesis_social_nav bash
#   cd /root/thesis_social_navigation_ws && ./run_v4_nowatch.sh
# Sets (bags):
#   1. narrow pair, v4 without watch-first, 6 trials   -> conv_F_v4nw_narrow_trialN
#   2. head-on corridor, v4 without watch-first, 5     -> headon_v4nw_corr_trialN
#   3. wide pair, v4 (with watch-first), 6             -> conv_F_v4_wide_trialN
# A trial counts if the goal succeeded and the bag carries the installed critic (version 4); a pair trial also
# needs the zone map. Otherwise it is repeated up to 3 times (suffix _a2, _a3). One line per attempt goes to
# logs/v4_nowatch_summary.txt. No rebuild: the installed critic must already be version 4 (it is, since 7 Oct).
WS=/root/thesis_social_navigation_ws; cd "$WS" || exit 1
source /opt/ros/jazzy/setup.bash; source install/setup.bash
S="$WS/logs/v4_nowatch_summary.txt"
grep -q approachConfirmed src/social_critic/src/social_critic.cpp || { echo "ERROR: critic source is not version 4" | tee -a "$S"; exit 1; }
H=$(md5sum install/social_critic/lib/libsocial_critic.so | cut -c1-8)
echo "# started $(date '+%F %T'), critic $H" >> "$S"
NOWATCH="$WS/config/social_nav2_headon_F_hwreq_block_blockedhold_approach_nowatch_sim.yaml"
APPROACH="$WS/config/social_nav2_headon_F_hwreq_block_blockedhold_approach_sim.yaml"
run() {   # run <script> <name> <cfg> [env...]
  local script=$1 base=$2 cfg=$3; shift 3
  local tries=0 ok=false name
  while [ $tries -lt 3 ] && [ "$ok" != true ]; do
    tries=$((tries + 1)); name=$base; [ $tries -gt 1 ] && name="${base}_a${tries}"
    echo "=== $name (attempt $tries) ==="
    env CFG="$cfg" "$@" ./$script "$name"; rc=$?
    goal=$(grep -i "finished" "bags/$name/goal_result.txt" 2>/dev/null | tail -1 | grep -o '[A-Z]*$')
    md5=$(cut -c1-8 "bags/$name/critic_so_md5.txt" 2>/dev/null | head -1)
    lanes=$(grep -c "lane rule now follows" "logs/$name/nav2.log" 2>/dev/null)
    zone=$(ros2 bag info "bags/$name" 2>/dev/null | grep -o 'social_zone_map.*Count: [0-9]*' | grep -o '[0-9]*$')
    if [ "$rc" -eq 0 ] && [ "$goal" = SUCCEEDED ] && [ "$md5" = "$H" ]; then ok=true; v=OK; else v="REPEAT (rc=$rc)"; fi
    if [ "$script" = run_conv_F_trial.sh ] && [ "${zone:-0}" -lt 20 ]; then ok=false; v="REPEAT (zone ${zone:-0})"; fi
    echo "$name | $v | goal ${goal:-none} | critic ${md5:-none} | lane lines ${lanes:-0} | zone ${zone:--}" | tee -a "$S"
    [ "$ok" = true ] || { pkill -f "gz sim"; sleep 5; }
  done
}
for i in 1 2 3 4 5 6; do run run_conv_F_trial.sh conv_F_v4nw_narrow_trial$i "$NOWATCH" CASE=narrow; done
for i in 1 2 3 4 5; do run run_headon_F_trial.sh headon_v4nw_corr_trial$i "$NOWATCH"; done
for i in 1 2 3 4 5 6; do run run_conv_F_trial.sh conv_F_v4_wide_trial$i "$APPROACH" CASE=wide; done
echo "# finished $(date '+%F %T')" >> "$S"
