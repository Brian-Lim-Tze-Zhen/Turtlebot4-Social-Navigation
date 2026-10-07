#!/bin/bash
# run_conv_F_gskip_batch.sh - the 12 group-skip trials (6 narrow, 6 wide), no RViz.
#
# Run inside the container, from the workspace root:
#   docker exec -it thesis_social_nav bash
#   cd /root/thesis_social_navigation_ws && ./run_conv_F_gskip_batch.sh
#
# Bags: conv_F_gskip_narrow_trial1..6 and conv_F_gskip_wide_trial1..6
# (a repeated attempt gets the suffix _a2, _a3). A run counts only if it
# finished, the bag holds at least MIN_ZONE /social_zone_map messages (the
# zone node got the map) and the goal succeeded; otherwise it is repeated, up
# to MAX_TRIES times. One line per run goes to logs/gskip_batch_summary.txt.
#
# Overridable: N (trials per case, default 6), MAX_TRIES (3), MIN_ZONE (20),
# PREFIX (bag name prefix, default conv_F_gskip), EXPECT_MD5 (first 8 characters
# of the critic library hash that every counted bag must carry).

WS=/root/thesis_social_navigation_ws
cd "$WS" || exit 1
export CFG="$WS/config/social_nav2_headon_F_hwreq_block_blockedhold_groupskip_sim.yaml"
N="${N:-6}"; MAX_TRIES="${MAX_TRIES:-3}"; MIN_ZONE="${MIN_ZONE:-20}"
PREFIX="${PREFIX:-conv_F_gskip}"          # bag name prefix
EXPECT_MD5="${EXPECT_MD5:-}"              # first characters of the critic library hash; a bag with another hash is repeated
SUMMARY="$WS/logs/gskip_batch_summary.txt"
mkdir -p "$WS/logs"
echo "# batch started $(date '+%F %T'), config $(basename "$CFG")" >> "$SUMMARY"

for case_ in narrow wide; do
  for i in $(seq 1 "$N"); do
    tries=0; ok=false
    while [ "$tries" -lt "$MAX_TRIES" ] && [ "$ok" != true ]; do
      tries=$((tries + 1))
      name="${PREFIX}_${case_}_trial${i}"
      [ "$tries" -gt 1 ] && name="${name}_a${tries}"
      echo "=== $name (attempt $tries of $MAX_TRIES) ==="
      CASE="$case_" ./run_conv_F_trial.sh "$name"
      rc=$?
      bag="$WS/bags/$name"
      zone=$(ros2 bag info "$bag" 2>/dev/null | grep -o 'social_zone_map.*Count: [0-9]*' | grep -o '[0-9]*$')
      zone="${zone:-0}"
      goal=$(grep -i "finished" "$bag/goal_result.txt" 2>/dev/null | tail -1 | grep -o '[A-Z]*$')
      md5=$(cut -c1-8 "$bag/critic_so_md5.txt" 2>/dev/null)
      lanes=$(grep -c "lane rule now follows" "$WS/logs/$name/nav2.log" 2>/dev/null)
      if [ "$rc" -eq 0 ] && [ "$zone" -ge "$MIN_ZONE" ] && [ "$goal" = SUCCEEDED ] \
       && { [ -z "$EXPECT_MD5" ] || [ "$md5" = "$EXPECT_MD5" ]; }; then
        ok=true; verdict=OK
      else
        verdict="REPEAT (rc=$rc zone=$zone goal=${goal:-none})"
      fi
      echo "$name | $verdict | zone msgs $zone | goal ${goal:-none} | critic md5 ${md5:-none} | lane lines ${lanes:-0}" | tee -a "$SUMMARY"
      # leftovers from a failed run must not block the next launch
      if [ "$ok" != true ]; then
        pkill -f "gz sim" 2>/dev/null; pkill -f rviz2 2>/dev/null; sleep 5
      fi
    done
    [ "$ok" = true ] || echo "$case_ trial $i: no valid run after $MAX_TRIES attempts" | tee -a "$SUMMARY"
  done
done
echo "# batch finished $(date '+%F %T')" >> "$SUMMARY"
echo "Summary: $SUMMARY"
