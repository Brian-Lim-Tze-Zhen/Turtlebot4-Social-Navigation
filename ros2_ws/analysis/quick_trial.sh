#!/bin/bash
# quick_trial.sh <bag name> - key head-on metrics + side/noghost events + global-plan side for one trial (8 Oct 2026)
cd /root/thesis_social_navigation_ws && source /opt/ros/jazzy/setup.bash
python3 analysis/analyse_headon_F.py bags/$1 2>&1 | grep -E "min centre|surface|passed|committed"
python3 analysis/hwsync_pass_timeline.py bags/$1 | grep -E "closest|event"
python3 analysis/plan_side_check.py bags/$1 2>&1 | grep "$1"
