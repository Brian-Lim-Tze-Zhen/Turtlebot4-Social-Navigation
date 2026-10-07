#!/bin/bash
# quick_pair_run.sh <bag name> - one-line result of a pair or head-on run (inside the container)
cd /root/thesis_social_navigation_ws; source /opt/ros/jazzy/setup.bash
b=bags/$1
if [[ $1 == headon* ]]; then
  python3 analysis/analyse_headon_F.py $b 2>/dev/null | grep -E "goal reached|min centre distance|robot passed|gap when committed|stopped / spin|wz sign|first detection range" | tr -s ' ' | tr '\n' ';' | sed "s/^/$1: /"; echo
else
  python3 analysis/analyse_F_narrow.py $b 2>/dev/null > /tmp/qa.txt
  r=$(grep -E "goal reached|time start->goal|stopped time|spin-in-place" /tmp/qa.txt | tr -s ' ' | tr '\n' ';')
  d=$(grep -E "min centre" /tmp/qa.txt | grep -o "min centre [0-9.]* m" | tr '\n' ' ')
  y=$(python3 - "$b" <<'PY'
import csv,sys
r=list(csv.DictReader(open(sys.argv[1]+"/trajectory.csv")))
p=min(r,key=lambda q:abs(float(q["x_ref"])-3.0)); ys=[abs(float(q["y_ref"])) for q in r if 0<=float(q["x_ref"])<=5.5]
print(f"y at pair {float(p['y_ref']):+.2f}, max|y| {max(ys):.2f}")
PY
)
  echo "$1: $r $d; $y; lane lines $(grep -c 'lane rule now follows' logs/$1/nav2.log 2>/dev/null)"
fi
