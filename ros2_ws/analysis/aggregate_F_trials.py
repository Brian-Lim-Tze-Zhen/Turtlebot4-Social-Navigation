#!/usr/bin/env python3
"""aggregate_F_trials.py - mean +/- sd over trials analysed by analyse_F_narrow.py.

Usage:
    python3 aggregate_F_trials.py "conv_F_wide_trial*"

Reads <bag>/trajectory.csv (written by analyse_F_narrow.py) for every bag
matching the pattern in BAG_DIR. Uses the SAME constants and definitions as
analyse_F_narrow.py, so wide and narrow numbers are directly comparable:
  run start = first |vx| > MOVE_V, goal = first sample within GOAL_TOL,
  centre = robot centre to true person pose, surface = centre - PERSON_R - ROBOT_R.
"""
import csv
import glob
import math
import os
import statistics
import sys

BAG_DIR = "/root/thesis_social_navigation_ws/bags"
GOAL = (6.0, 0.0)
GOAL_TOL = 0.30
ROBOT_R = 0.189
PERSON_R = 0.25
STALL_V = 0.02
MOVE_V = 0.05


def one(bag):
    path = os.path.join(bag, "trajectory.csv")
    if not os.path.exists(path):
        return None, "no trajectory.csv (run analyse_F_narrow.py on it first)"
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, "empty trajectory.csv"
    dcols = [c for c in rows[0] if c.startswith("d_")]
    R = [(float(r["t"]), float(r["x_ref"]), float(r["y_ref"]), float(r["vx"]),
          float(r["wz"]), [float(r[c]) for c in dcols]) for r in rows]
    t0 = next((r[0] for r in R if abs(r[3]) > MOVE_V), R[0][0])
    tg = next((r[0] for r in R if math.hypot(r[1] - GOAL[0], r[2] - GOAL[1]) < GOAL_TOL), None)
    t_end = tg if tg is not None else R[-1][0]
    win = [r for r in R if t0 <= r[0] <= t_end]
    stall = sum(b[0] - a[0] for a, b in zip(win, win[1:])
                if abs(a[3]) < STALL_V and abs(a[4]) < 0.1)
    mins = [min(r[5][i] for r in R) for i in range(len(dcols))]
    return {
        "reached": tg is not None,
        "time_s": t_end - t0,
        "stopped_s": stall,
        "min_centre_m": min(mins),
        "min_surface_m": min(mins) - PERSON_R - ROBOT_R,
        "per_person": dict(zip(dcols, mins)),
    }, None


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    bags = sorted(b for b in glob.glob(os.path.join(BAG_DIR, sys.argv[1])) if os.path.isdir(b))
    if not bags:
        sys.exit(f"no bags match {sys.argv[1]} in {BAG_DIR}")
    ok = []
    for b in bags:
        r, err = one(b)
        name = os.path.basename(b)
        if err:
            print(f"{name}: SKIPPED - {err}")
            continue
        pp = ", ".join(f"{k[2:]} {v:.3f}" for k, v in r["per_person"].items())
        print(f"{name}: reached={r['reached']} time={r['time_s']:.1f}s "
              f"stopped={r['stopped_s']:.1f}s min_centre={r['min_centre_m']:.3f} "
              f"min_surface={r['min_surface_m']:+.3f}  [{pp}]")
        if r["reached"]:
            ok.append(r)
    print(f"\nreached goal: {len(ok)}/{len(bags)}")
    if len(ok) >= 2:
        for k in ("min_centre_m", "min_surface_m", "time_s", "stopped_s"):
            v = [r[k] for r in ok]
            print(f"{k:14s} {statistics.mean(v):.3f} +/- {statistics.stdev(v):.3f} (n={len(v)})")


if __name__ == "__main__":
    main()
