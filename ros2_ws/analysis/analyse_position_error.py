#!/usr/bin/env python3
"""
Position-error decomposition by walking direction.

Compares the KF's CURRENT position estimate against ground truth at the SAME
timestamp — no extrapolation involved — so whatever this reports is pure
position error, not prediction error.

WHY SPLIT BY DIRECTION
----------------------
Hypothesis: get_valid_depth_from_bbox() samples the torso patch, which measures
the surface of the person FACING THE CAMERA, while /person_ground_truth reports
the model origin. For a human mesh a few tens of cm deep, that produces a
systematic offset toward the robot.

The discriminating test is the sign. If the offset is geometric (near-face vs
origin), the signed error along the walking axis FLIPS when the person walks
away versus toward the robot, because the camera always sees the near face
regardless of travel direction. If the sign is the same in both directions, the
cause is not this geometry and the hypothesis is wrong.

The error is also projected onto the camera bearing (robot -> person), since a
surface-vs-origin offset acts along the line of sight, not along the map axes.

Usage:
    python3 analyse_position_error.py PRED_CSV GT_CSV [ROBOT_X ROBOT_Y]

    ROBOT_X/ROBOT_Y default to the head-on spawn (-3.0, 0.0) and are only used
    for the line-of-sight projection.

Columns expected:
    prediction   : sim_time,track_id,conf,x,y,vx,vy,pred_x,pred_y,horizon
    ground truth : sim_time,person_index,x,y,z
"""

import csv
import math
import sys


# Ground-truth samples slower than this are treated as paused (endpoint hold)
# and excluded, since direction is undefined there.
MOVING_SPEED_MIN = 0.15   # m/s


def load_predictions(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "t": float(r["sim_time"]),
                    "x": float(r["x"]),
                    "y": float(r["y"]),
                })
            except (ValueError, KeyError):
                continue
    return rows


def load_ground_truth(path, person_index=0):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                if int(float(r["person_index"])) != person_index:
                    continue
                rows.append((float(r["sim_time"]), float(r["x"]), float(r["y"])))
            except (ValueError, KeyError):
                continue
    rows.sort(key=lambda e: e[0])
    return rows


def interp_gt(gt, t):
    if not gt or t < gt[0][0] or t > gt[-1][0]:
        return None

    lo, hi = 0, len(gt) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if gt[mid][0] <= t:
            lo = mid
        else:
            hi = mid

    t0, x0, y0 = gt[lo]
    t1, x1, y1 = gt[hi]

    if t1 == t0:
        return x0, y0

    f = (t - t0) / (t1 - t0)
    return x0 + f * (x1 - x0), y0 + f * (y1 - y0)


def gt_velocity(gt, t, window=0.3):
    """Local ground-truth velocity by central difference."""
    a = interp_gt(gt, t - window / 2.0)
    b = interp_gt(gt, t + window / 2.0)
    if a is None or b is None:
        return None
    return (b[0] - a[0]) / window, (b[1] - a[1]) / window


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def stdev(v):
    if len(v) < 2:
        return float("nan")
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def report(label, dx, dy, los, dist):
    if not dx:
        print(f"  {label}: no samples")
        return
    print(f"  {label}  (n={len(dx)})")
    print(f"    signed dx        : {mean(dx):+.3f} m   (sd {stdev(dx):.3f})")
    print(f"    signed dy        : {mean(dy):+.3f} m   (sd {stdev(dy):.3f})")
    print(f"    along line-of-sight: {mean(los):+.3f} m   (sd {stdev(los):.3f})")
    print(f"      (negative = estimate is NEARER the robot than ground truth)")
    print(f"    mean abs distance: {mean(dist):.3f} m")


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__)
        sys.exit(1)

    pred_path, gt_path = args[0], args[1]
    robot_x = float(args[2]) if len(args) > 2 else -3.0
    robot_y = float(args[3]) if len(args) > 3 else 0.0

    preds = load_predictions(pred_path)
    gt = load_ground_truth(gt_path)

    if not preds or not gt:
        print("  missing prediction or ground-truth rows")
        sys.exit(1)

    # toward = person moving toward the robot, away = moving away
    groups = {
        "TOWARD robot": ([], [], [], []),
        "AWAY from robot": ([], [], [], []),
    }
    all_dx, all_dy, all_los, all_dist = [], [], [], []
    n_paused = 0

    for p in preds:
        actual = interp_gt(gt, p["t"])
        if actual is None:
            continue

        gx, gy = actual
        ex = p["x"] - gx
        ey = p["y"] - gy
        d = math.hypot(ex, ey)

        # Project the error onto the robot->person bearing. A surface-vs-origin
        # offset acts along this line, not along the map axes.
        bx = gx - robot_x
        by = gy - robot_y
        bnorm = math.hypot(bx, by)
        los = (ex * bx + ey * by) / bnorm if bnorm > 1e-6 else 0.0

        all_dx.append(ex)
        all_dy.append(ey)
        all_los.append(los)
        all_dist.append(d)

        v = gt_velocity(gt, p["t"])
        if v is None:
            continue
        speed = math.hypot(*v)
        if speed < MOVING_SPEED_MIN:
            n_paused += 1
            continue

        # Positive dot product with the robot->person bearing means the person
        # is moving away from the robot.
        radial = (v[0] * bx + v[1] * by) / bnorm if bnorm > 1e-6 else 0.0
        key = "AWAY from robot" if radial > 0 else "TOWARD robot"

        g = groups[key]
        g[0].append(ex)
        g[1].append(ey)
        g[2].append(los)
        g[3].append(d)

    print(f"\n=== {pred_path} ===")
    print(f"  robot assumed at ({robot_x}, {robot_y})")
    print(f"  {n_paused} samples excluded as paused\n")

    report("ALL samples", all_dx, all_dy, all_los, all_dist)
    print()
    for key in ("TOWARD robot", "AWAY from robot"):
        report(key, *groups[key])
        print()

    t_los = mean(groups["TOWARD robot"][2])
    a_los = mean(groups["AWAY from robot"][2])

    print("  interpretation:")
    if math.isnan(t_los) or math.isnan(a_los):
        print("    insufficient samples in one direction to compare")
    elif t_los * a_los > 0:
        print(f"    line-of-sight error has the SAME sign both directions")
        print(f"    ({t_los:+.3f} toward, {a_los:+.3f} away)")
        print(f"    -> consistent with a fixed range/calibration offset,")
        print(f"       NOT the surface-vs-origin hypothesis")
    else:
        print(f"    line-of-sight error FLIPS sign with direction")
        print(f"    ({t_los:+.3f} toward, {a_los:+.3f} away)")
        print(f"    -> the surface-vs-origin hypothesis does not hold either;")
        print(f"       a near-face offset would keep the same sign")
    print()


if __name__ == "__main__":
    main()
