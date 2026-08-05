#!/usr/bin/env python3
"""
Pipeline lag measurement.

The KF's current-position estimate trails the person's true position. This
script finds the time shift that best explains that trailing, by asking: if we
compare the estimate at time t against where the person ACTUALLY WAS at time
t - lag, which value of lag minimises the error?

WHY MEASURE RATHER THAN DIVIDE
------------------------------
Dividing an observed offset by speed assumes the lag is purely temporal. If part
of the offset is a fixed range bias instead, a velocity-based correction will
overshoot at some speeds and undershoot at others. Sweeping the lag and looking
at the shape of the curve distinguishes the two:

  - A sharp minimum near a single lag value, with low residual error, means the
    offset really is a time delay and a v * t_lag correction is appropriate.
  - A shallow curve, or a minimum with large residual error still remaining,
    means something other than delay is contributing and the correction will
    only partly help.

The residual at the optimum is reported for exactly this reason: it is the part
of the error that lag compensation CANNOT fix.

Direction split is included because a genuine time delay produces the same lag
in both travel directions, whereas a fixed geometric offset does not.

Usage:
    python3 measure_pipeline_lag.py PRED_CSV GT_CSV [ROBOT_X ROBOT_Y]

Columns expected:
    prediction   : sim_time,track_id,conf,x,y,vx,vy,pred_x,pred_y,horizon
    ground truth : sim_time,person_index,x,y,z
"""

import csv
import math
import sys


LAG_MIN = 0.0    # s
LAG_MAX = 1.2    # s
LAG_STEP = 0.05  # s

MOVING_SPEED_MIN = 0.15   # m/s; below this the person is paused


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
    a = interp_gt(gt, t - window / 2.0)
    b = interp_gt(gt, t + window / 2.0)
    if a is None or b is None:
        return None
    return (b[0] - a[0]) / window, (b[1] - a[1]) / window


def mean_error_at_lag(preds, gt, lag, direction=None, robot=(-3.0, 0.0)):
    """Mean distance between the estimate at t and ground truth at t - lag.

    direction: None for all samples, 'toward' or 'away' to filter by whether
    the person is closing on the robot."""
    errs = []
    for p in preds:
        actual = interp_gt(gt, p["t"] - lag)
        if actual is None:
            continue

        if direction is not None:
            v = gt_velocity(gt, p["t"] - lag)
            if v is None:
                continue
            if math.hypot(*v) < MOVING_SPEED_MIN:
                continue
            bx = actual[0] - robot[0]
            by = actual[1] - robot[1]
            bn = math.hypot(bx, by)
            if bn < 1e-6:
                continue
            radial = (v[0] * bx + v[1] * by) / bn
            is_away = radial > 0
            if direction == "away" and not is_away:
                continue
            if direction == "toward" and is_away:
                continue

        errs.append(math.hypot(p["x"] - actual[0], p["y"] - actual[1]))

    if not errs:
        return None, 0
    return sum(errs) / len(errs), len(errs)


def sweep(preds, gt, direction, robot):
    results = []
    lag = LAG_MIN
    while lag <= LAG_MAX + 1e-9:
        err, n = mean_error_at_lag(preds, gt, lag, direction, robot)
        if err is not None:
            results.append((lag, err, n))
        lag += LAG_STEP
    return results


def report(label, results):
    if not results:
        print(f"  {label}: no samples")
        return None

    best_lag, best_err, best_n = min(results, key=lambda r: r[1])
    zero_err = results[0][1]

    print(f"  {label}")
    print(f"    best lag        : {best_lag:.2f} s")
    print(f"    error at best   : {best_err:.3f} m   (n={best_n})")
    print(f"    error at lag=0  : {zero_err:.3f} m")
    print(f"    improvement     : {(1 - best_err / zero_err) * 100:+.1f}%")
    print(f"    curve:")

    for lag, err, _ in results:
        marker = "  <-- best" if abs(lag - best_lag) < 1e-9 else ""
        bar = "#" * int(err / max(r[1] for r in results) * 40)
        print(f"      {lag:.2f}s  {err:6.3f} m  {bar}{marker}")

    return best_lag, best_err


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__)
        sys.exit(1)

    pred_path, gt_path = args[0], args[1]
    robot = (
        float(args[2]) if len(args) > 2 else -3.0,
        float(args[3]) if len(args) > 3 else 0.0,
    )

    preds = load_predictions(pred_path)
    gt = load_ground_truth(gt_path)

    if not preds or not gt:
        print("  missing prediction or ground-truth rows")
        sys.exit(1)

    print(f"\n=== {pred_path} ===\n")

    overall = report("ALL samples", sweep(preds, gt, None, robot))
    print()
    toward = report("TOWARD robot", sweep(preds, gt, "toward", robot))
    print()
    away = report("AWAY from robot", sweep(preds, gt, "away", robot))
    print()

    print("  interpretation:")
    if toward and away:
        dl = abs(toward[0] - away[0])
        print(f"    lag toward = {toward[0]:.2f} s, lag away = {away[0]:.2f} s "
              f"(difference {dl:.2f} s)")
        if dl <= 2 * LAG_STEP:
            print("    -> consistent across directions: this is a genuine time")
            print("       delay, and a v * t_lag correction is appropriate")
        else:
            print("    -> differs by direction: not a pure time delay, so a")
            print("       v * t_lag correction will only partly help")
    if overall:
        print(f"    residual error at best lag: {overall[1]:.3f} m")
        print("       this is the part lag compensation CANNOT remove")
    print()


if __name__ == "__main__":
    main()
