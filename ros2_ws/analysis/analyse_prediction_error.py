#!/usr/bin/env python3
"""
Prediction horizon error analysis, with reversal/pause exclusion.

For each prediction row (issued at sim_time t, predicting position at t+horizon),
looks up where the person ACTUALLY was at t+horizon by linear interpolation of
the ground-truth track, and reports the error distribution.

WHY EXCLUDE REVERSALS
---------------------
move_person_gazebo.py walks the person back and forth between endpoints with a
pause at each turn. The real head-on scenario is a SINGLE PASS: the person walks
toward the robot once, and the encounter is over before any reversal. A long
prediction issued mid-leg in the test pattern extrapolates straight through an
endpoint that would not exist in a real trial, so it is scored against a
direction change the scenario never contains. That penalises long horizons for
an artifact of the test rig rather than a property of the horizon.

This script therefore scores a prediction only if the person is moving
consistently in one direction across the whole interval [t, t+horizon]. Any
prediction whose window contains a direction change, a pause, or an endpoint is
excluded and reported separately.

Both filtered and unfiltered results are printed so the effect of the exclusion
is visible rather than hidden.

Usage:
    python3 analyse_prediction_error.py /root/h1_pred.csv /root/h1_gt.csv \
                                        /root/h2_pred.csv /root/h2_gt.csv \
                                        /root/h3_pred.csv /root/h3_gt.csv

Columns expected:
    prediction   : sim_time,track_id,conf,x,y,vx,vy,pred_x,pred_y,horizon
    ground truth : sim_time,person_index,x,y,z
"""

import csv
import math
import sys


# A ground-truth sample is treated as "moving" when its local speed exceeds
# this. The mover pauses fully at endpoints, so anything near zero is a pause,
# not slow walking.
MOVING_SPEED_MIN = 0.15   # m/s

# Direction is considered consistent when heading across the window varies by
# less than this. A reversal is ~180 deg, so this is a generous bound that still
# catches genuine turns.
MAX_HEADING_CHANGE = 0.52   # rad (~30 deg)


def load_predictions(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "t": float(r["sim_time"]),
                    "x": float(r["x"]),
                    "y": float(r["y"]),
                    "pred_x": float(r["pred_x"]),
                    "pred_y": float(r["pred_y"]),
                    "horizon": float(r["horizon"]),
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
    """Linear interpolation of the ground-truth track at time t.
    Returns None if t is outside the recorded span."""
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


def gt_segments_in(gt, t_start, t_end):
    """Consecutive ground-truth sample pairs overlapping [t_start, t_end]."""
    out = []
    for i in range(1, len(gt)):
        t0 = gt[i - 1][0]
        t1 = gt[i][0]
        if t1 < t_start or t0 > t_end:
            continue
        out.append((gt[i - 1], gt[i]))
    return out


def window_is_single_pass(gt, t_start, t_end):
    """True when the person moves continuously in one direction across the
    whole window: no pause, no endpoint, no reversal."""
    segs = gt_segments_in(gt, t_start, t_end)
    if not segs:
        return False

    headings = []
    for (t0, x0, y0), (t1, x1, y1) in segs:
        dt = t1 - t0
        if dt <= 0:
            continue
        dx = x1 - x0
        dy = y1 - y0
        speed = math.hypot(dx, dy) / dt

        # A pause or endpoint hold anywhere in the window disqualifies it.
        if speed < MOVING_SPEED_MIN:
            return False

        headings.append(math.atan2(dy, dx))

    if len(headings) < 2:
        return False

    h0 = headings[0]
    for h in headings[1:]:
        diff = math.atan2(math.sin(h - h0), math.cos(h - h0))
        if abs(diff) > MAX_HEADING_CHANGE:
            return False

    return True


def percentile(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarise(label, errors, naive_errors, travel, n_total, n_excluded):
    if not errors:
        print(f"  {label}\n    no scorable predictions")
        return

    n = len(errors)
    mean_err = sum(errors) / n
    mean_naive = sum(naive_errors) / n
    mean_travel = sum(travel) / len(travel) if travel else float("nan")
    improvement = (1.0 - mean_err / mean_naive) * 100.0 if mean_naive > 0 else float("nan")

    print(f"  {label}")
    print(f"    scored          : {n} / {n_total}   ({n_excluded} excluded)")
    print(f"    mean travel     : {mean_travel:.3f} m")
    print(f"    KF error   mean : {mean_err:.3f} m")
    print(f"               median: {percentile(errors, 50):.3f} m")
    print(f"               p95   : {percentile(errors, 95):.3f} m")
    print(f"    naive error mean: {mean_naive:.3f} m")
    print(f"    improvement     : {improvement:+.1f}%")


def analyse(pred_path, gt_path):
    preds = load_predictions(pred_path)
    gt = load_ground_truth(gt_path)

    if not preds:
        print(f"  no prediction rows in {pred_path}")
        return
    if not gt:
        print(f"  no ground-truth rows in {gt_path}")
        return

    horizon = preds[0]["horizon"]
    print(f"  horizon : {horizon:.1f} s\n")

    all_e, all_n, all_tr = [], [], []
    sp_e, sp_n, sp_tr = [], [], []

    n_unscorable = 0
    n_multi_pass = 0

    for p in preds:
        target_t = p["t"] + p["horizon"]

        actual = interp_gt(gt, target_t)
        if actual is None:
            n_unscorable += 1
            continue

        ax, ay = actual
        err = math.hypot(p["pred_x"] - ax, p["pred_y"] - ay)
        naive = math.hypot(p["x"] - ax, p["y"] - ay)

        now = interp_gt(gt, p["t"])
        tr = math.hypot(ax - now[0], ay - now[1]) if now else None

        all_e.append(err)
        all_n.append(naive)
        if tr is not None:
            all_tr.append(tr)

        if window_is_single_pass(gt, p["t"], target_t):
            sp_e.append(err)
            sp_n.append(naive)
            if tr is not None:
                sp_tr.append(tr)
        else:
            n_multi_pass += 1

    total = len(preds)

    summarise("ALL windows (includes reversals/pauses):",
              all_e, all_n, all_tr, total, n_unscorable)
    print()
    summarise("SINGLE-PASS windows only (matches real scenario):",
              sp_e, sp_n, sp_tr, total, n_unscorable + n_multi_pass)


def main():
    args = sys.argv[1:]

    if len(args) < 2 or len(args) % 2 != 0:
        print(__doc__)
        sys.exit(1)

    for i in range(0, len(args), 2):
        pred_path, gt_path = args[i], args[i + 1]
        print(f"\n=== {pred_path} ===")
        try:
            analyse(pred_path, gt_path)
        except FileNotFoundError as e:
            print(f"  {e}")

    print()


if __name__ == "__main__":
    main()
