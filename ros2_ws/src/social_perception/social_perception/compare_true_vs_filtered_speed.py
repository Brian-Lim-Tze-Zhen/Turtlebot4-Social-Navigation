#!/usr/bin/env python3
"""
compare_true_vs_filtered_speed.py

Compares the TRUE ground-truth walking speed (computed by differentiating
/person_ground_truth positions) against the Kalman filter's EMA-smoothed
velocity estimate (logged from /predicted_person_positions), for a given
track_id and ground-truth person_index, over a chosen sim_time window.

WHY THIS COMPARISON MATTERS:
    human_kf_predictor.py's published vx,vy is NOT the person's raw
    instantaneous velocity. It is an EMA-smoothed estimate
    (smooth_alpha=0.12) of the Kalman filter's internal velocity state,
    deliberately damped to reduce prediction jitter (see the THESIS
    MODIFICATION comments in that file). It will systematically read
    LOWER than the true speed and will lag behind real direction changes.
    Reporting it as "the person's walking speed" would be incorrect -
    this script makes the distinction visible and quantifies the gap.

GROUND TRUTH SPEED CALCULATION:
    /person_ground_truth has no velocity field, only position
    (geometry_msgs/PoseArray, one pose per person, indexed by order in
    move_person_gazebo2.py's self.people list - index 0 = person_1,
    index 1 = person_2). True speed here is computed by finite-differencing
    consecutive (x, y) ground-truth samples: speed = dist(p2,p1) / dt.
    This is itself noisy at the sample-to-sample level (especially right
    at the 1.0s set_pose update boundaries in move_person_gazebo2.py,
    which moves the person in discrete steps rather than continuously) -
    a short rolling average is applied to make the comparison fair against
    the already-smoothed KF estimate, not to hide real variation.

TIME ALIGNMENT:
    The two logs are independent ROS topics with independent publish
    timing - there is no row-for-row correspondence. This script aligns
    them via merge_asof on sim_time (nearest match within a tolerance),
    not by assuming equal sampling rates.

USAGE:
    python3 compare_true_vs_filtered_speed.py \
        --pred ~/kf_prediction_log.csv \
        --gt ~/ground_truth_log.csv \
        --track-id 1 \
        --person-index 0 \
        --min-sim-time 430 \
        --max-sim-time 460 \
        --min-speed-filter 0.05

    (defaults: --pred ~/kf_prediction_log.csv, --gt ~/ground_truth_log.csv,
     --track-id 1, --person-index 0, --min-sim-time 0, --max-sim-time None
     [no upper bound], --min-speed-filter None [no filtering])

    --max-sim-time is the key option for isolating ONE clean walking leg:
    set it to just before a waypoint turnaround (visible as a sudden drop
    toward 0 m/s in a first pass plot) so the comparison doesn't mix a
    real deceleration-to-stop event into your steady-state bias numbers.

OUTPUT:
    speed_comparison_track_<id>.png  - two panels:
      1. True ground-truth speed vs. KF-smoothed speed over time
      2. Trajectory with both traces, for context

    Also prints summary statistics (mean/std of each, and mean signed/
    absolute difference) to the console.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_ground_truth_speed(gt_person, rolling_window=3):
    """
    gt_person: DataFrame for ONE person_index, columns sim_time, x, y,
               already sorted by sim_time.
    Returns the same DataFrame with added columns: dt, speed_raw, speed.
    speed_raw is the per-step finite-difference speed; speed is a short
    centered rolling average of speed_raw (smoothing only to make the
    comparison fair against the already-smoothed KF output - NOT to hide
    genuine step-to-step variation, which is itself meaningful given
    move_person_gazebo2.py's discrete 1.0s set_pose update steps).
    """
    g = gt_person.sort_values("sim_time").reset_index(drop=True).copy()

    dt = g["sim_time"].diff()
    dx = g["x"].diff()
    dy = g["y"].diff()
    dist = np.sqrt(dx**2 + dy**2)

    # Avoid divide-by-zero on duplicate/near-duplicate timestamps.
    speed_raw = np.where(dt > 1e-6, dist / dt, np.nan)

    g["dt"] = dt
    g["speed_raw"] = speed_raw
    g["speed"] = (
        pd.Series(speed_raw)
        .rolling(rolling_window, center=True, min_periods=1)
        .mean()
    )

    return g


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", default=os.path.expanduser("~/kf_prediction_log.csv"))
    parser.add_argument("--gt", default=os.path.expanduser("~/ground_truth_log.csv"))
    parser.add_argument("--track-id", type=int, default=1)
    parser.add_argument("--person-index", type=int, default=0)
    parser.add_argument("--min-sim-time", type=float, default=0.0,
                         help="Exclude rows before this sim_time (use to skip spurious early detections).")
    parser.add_argument("--max-sim-time", type=float, default=None,
                         help="Exclude rows at/after this sim_time (use to cut off a window before a "
                              "waypoint turnaround/direction reversal, so the comparison stays within "
                              "one clean constant-velocity leg). If omitted, no upper bound is applied.")
    parser.add_argument("--min-speed-filter", type=float, default=None,
                         help="Optional: exclude rows where ground-truth speed falls below this value "
                              "(m/s) when computing summary statistics - useful for dropping turnaround/"
                              "near-stationary samples that would otherwise bias the mean toward zero. "
                              "Rows are still shown in the plot; only the printed stats exclude them. "
                              "Off by default.")
    parser.add_argument("--merge-tolerance", type=float, default=0.5,
                         help="Max time difference (s) allowed when aligning prediction rows to ground-truth rows.")
    args = parser.parse_args()

    if not os.path.exists(args.pred):
        print(f"Prediction CSV not found: {args.pred}")
        sys.exit(1)
    if not os.path.exists(args.gt):
        print(f"Ground-truth CSV not found: {args.gt}")
        sys.exit(1)

    pred = pd.read_csv(args.pred)
    gt = pd.read_csv(args.gt)

    pred_mask = (pred["track_id"] == args.track_id) & (pred["sim_time"] >= args.min_sim_time)
    gt_mask = (gt["person_index"] == args.person_index) & (gt["sim_time"] >= args.min_sim_time)

    if args.max_sim_time is not None:
        pred_mask &= pred["sim_time"] < args.max_sim_time
        gt_mask &= gt["sim_time"] < args.max_sim_time

    pred = pred[pred_mask]
    gt_person = gt[gt_mask]

    window_desc = (
        f"sim_time in [{args.min_sim_time}, {args.max_sim_time})"
        if args.max_sim_time is not None
        else f"sim_time >= {args.min_sim_time}"
    )

    if pred.empty:
        print(f"No prediction rows for track_id={args.track_id} with {window_desc}.")
        sys.exit(1)
    if gt_person.empty:
        print(f"No ground-truth rows for person_index={args.person_index} with {window_desc}.")
        sys.exit(1)

    pred = pred.sort_values("sim_time").reset_index(drop=True)
    gt_speed = compute_ground_truth_speed(gt_person)

    pred["kf_speed"] = (pred["vx"] ** 2 + pred["vy"] ** 2) ** 0.5

    # Align: for each prediction row, find the nearest ground-truth speed
    # sample within merge_tolerance seconds. merge_asof requires sorted
    # keys on both sides, which we already have.
    merged = pd.merge_asof(
        pred[["sim_time", "x", "y", "kf_speed"]],
        gt_speed[["sim_time", "x", "y", "speed"]].rename(
            columns={"x": "gt_x", "y": "gt_y", "speed": "gt_speed"}
        ),
        on="sim_time",
        direction="nearest",
        tolerance=args.merge_tolerance,
    )

    merged = merged.dropna(subset=["gt_speed"])

    if merged.empty:
        print(
            "No overlapping timestamps within tolerance "
            f"({args.merge_tolerance}s) between prediction and ground-truth data. "
            "Check --track-id / --person-index / --min-sim-time, or increase --merge-tolerance."
        )
        sys.exit(1)

    t0 = merged["sim_time"].min()
    merged["t"] = merged["sim_time"] - t0

    # Rows used for the PLOT always include everything in the window
    # (so turnarounds/stops are visible in context). Rows used for the
    # PRINTED STATS optionally exclude low-speed samples if requested,
    # since a turnaround dragging the mean toward zero would otherwise
    # misrepresent "how well does the filter track a walking person"
    # with "how well does it track a person standing still for an
    # instant mid-turnaround" - two different, both-valid questions
    # that --min-speed-filter lets you separate.
    stats_df = merged
    excluded_note = ""
    if args.min_speed_filter is not None:
        before_n = len(merged)
        stats_df = merged[merged["gt_speed"] >= args.min_speed_filter]
        excluded_n = before_n - len(stats_df)
        excluded_note = (
            f" ({excluded_n} of {before_n} rows excluded from stats below "
            f"for gt_speed < {args.min_speed_filter} m/s; plot still shows all rows)"
        )
        if stats_df.empty:
            print(f"All rows excluded by --min-speed-filter={args.min_speed_filter} - nothing left to compute stats on.")
            sys.exit(1)

    diff = stats_df["kf_speed"] - stats_df["gt_speed"]

    print(f"Aligned samples: {len(merged)}{excluded_note}")
    print()
    print(f"True (ground truth) speed  - mean: {stats_df['gt_speed'].mean():.4f} m/s, std: {stats_df['gt_speed'].std():.4f}")
    print(f"KF-smoothed speed          - mean: {stats_df['kf_speed'].mean():.4f} m/s, std: {stats_df['kf_speed'].std():.4f}")
    print()
    print(f"Mean signed difference (KF - true): {diff.mean():.4f} m/s")
    print(f"Mean absolute difference          : {diff.abs().mean():.4f} m/s")
    print(f"As % of true mean speed           : {100 * diff.mean() / stats_df['gt_speed'].mean():.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Track {args.track_id} vs Ground Truth Person {args.person_index}: True vs KF-Smoothed Speed")

    axes[0].plot(merged["t"], merged["gt_speed"], "k.-", label="true (ground truth)", markersize=4, alpha=0.7)
    axes[0].plot(merged["t"], merged["kf_speed"], "g.-", label="KF-smoothed estimate", markersize=4, alpha=0.7)
    axes[0].set_xlabel("sim time [s]")
    axes[0].set_ylabel("speed [m/s]")
    axes[0].set_title("Speed over time")
    axes[0].legend()

    axes[1].plot(merged["gt_x"], merged["gt_y"], "k--", label="ground truth", linewidth=1.5, alpha=0.6)
    axes[1].plot(merged["x"], merged["y"], "b.-", label="KF filtered position", markersize=4)
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("y [m]")
    axes[1].set_title("Trajectory (context)")
    axes[1].axis("equal")
    axes[1].legend()

    plt.tight_layout()
    out_name = f"speed_comparison_track_{args.track_id}.png"
    plt.savefig(out_name)
    print(f"\nSaved {out_name}")


if __name__ == "__main__":
    main()