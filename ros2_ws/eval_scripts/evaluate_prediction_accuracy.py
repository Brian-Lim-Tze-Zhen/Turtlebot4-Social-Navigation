#!/usr/bin/env python3
"""
evaluate_prediction_accuracy.py

Evaluates the ACTUAL predictive output of human_kf_predictor.py:
pred_x, pred_y (the position extrapolated `horizon` seconds into the
future using the EMA-smoothed velocity), compared against where the
person genuinely was at that future time, per ground truth.

WHY THIS IS DIFFERENT FROM compare_true_vs_filtered_speed.py:
    That script compares vx,vy (the RAW, unsmoothed Kalman velocity
    state - kept only for logging/diagnostics) against true speed at
    the SAME timestamp. It is a velocity-bias diagnostic, not a
    measure of prediction quality, and vx,vy never reaches the robot's
    actual obstacle-avoidance logic.

    pred_x, pred_y is what predicted_person_cloud_node.py and
    prediction_marker_node.py actually consume to draw the robot's
    risk/obstacle zones. This script measures what matters for the
    robot's actual behavior: "if the filter says the person will be
    HERE in `horizon` seconds, how far off was it from where they
    actually ended up?"

METHOD:
    For each prediction row at sim_time=t with horizon=h, look up
    the ground-truth position at sim_time=t+h (nearest match within
    --merge-tolerance) and compute the Euclidean distance between
    (pred_x, pred_y) and that future ground-truth position. This is
    the prediction's positional error.

    Note this requires ground-truth data to extend at least `horizon`
    seconds PAST the end of your prediction window, or the last few
    seconds of predictions won't have a future ground-truth point to
    compare against and will be dropped (reported, not silently lost).

USAGE:
    python3 evaluate_prediction_accuracy.py \
        --pred ~/kf_prediction_log.csv \
        --gt ~/ground_truth_log.csv \
        --track-id 1 \
        --person-index 0 \
        --min-sim-time 39.711 \
        --max-sim-time 47.487

OUTPUT:
    prediction_error_track_<id>.png - two panels:
      1. Positional prediction error over time
      2. Trajectory: actual path, predicted points, and the true
         future position each prediction was aiming for

    Prints: mean/median/max error, and for comparison, the error a
    "naive" zero-velocity prediction (assume the person doesn't move
    at all) would have had over the same horizon - this baseline
    tells you whether the KF's prediction is actually adding value
    over just guessing "they'll stay where they are now".
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", default=os.path.expanduser("~/kf_prediction_log.csv"))
    parser.add_argument("--gt", default=os.path.expanduser("~/ground_truth_log.csv"))
    parser.add_argument("--track-id", type=int, default=1)
    parser.add_argument("--person-index", type=int, default=0)
    parser.add_argument("--min-sim-time", type=float, default=0.0)
    parser.add_argument("--max-sim-time", type=float, default=None,
                         help="Upper bound on the PREDICTION's sim_time (not the ground-truth "
                              "lookup time, which will extend horizon seconds further).")
    parser.add_argument("--merge-tolerance", type=float, default=0.3,
                         help="Max time difference (s) allowed when matching a prediction's "
                              "target time (sim_time + horizon) to an actual ground-truth sample.")
    parser.add_argument("--high-error-threshold", type=float, default=0.6,
                         help="KF prediction error threshold (m) used to count large-error spikes.")
    parser.add_argument("--direction-heading-threshold", type=float, default=45.0,
                         help="Heading change threshold (degrees) used to flag direction changes.")
    parser.add_argument("--direction-accel-threshold", type=float, default=0.5,
                         help="Ground-truth acceleration threshold (m/s^2) used to flag abrupt motion changes.")
    parser.add_argument("--min-moving-speed", type=float, default=0.05,
                         help="Minimum GT speed (m/s) required before heading is considered reliable.")
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
    if args.max_sim_time is not None:
        pred_mask &= pred["sim_time"] < args.max_sim_time
    pred = pred[pred_mask].sort_values("sim_time").reset_index(drop=True)

    gt_person = gt[gt["person_index"] == args.person_index].sort_values("sim_time").reset_index(drop=True)
    if pred.empty:
        print(f"No prediction rows for track_id={args.track_id} in the requested window.")
        sys.exit(1)
    if gt_person.empty:
        print(f"No ground-truth rows for person_index={args.person_index}.")
        sys.exit(1)

    # Estimate ground-truth velocity from GT positions
    gt_person["dt"] = gt_person["sim_time"].diff()

    # Avoid division by zero or invalid dt
    gt_person.loc[gt_person["dt"] <= 0, "dt"] = np.nan

    gt_person["vx_gt"] = gt_person["x"].diff() / gt_person["dt"]
    gt_person["vy_gt"] = gt_person["y"].diff() / gt_person["dt"]

    gt_person["speed_gt"] = np.sqrt(
        gt_person["vx_gt"]**2 + gt_person["vy_gt"]**2
    )

    # Acceleration magnitude, useful for detecting sudden motion changes
    gt_person["ax_gt"] = gt_person["vx_gt"].diff() / gt_person["dt"]
    gt_person["ay_gt"] = gt_person["vy_gt"].diff() / gt_person["dt"]

    gt_person["accel_gt"] = np.sqrt(
        gt_person["ax_gt"]**2 + gt_person["ay_gt"]**2
    )

    # Heading angle of motion
    gt_person["heading_gt"] = np.arctan2(
        gt_person["vy_gt"],
        gt_person["vx_gt"]
    )

    # The time each prediction is actually "aiming for" is sim_time + horizon,
    # not sim_time itself - that's the whole point of a predictive filter.
    pred["target_time"] = pred["sim_time"] + pred["horizon"]

    # merge_asof needs the key column sorted and matching names; do the
    # lookup against ground truth's sim_time using target_time as the key.
    merged = pd.merge_asof(
        pred[["sim_time", "horizon", "target_time", "x", "y", "pred_x", "pred_y"]],
        gt_person[["sim_time", "x", "y"]].rename(
            columns={"sim_time": "gt_sim_time", "x": "future_gt_x", "y": "future_gt_y"}
        ),
        left_on="target_time",
        right_on="gt_sim_time",
        direction="nearest",
        tolerance=args.merge_tolerance,
    )

    total_rows = len(merged)

    # Diagnose why predictions failed to match future ground truth
    gt_min_time = gt_person["sim_time"].min()
    gt_max_time = gt_person["sim_time"].max()

    unmatched = merged[merged["future_gt_x"].isna() | merged["future_gt_y"].isna()].copy()

    dataset_end = unmatched["target_time"] > gt_max_time
    dataset_before_start = unmatched["target_time"] < gt_min_time

    # If target_time is inside GT time range but merge_asof failed,
    # then nearest GT sample was farther than merge_tolerance.
    timestamp_mismatch = (
        ~dataset_end
        & ~dataset_before_start
    )

    print()
    print("Drop reason breakdown:")
    print(f"- Target time after GT ends: {dataset_end.sum()}")
    print(f"- Target time before GT starts: {dataset_before_start.sum()}")
    print(f"- No GT timestamp within tolerance: {timestamp_mismatch.sum()}")
    print(f"- GT time range: {gt_min_time:.3f} to {gt_max_time:.3f} s")
    print(f"- Prediction target time range: {pred['target_time'].min():.3f} to {pred['target_time'].max():.3f} s")
    print(f"- Merge tolerance: {args.merge_tolerance:.3f} s")
    print()

    merged = merged.dropna(subset=["future_gt_x", "future_gt_y"])
    dropped = total_rows - len(merged)

    if merged.empty:
        print(
            "No predictions could be matched to a future ground-truth sample. "
            "This usually means ground truth doesn't extend far enough past your "
            "--max-sim-time (it needs to cover up to roughly max_sim_time + horizon), "
            "or --merge-tolerance is too tight. Check your ground-truth log's sim_time range."
        )
        sys.exit(1)

    # Prediction error: how far was the predicted point from where the
    # person actually was at the time the prediction was "for".
    merged["pred_error"] = np.sqrt(
        (merged["pred_x"] - merged["future_gt_x"]) ** 2
        + (merged["pred_y"] - merged["future_gt_y"]) ** 2
    )

    # Naive baseline: "the person will still be where they are right now"
    # (zero-velocity assumption). If the KF's prediction isn't beating
    # this, the prediction step isn't adding value over doing nothing.
    merged["naive_error"] = np.sqrt(
        (merged["x"] - merged["future_gt_x"]) ** 2
        + (merged["y"] - merged["future_gt_y"]) ** 2
    )
    debug_t0 = 29.5
    debug_t1 = 30.8

    cols = [
        "sim_time", "target_time",
        "x", "y",
        "pred_x", "pred_y",
        "future_gt_x", "future_gt_y",
        "pred_error", "naive_error",
    ]

    for c in [
        "vx", "vy",
        "kf_vx", "kf_vy",
        "vel_x", "vel_y",
        "range_m",
        "depth",
        "valid_depth_pixels",
        "bbox_width",
        "bbox_height",
    ]:
        if c in merged.columns and c not in cols:
            cols.append(c)

    print("\nDebug window around 1m spike:")
    print(
        merged[
            (merged["sim_time"] >= debug_t0) &
            (merged["sim_time"] <= debug_t1)
        ][cols].to_string(index=False)
    )
    def interval_motion_change(row):
        """
        Diagnose whether the ground-truth person changed motion between
        the prediction time t and the target time t + horizon.

        This is not used to compute prediction error. It only explains
        whether large KF errors coincide with abrupt GT motion changes.
        """
        seg = gt_person[
            (gt_person["sim_time"] >= row["sim_time"])
            & (gt_person["sim_time"] <= row["target_time"])
        ].copy()

        if len(seg) < 3:
            return pd.Series({
                "gt_vx_min": np.nan,
                "gt_vx_max": np.nan,
                "gt_speed_max": np.nan,
                "gt_accel_max": np.nan,
                "gt_heading_change_deg": np.nan,
                "vx_sign_change": False,
                "direction_change": False,
            })

        moving = seg[seg["speed_gt"] > args.min_moving_speed].copy()

        if len(moving) >= 3:
            headings = np.unwrap(moving["heading_gt"].dropna().values)
            if len(headings) >= 2:
                heading_change_deg = float(np.degrees(headings.max() - headings.min()))
            else:
                heading_change_deg = 0.0
        else:
            heading_change_deg = 0.0

        vx_min = float(seg["vx_gt"].min())
        vx_max = float(seg["vx_gt"].max())
        speed_max = float(seg["speed_gt"].max())
        accel_max = float(seg["accel_gt"].max())

        # Since your shown experiments are mostly along x, vx sign reversal is
        # a useful simple indicator of forward/backward direction change.
        vx_sign_change = (vx_min < -args.min_moving_speed) and (vx_max > args.min_moving_speed)

        direction_change = bool(
            vx_sign_change
            or (heading_change_deg > args.direction_heading_threshold)
            or (accel_max > args.direction_accel_threshold)
        )

        return pd.Series({
            "gt_vx_min": vx_min,
            "gt_vx_max": vx_max,
            "gt_speed_max": speed_max,
            "gt_accel_max": accel_max,
            "gt_heading_change_deg": heading_change_deg,
            "vx_sign_change": vx_sign_change,
            "direction_change": direction_change,
        })

    motion_diag = merged.apply(interval_motion_change, axis=1)
    merged = pd.concat([merged, motion_diag], axis=1)

    merged["kf_better"] = merged["pred_error"] < merged["naive_error"]
    merged["high_kf_error"] = merged["pred_error"] > args.high_error_threshold

    t0 = merged["sim_time"].min()
    merged["t"] = merged["sim_time"] - t0

    print(f"Matched predictions: {len(merged)} (dropped {dropped} with no future ground-truth match)")
    print(f"Horizon used: {merged['horizon'].iloc[0]:.2f} s (from data)")
    print()
    print(f"KF prediction error    - mean: {merged['pred_error'].mean():.4f} m, "
          f"median: {merged['pred_error'].median():.4f} m, max: {merged['pred_error'].max():.4f} m")
    print(f"Naive (zero-vel) error  - mean: {merged['naive_error'].mean():.4f} m, "
          f"median: {merged['naive_error'].median():.4f} m, max: {merged['naive_error'].max():.4f} m")
    print()
    improvement = 100 * (merged["naive_error"].mean() - merged["pred_error"].mean()) / merged["naive_error"].mean()
    print(f"KF prediction improves on naive baseline by: {improvement:.1f}%")
    if improvement < 0:
        print("WARNING: KF prediction is WORSE than just assuming the person doesn't move.")

    print()
    print(f"KF better than naive in: {100 * merged['kf_better'].mean():.1f}% of matched predictions")
    print(f"KF worse/equal than naive in: {100 * (~merged['kf_better']).mean():.1f}% of matched predictions")

    high_count = int(merged["high_kf_error"].sum())
    high_with_direction = int((merged["high_kf_error"] & merged["direction_change"]).sum())

    print()
    print("Direction-change diagnostic:")
    print(f"High KF error threshold: {args.high_error_threshold:.2f} m")
    print(f"High-error predictions: {high_count} / {len(merged)}")
    if high_count > 0:
        print(f"High-error predictions with direction change: {high_with_direction} / {high_count} "
              f"({100 * high_with_direction / high_count:.1f}%)")
    else:
        print("High-error predictions with direction change: 0 / 0")

    print()
    print("Mean KF error grouped by direction_change:")
    grouped = merged.groupby("direction_change")["pred_error"].agg(["count", "mean", "median", "max"])
    print(grouped.to_string())

    print()
    print("Worst KF prediction errors:")
    worst_cols = [
        "sim_time", "target_time", "pred_error", "naive_error",
        "gt_vx_min", "gt_vx_max", "gt_accel_max",
        "gt_heading_change_deg", "vx_sign_change", "direction_change"
    ]
    print(
        merged.sort_values("pred_error", ascending=False)
        [worst_cols]
        .head(10)
        .to_string(index=False)
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Track {args.track_id} vs Ground Truth Person {args.person_index}: "
                 f"Prediction Error ({merged['horizon'].iloc[0]:.1f}s horizon)")

    # Consistent colour scheme (matches plot_kf_log.py and compare_true_vs_filtered_speed.py):
    #   KF error / KF estimates -> Green  #2E7D32
    #   Naive baseline          -> Grey   #9E9E9E
    #   Current filtered pos    -> Blue   #2196F3
    #   Predicted position      -> Orange #FF6F00
    #   Ground truth            -> Black  #000000
    axes[0].plot(merged["t"], merged["pred_error"],
                 color="#2E7D32", marker=".", linestyle="-",
                 label="KF prediction error", markersize=4, alpha=0.8)
    axes[0].plot(merged["t"], merged["naive_error"],
                 color="#9E9E9E", marker=".", linestyle="--",
                 label="naive (zero-velocity) error", markersize=3, alpha=0.7)
    axes[0].set_xlabel("elapsed time since first matched prediction [s]")
    axes[0].set_ylabel("position error [m]")
    axes[0].set_title("Prediction error over time")
    axes[0].legend()

    axes[1].plot(merged["x"], merged["y"],
                 color="#2196F3", marker=".", linestyle="-",
                 label="current position", markersize=3, alpha=0.5)

    # Subsample predicted position to match GT density (~2 Hz vs ~15 Hz).
    # Every 8th point gives roughly one orange dot per GT black dot so
    # neither overwhelms the other visually.
    pred_sub = merged.iloc[::8]
    axes[1].plot(pred_sub["pred_x"], pred_sub["pred_y"],
                 color="#FF6F00", marker=".", linestyle="none",
                 label="predicted position", markersize=5, alpha=0.8)
    axes[1].plot(merged["future_gt_x"], merged["future_gt_y"],
                 color="#000000", marker=".", linestyle="none",
                 label="actual future position", markersize=4, alpha=0.6)
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("y [m]")
    axes[1].set_title("Predicted vs actual future position")
    axes[1].axis("equal")
    axes[1].legend()

    plt.tight_layout()
    out_name = f"prediction_error_track_{args.track_id}.png"
    plt.savefig(out_name)
    print(f"\nSaved {out_name}")


if __name__ == "__main__":
    main()