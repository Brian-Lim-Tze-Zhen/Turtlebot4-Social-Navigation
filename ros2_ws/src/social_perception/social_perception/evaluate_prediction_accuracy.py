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

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Track {args.track_id} vs Ground Truth Person {args.person_index}: "
                 f"Prediction Error ({merged['horizon'].iloc[0]:.1f}s horizon)")

    axes[0].plot(merged["t"], merged["pred_error"], "r.-", label="KF prediction error", markersize=4, alpha=0.8)
    axes[0].plot(merged["t"], merged["naive_error"], "k.--", label="naive (zero-velocity) error", markersize=3, alpha=0.5)
    axes[0].set_xlabel("sim time [s] (prediction made at this time)")
    axes[0].set_ylabel("position error [m]")
    axes[0].set_title("Prediction error over time")
    axes[0].legend()

    axes[1].plot(merged["x"], merged["y"], "b.-", label="current position", markersize=3, alpha=0.5)
    axes[1].plot(merged["pred_x"], merged["pred_y"], "r.", label="predicted position", markersize=4, alpha=0.6)
    axes[1].plot(merged["future_gt_x"], merged["future_gt_y"], "k.", label="actual future position", markersize=4, alpha=0.6)
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
