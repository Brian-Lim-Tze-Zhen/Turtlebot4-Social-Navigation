#!/usr/bin/env python3
"""
plot_trajectory_offset_figure.py

Generates a publication-quality trajectory figure comparing the
Kalman-filtered position against ground truth, with the mean
positional offset annotated directly on the plot and reported in
the console. Intended as a thesis figure (Figure: "Filtered vs.
ground-truth trajectory").

USAGE:
    python3 plot_trajectory_offset_figure.py \
        --pred ~/kf_prediction_log.csv \
        --gt ~/ground_truth_log.csv \
        --track-id 1 \
        --person-index 0 \
        --min-sim-time <start> \
        --max-sim-time <end>

    (omit --min-sim-time/--max-sim-time to use the whole file)

OUTPUT:
    trajectory_offset_figure.png - a single, clean trajectory panel:
      - ground truth path (dashed black)
      - KF filtered path (solid blue)
      - an arrow/annotation showing the mean offset vector and its
        magnitude in cm
      - axis labels, title, and a caption-ready text box with the
        numeric offset for direct use in a thesis figure caption
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
    parser.add_argument("--max-sim-time", type=float, default=None)
    parser.add_argument("--merge-tolerance", type=float, default=0.5)
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

    pred = pred[pred_mask].sort_values("sim_time").reset_index(drop=True)
    gt_person = gt[gt_mask].sort_values("sim_time").reset_index(drop=True)

    if pred.empty:
        print(f"No prediction rows for track_id={args.track_id} in the requested window.")
        sys.exit(1)
    if gt_person.empty:
        print(f"No ground-truth rows for person_index={args.person_index} in the requested window.")
        sys.exit(1)

    # Align on nearest timestamp to compute a per-sample offset, then
    # average - this is the same merge approach used elsewhere in this
    # analysis (compare_true_vs_filtered_speed.py), kept consistent.
    merged = pd.merge_asof(
        pred[["sim_time", "x", "y"]],
        gt_person[["sim_time", "x", "y"]].rename(columns={"x": "gt_x", "y": "gt_y"}),
        on="sim_time",
        direction="nearest",
        tolerance=args.merge_tolerance,
    ).dropna(subset=["gt_x", "gt_y"])

    if merged.empty:
        print("No overlapping timestamps between prediction and ground truth within tolerance. "
              "Check track-id/person-index/time window, or increase --merge-tolerance.")
        sys.exit(1)

    dx = (merged["x"] - merged["gt_x"]).mean()
    dy = (merged["y"] - merged["gt_y"]).mean()
    offset_mag = np.sqrt(dx**2 + dy**2)

    print(f"Aligned samples: {len(merged)}")
    print(f"Mean offset: dx={dx:+.4f} m, dy={dy:+.4f} m")
    print(f"Mean offset magnitude: {offset_mag:.4f} m ({offset_mag*100:.1f} cm)")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(gt_person["x"], gt_person["y"], "k--", linewidth=1.8, label="Ground truth", zorder=2)
    ax.plot(pred["x"], pred["y"], "-", color="#1f77b4", linewidth=2.0, label="KF filtered position", zorder=3)

    # Annotate the mean offset as an arrow from a representative ground-truth
    # point to the corresponding mean-shifted KF point, placed near the
    # middle of the path so it doesn't collide with the trajectory ends.
    mid_idx = len(merged) // 2
    gx0, gy0 = merged["gt_x"].iloc[mid_idx], merged["gt_y"].iloc[mid_idx]
    fx0, fy0 = gx0 + dx, gy0 + dy

    ax.annotate(
        "",
        xy=(fx0, fy0), xycoords="data",
        xytext=(gx0, gy0), textcoords="data",
        arrowprops=dict(arrowstyle="->", color="crimson", lw=2),
        zorder=4,
    )
    ax.plot([gx0], [gy0], "o", color="black", markersize=5, zorder=5)

    ax.text(
        0.02, 0.06,
        f"Mean offset: {offset_mag*100:.1f} cm\n(dx={dx*100:+.1f} cm, dy={dy*100:+.1f} cm)",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
        verticalalignment="bottom",
    )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Filtered vs. Ground-Truth Trajectory")
    ax.legend(loc="upper right")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_name = "trajectory_offset_figure.png"
    plt.savefig(out_name, dpi=200)
    print(f"\nSaved {out_name}")
    print(f"\nSuggested caption: \"Filtered vs. ground-truth trajectory; consistent "
          f"{offset_mag*100:.1f} cm positional offset attributed to depth-sampling "
          f"location on the body.\"")


if __name__ == "__main__":
    main()
