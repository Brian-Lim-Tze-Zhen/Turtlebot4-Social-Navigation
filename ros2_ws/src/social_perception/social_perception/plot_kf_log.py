#!/usr/bin/env python3
"""
plot_kf_log.py

Reads the CSVs produced by kf_prediction_logger.py and plots, per track_id:
  1. Trajectory: filtered position, predicted (+horizon) position, and
     (if available) ground truth, overlaid on the same axes.
  2. Velocity magnitude over time.

Ground truth matching:
    /person_ground_truth has no track_id - it's just an ordered PoseArray,
    one pose per person in move_person_gazebo2.py's self.people list (index
    0 = person_1, index 1 = person_2, ...). There is no guaranteed mapping
    from a YOLO/ByteTrack track_id to a ground-truth person_index - YOLO
    assigns IDs based on detection order, not identity. This script plots
    ALL ground-truth person_index traces alongside each predicted track, so
    you can visually pick out which one corresponds to which - it does not
    attempt automatic ID association.

Usage:
    python3 plot_kf_log.py ~/kf_prediction_log.csv
    python3 plot_kf_log.py ~/kf_prediction_log.csv ~/ground_truth_log.csv
    (if the ground truth path is omitted, it defaults to
     ~/ground_truth_log.csv if that file exists)
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt


def load_ground_truth(path):
    if path is None or not os.path.exists(path):
        return None
    gt = pd.read_csv(path)
    if gt.empty:
        return None
    return gt


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_kf_log.py <prediction_csv> [ground_truth_csv]")
        sys.exit(1)

    pred_path = sys.argv[1]
    gt_path = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/ground_truth_log.csv")

    df = pd.read_csv(pred_path)
    if df.empty:
        print(f"'{pred_path}' has no data rows - nothing to plot.")
        sys.exit(1)

    t0 = df["sim_time"].min()
    df["t"] = df["sim_time"] - t0

    gt = load_ground_truth(gt_path)
    if gt is not None:
        gt["t"] = gt["sim_time"] - t0
        print(f"Loaded ground truth: {gt_path} ({gt['person_index'].nunique()} person(s))")
    else:
        print(f"No ground truth loaded (looked for '{gt_path}') - plotting predictions only.")

    # Gaps larger than this (seconds) are treated as separate detection
    # episodes for the SAME track_id (e.g. ByteTrack re-using an id number
    # after the person was gone for a while) and are NOT connected by a
    # line - only individual gaps this large are broken, real frame-to-frame
    # gaps stay connected normally.
    GAP_BREAK_SECONDS = 5.0

    for track_id, g in df.groupby("track_id"):
        g = g.sort_values("t").reset_index(drop=True)

        dt = g["t"].diff()
        big_gaps = dt[dt > GAP_BREAK_SECONDS]
        if not big_gaps.empty:
            print(
                f"Track {track_id}: found {len(big_gaps)} gap(s) > {GAP_BREAK_SECONDS}s "
                f"(largest: {big_gaps.max():.1f}s) - these will NOT be connected by a line, "
                f"to avoid drawing a fake straight-line trajectory across a real detection gap."
            )

        # Insert NaN rows at gap boundaries so matplotlib breaks the line
        # there instead of joining distant points.
        break_mask = dt > GAP_BREAK_SECONDS
        g_plot = g.copy()
        g_plot.loc[break_mask, ["x", "y", "pred_x", "pred_y", "vx", "vy"]] = float("nan")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Track {track_id}")

        # --- Trajectory ---
        axes[0].plot(g_plot["x"], g_plot["y"], "b.-", label="current (filtered)", markersize=4)
        axes[0].plot(g_plot["pred_x"], g_plot["pred_y"], "r.-", label="predicted (+horizon)", markersize=4)

        if gt is not None:
            for person_index, pg in gt.groupby("person_index"):
                pg = pg.sort_values("t")
                axes[0].plot(
                    pg["x"], pg["y"], "--",
                    label=f"ground truth (person {person_index})",
                    linewidth=0.5, alpha=0.3,
                )

        axes[0].set_xlabel("x [m]")
        axes[0].set_ylabel("y [m]")
        axes[0].set_title("Trajectory")
        axes[0].legend(fontsize=8)
        axes[0].axis("equal")

        # --- Velocity magnitude ---
        speed = (g_plot["vx"] ** 2 + g_plot["vy"] ** 2) ** 0.5
        axes[1].plot(g_plot["t"], speed, "g.-", markersize=4)
        axes[1].set_xlabel("sim time [s]")
        axes[1].set_ylabel("speed [m/s]")
        axes[1].set_title("Velocity magnitude")

        plt.tight_layout()
        out_name = f"track_{track_id}.png"
        plt.savefig(out_name)
        print(f"Saved {out_name}")


if __name__ == "__main__":
    main()