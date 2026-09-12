#!/usr/bin/env python3
"""
inspect_run.py

THESIS DIAGNOSTIC — time-series view of a single bag, for the crossing
scenario.

analyse_avoidance.py reports single-value summary metrics per trial.
That can't distinguish "the robot slowed down" from "the robot swerved"
for the same min_distance, and can't show whether the global planner
kept re-routing during an encounter. This script plots the time series
behind those summaries instead.

Frame handling (lifting /odom into map via map_T_odom from /tf) is
imported directly from analyse_avoidance.py rather than re-derived, so
numbers from the two scripts are comparable. Only the bag-reading
function is extended locally (to also pull /plan), since
analyse_avoidance.read_bag() doesn't need that topic for its own
metrics.

Usage:
    python3 inspect_run.py bags/crossing_full_probe
"""

import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError as e:
    sys.exit(
        f"Missing ROS 2 Python packages ({e}).\n"
        "Run this inside the container with /opt/ros/jazzy sourced."
    )

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import analyse_avoidance as aa

# Matches COMMIT_DEV_THRESH in analyse_avoidance.compute_metrics(). Kept as
# a separate constant here (not imported) since that one is a local, not a
# module-level name. Used as the cross-check threshold in the stdout report.
COMMIT_DEV_THRESH = 0.20   # m

# Separation threshold defining the "encounter window" for the min-speed report.
ENCOUNTER_DIST = 3.0       # m


def read_bag_with_plan(path):
    """Same topic set as analyse_avoidance.read_bag(), plus /plan.
    Kept separate (rather than modifying analyse_avoidance.py) since that
    script's read_bag() intentionally only reads the topics its own
    metrics need."""
    wanted = {
        "/person_ground_truth",
        "/odom",
        "/tf",
        "/tf_static",
        "/amcl_pose",
        "/plan",
    }

    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=path, storage_id="mcap"),
        ConverterOptions("", ""),
    )

    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    msg_classes = {}
    out = {t: [] for t in wanted}

    while reader.has_next():
        topic, data, stamp = reader.read_next()
        if topic not in wanted:
            continue
        if topic not in msg_classes:
            msg_classes[topic] = get_message(type_map[topic])
        out[topic].append((stamp, deserialize_message(data, msg_classes[topic])))

    return out


def plan_length(msg):
    pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
    length = 0.0
    for i in range(len(pts) - 1):
        length += math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
    return length


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("bag", help="bag directory path")
    args = ap.parse_args()

    bag_path = args.bag.rstrip("/")
    name = os.path.basename(bag_path)

    try:
        data = read_bag_with_plan(bag_path)
    except Exception as e:
        sys.exit(f"[{name}] FAILED to read bag: {e}")

    tf_all = data["/tf"] + data["/tf_static"]
    m2o = aa.extract_map_to_odom(tf_all)
    odom_list = data["/odom"]
    if not odom_list:
        sys.exit(f"[{name}] no /odom messages — nothing to plot")

    robot = aa.robot_path_in_map(odom_list, m2o)
    person_tracks = aa.person_tracks_all(data["/person_ground_truth"])

    if len(robot) < 2:
        sys.exit(f"[{name}] insufficient /odom samples — nothing to plot")

    if not m2o:
        print(f"[{name}] WARNING: no map->odom in /tf; "
              f"treating odom as map (check localisation was running)")

    t0 = robot[0][0]
    x0, y0 = robot[0][1], robot[0][2]
    x1, y1 = robot[-1][1], robot[-1][2]
    straight = math.hypot(x1 - x0, y1 - y0)
    ux, uy = ((x1 - x0) / straight, (y1 - y0) / straight) if straight > 1e-6 else (0.0, 0.0)

    times = []
    speeds = []
    devs = []
    # THESIS MODIFICATION: one separation series per tracked person,
    # not just person 0 - a multi-person bag used to silently plot only
    # the first person, hiding the actual closest approach if it
    # happened to be to someone else (see FIX 1 context: min_dist=0.387
    # for bags/multi_full_probe was to person 1, invisible when only
    # person 0's separation was ever computed here).
    seps_per_person = {idx: [] for idx in person_tracks}

    for (t, px, py), (_, om) in zip(robot, odom_list):
        times.append(t - t0)
        speeds.append(om.twist.twist.linear.x)

        for idx, track in person_tracks.items():
            q = aa.interp_person(track, t) if track else None
            seps_per_person[idx].append(
                math.hypot(px - q[0], py - q[1]) if q is not None else None
            )

        rx, ry = px - x0, py - y0
        devs.append(-rx * uy + ry * ux)   # signed; + = left

    # Whichever person sets the overall minimum separation - matches
    # analyse_avoidance.compute_metrics()'s closest_idx, used below so
    # the commit_dist cross-check compares against the same person.
    min_sep_per_person = {}
    for idx, sv in seps_per_person.items():
        valid = [s for s in sv if s is not None]
        min_sep_per_person[idx] = min(valid) if valid else None

    closest_idx = min(
        (idx for idx in min_sep_per_person if min_sep_per_person[idx] is not None),
        key=lambda idx: min_sep_per_person[idx],
        default=None,
    )

    # --- /plan series ---
    # NOTE: use msg.header.stamp here, not the bag-recording stamp the
    # tuple's first element holds - the latter is wall-clock nanoseconds
    # since epoch and not comparable to the sim-time basis t0 is in.
    plan_msgs = data["/plan"]
    plan_times = []
    plan_lengths = []
    for _, msg in plan_msgs:
        if not msg.poses:
            continue
        t = aa.stamp_to_sec(msg.header.stamp)
        plan_times.append(t - t0)
        plan_lengths.append(plan_length(msg))

    # =================================================================
    # stdout summary
    # =================================================================
    print(f"[{name}]")

    # THESIS MODIFICATION: per-person, not just person 0 - reporting only
    # person 0's encounter used to print a misleading number whenever
    # person 0's own closest approach happened before the robot had even
    # started moving.
    for idx in sorted(seps_per_person):
        sv = seps_per_person[idx]
        encounter = [(t, s, v) for t, s, v in zip(times, sv, speeds)
                     if s is not None and s < ENCOUNTER_DIST]
        if encounter:
            t_min, s_min, v_min = min(encounter, key=lambda r: r[2])
            print(f"  person {idx}: min speed during encounter "
                  f"(sep<{ENCOUNTER_DIST:.1f}m): {v_min:.3f} m/s at "
                  f"t={t_min:.2f}s, separation={s_min:.3f} m")
        else:
            print(f"  person {idx}: no samples with separation < "
                  f"{ENCOUNTER_DIST:.1f} m — no encounter window found")

    print(f"  /plan messages: {len(plan_msgs)}")
    if len(plan_lengths) >= 2:
        mean_len = sum(plan_lengths) / len(plan_lengths)
        var = sum((v - mean_len) ** 2 for v in plan_lengths) / (len(plan_lengths) - 1)
        print(f"  plan length: mean={mean_len:.3f} m, std={var ** 0.5:.3f} m "
              f"(n={len(plan_lengths)})")
    elif len(plan_lengths) == 1:
        print(f"  plan length std dev: n/a (only 1 valid /plan message, "
              f"length={plan_lengths[0]:.3f} m)")
    else:
        print("  plan length std dev: n/a (no /plan messages with poses)")

    # Cross-check against whichever person sets the overall min
    # separation, matching analyse_avoidance.compute_metrics()'s
    # commit_dist definition (closest_idx there, same here).
    commit_seps = seps_per_person.get(closest_idx) if closest_idx is not None else None

    commit_t = None
    commit_sep = None
    for i, (t, dev) in enumerate(zip(times, devs)):
        if abs(dev) > COMMIT_DEV_THRESH:
            commit_t = t
            if commit_seps is not None:
                commit_sep = commit_seps[i]
            break
    if commit_t is not None:
        sep_str = f"{commit_sep:.3f} m" if commit_sep is not None else "n/a"
        who = f", person {closest_idx}" if len(person_tracks) > 1 and closest_idx is not None else ""
        print(f"  |lateral dev| first exceeds {COMMIT_DEV_THRESH:.2f} m at "
              f"t={commit_t:.2f}s, separation={sep_str}{who} "
              f"(cross-check vs analyse_avoidance.py's commit_dist)")
    else:
        print(f"  lateral deviation never exceeds {COMMIT_DEV_THRESH:.2f} m")

    # =================================================================
    # plot
    # =================================================================
    out_dir = "eval_plots"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}_inspect.png")

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 12))

    axes[0].plot(times, speeds)
    axes[0].axhline(0, color="0.7", linewidth=0.8)
    axes[0].set_ylabel("Speed (m/s)")
    axes[0].set_title("Robot forward speed")

    for idx in sorted(seps_per_person):
        sv = seps_per_person[idx]
        sep_t = [t for t, s in zip(times, sv) if s is not None]
        sep_v = [s for s in sv if s is not None]
        axes[1].plot(sep_t, sep_v, label=f"person {idx}")
    axes[1].axhline(ENCOUNTER_DIST, color="orange", linestyle="--",
                    linewidth=1, label=f"encounter window ({ENCOUNTER_DIST:.1f} m)")
    axes[1].set_ylabel("Separation (m)")
    axes[1].set_title("Robot-person separation")
    axes[1].legend(loc="upper right", fontsize=8)

    axes[2].plot(times, devs)
    axes[2].axhline(COMMIT_DEV_THRESH, color="red", linestyle="--", linewidth=1)
    axes[2].axhline(-COMMIT_DEV_THRESH, color="red", linestyle="--", linewidth=1)
    axes[2].axhline(0, color="0.7", linewidth=0.8)
    axes[2].set_ylabel("Lateral dev (m)\n(+ = left)")
    axes[2].set_title("Signed lateral deviation from start->goal line")

    if plan_lengths:
        axes[3].plot(plan_times, plan_lengths, marker=".")
    else:
        axes[3].text(0.5, 0.5, "no /plan messages recorded",
                     ha="center", va="center", transform=axes[3].transAxes)
    axes[3].set_ylabel("Plan length (m)")
    axes[3].set_title("Global plan length (re-routing indicator)")
    axes[3].set_xlabel("Time (s)")

    fig.suptitle(name)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
