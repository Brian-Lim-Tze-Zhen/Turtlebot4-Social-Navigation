#!/usr/bin/env python3
"""
analyse_avoidance.py

THESIS EVALUATION — human motion avoidance metrics from recorded bags.

Computes per-trial metrics from bags recorded with:

    ros2 bag record --topics /person_ground_truth /odom /amcl_pose \
        /tf /tf_static /plan /predicted_person_positions \
        -o bags/<scenario>_<config>_t<NN>

-----------------------------------------------------------------------
METRICS AND THEIR RTF SENSITIVITY
-----------------------------------------------------------------------
The evaluation platform shows unstable real-time factor when Gazebo's
GUI client is running (measured: cumulative RTF ~0.34, per-sample swing
0.025-0.98). This splits the metrics into two classes:

  RTF-SAFE (geometric — computed from positions only):
    - min_distance        : closest robot-person approach (m)
    - path_length         : distance travelled by robot (m)
    - straight_line_dist  : start-to-end straight line (m)
    - path_ratio          : path_length / straight_line_dist
    - max_lateral_dev     : max perpendicular deviation from the
                            start->end straight line (m)

  RTF-SENSITIVE (wall-clock dependent — reported but NOT comparable
  across runs recorded at different RTF):
    - duration_s          : bag duration in recorded (sim) seconds
    - mean_speed          : path_length / duration_s

Only quote the RTF-safe metrics in the ablation table unless every
trial was recorded at a verified stable RTF.

-----------------------------------------------------------------------
FRAME HANDLING
-----------------------------------------------------------------------
/person_ground_truth is published in the map frame by
move_person_gazebo.py. /odom is in the odom frame. To compare them the
robot pose must be lifted into map:

    map_pose = map_T_odom  o  odom_pose

map_T_odom is taken from /tf (published by AMCL). Everything here is
planar, so the composition is done directly in 2D rather than through
tf2 — fewer dependencies, and the arithmetic is easy to verify:

    map_x = mo_x + cos(mo_yaw)*ox - sin(mo_yaw)*oy
    map_y = mo_y + sin(mo_yaw)*ox + cos(mo_yaw)*oy

For each odom sample the most recent map->odom at or before its
timestamp is used (zero-order hold). AMCL corrections are infrequent
and small, so this is accurate to well under the ground-truth sampling
resolution.

-----------------------------------------------------------------------
GROUND-TRUTH INTERPOLATION
-----------------------------------------------------------------------
/person_ground_truth is published at the mover's update_dt (0.5 s), so
a 112 s trial yields only ~75 samples while /odom yields ~2350. Taking
min-distance only at ground-truth sample times would quantise the
result to roughly the distance the person moves between samples.

Since the person walks in a straight line at constant speed between
waypoints, linear interpolation between consecutive ground-truth
samples is exact apart from the brief endpoint pauses. Interpolating
the person's position onto the odom timeline therefore recovers full
resolution without inventing motion that did not occur.

Usage:
    python3 analyse_avoidance.py bags/headon_full_t01
    python3 analyse_avoidance.py bags/headon_full_*
    python3 analyse_avoidance.py --csv results.csv bags/*
"""

import argparse
import glob
import math
import os
import sys
from bisect import bisect_right

try:
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError as e:
    sys.exit(
        f"Missing ROS 2 Python packages ({e}).\n"
        "Run this inside the container with /opt/ros/jazzy sourced."
    )


# ---------------------------------------------------------------------
# Bag reading
# ---------------------------------------------------------------------

def read_bag(path):
    """Return {topic: [(t_nanosec, msg), ...]} for the topics we need."""
    wanted = {
        "/person_ground_truth",
        "/odom",
        "/tf",
        "/tf_static",
        "/amcl_pose",
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


def stamp_to_sec(header_stamp):
    return header_stamp.sec + header_stamp.nanosec * 1e-9


def yaw_from_quat(q):
    # Planar: only the z/w components matter.
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


# ---------------------------------------------------------------------
# Frame composition
# ---------------------------------------------------------------------

def extract_map_to_odom(tf_msgs):
    """Collect (t, x, y, yaw) for every map->odom transform, sorted."""
    out = []
    for _, msg in tf_msgs:
        for tr in msg.transforms:
            if tr.header.frame_id == "map" and tr.child_frame_id == "odom":
                out.append((
                    stamp_to_sec(tr.header.stamp),
                    tr.transform.translation.x,
                    tr.transform.translation.y,
                    yaw_from_quat(tr.transform.rotation),
                ))
    out.sort(key=lambda r: r[0])
    return out


def robot_path_in_map(odom_msgs, map_to_odom):
    """Lift every odom pose into the map frame. Returns [(t, x, y), ...]."""
    if not map_to_odom:
        # No AMCL correction recorded: odom and map coincide. Note this
        # is only valid if the robot was localised at the origin.
        return [
            (stamp_to_sec(m.header.stamp),
             m.pose.pose.position.x,
             m.pose.pose.position.y)
            for _, m in odom_msgs
        ]

    times = [r[0] for r in map_to_odom]
    path = []

    for _, m in odom_msgs:
        t = stamp_to_sec(m.header.stamp)
        ox = m.pose.pose.position.x
        oy = m.pose.pose.position.y

        # Zero-order hold: most recent map->odom at or before t.
        i = bisect_right(times, t) - 1
        if i < 0:
            i = 0
        _, mx, my, myaw = map_to_odom[i]

        c, s = math.cos(myaw), math.sin(myaw)
        path.append((t, mx + c * ox - s * oy, my + s * ox + c * oy))

    return path


def person_track(gt_msgs):
    """Returns [(t, x, y), ...] for person index 0 (single-person runs)."""
    out = []
    for _, m in gt_msgs:
        if not m.poses:
            continue
        p = m.poses[0].position
        out.append((stamp_to_sec(m.header.stamp), p.x, p.y))
    out.sort(key=lambda r: r[0])
    return out


def interp_person(person, t):
    """Linear interpolation of the person's position at time t.
    Returns None outside the recorded span rather than extrapolating."""
    times = [p[0] for p in person]
    if t < times[0] or t > times[-1]:
        return None

    i = bisect_right(times, t) - 1
    if i >= len(person) - 1:
        return person[-1][1], person[-1][2]

    t0, x0, y0 = person[i]
    t1, x1, y1 = person[i + 1]
    if t1 <= t0:
        return x0, y0

    f = (t - t0) / (t1 - t0)
    return x0 + f * (x1 - x0), y0 + f * (y1 - y0)


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def compute_metrics(path, person):
    if len(path) < 2:
        return None

    # --- path length ---
    length = 0.0
    for i in range(len(path) - 1):
        length += math.hypot(path[i + 1][1] - path[i][1],
                             path[i + 1][2] - path[i][2])

    x0, y0 = path[0][1], path[0][2]
    x1, y1 = path[-1][1], path[-1][2]
    straight = math.hypot(x1 - x0, y1 - y0)

    # --- max lateral deviation from the start->end straight line ---
    max_dev = 0.0
    if straight > 1e-6:
        ux, uy = (x1 - x0) / straight, (y1 - y0) / straight
        for _, px, py in path:
            rx, ry = px - x0, py - y0
            max_dev = max(max_dev, abs(-rx * uy + ry * ux))

    # --- minimum robot-person distance ---
    min_dist = None
    min_at = None
    if person:
        for t, px, py in path:
            q = interp_person(person, t)
            if q is None:
                continue
            d = math.hypot(px - q[0], py - q[1])
            if min_dist is None or d < min_dist:
                min_dist = d
                min_at = (t, px, py, q[0], q[1])

    duration = path[-1][0] - path[0][0]

    return {
        "min_distance": min_dist,
        "min_at": min_at,
        "path_length": length,
        "straight_line": straight,
        "path_ratio": length / straight if straight > 1e-6 else float("nan"),
        "max_lateral_dev": max_dev,
        "duration_s": duration,
        "mean_speed": length / duration if duration > 0 else float("nan"),
        "n_odom": len(path),
        "n_person": len(person),
        "start": (x0, y0),
        "end": (x1, y1),
    }


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------

def analyse(path):
    name = os.path.basename(path.rstrip("/"))
    try:
        data = read_bag(path)
    except Exception as e:
        print(f"[{name}] FAILED to read: {e}")
        return None

    tf_all = data["/tf"] + data["/tf_static"]
    m2o = extract_map_to_odom(tf_all)
    robot = robot_path_in_map(data["/odom"], m2o)
    person = person_track(data["/person_ground_truth"])

    if not robot:
        print(f"[{name}] no /odom messages — skipping")
        return None

    m = compute_metrics(robot, person)
    if m is None:
        print(f"[{name}] insufficient data — skipping")
        return None

    m["trial"] = name
    if not m2o:
        print(f"[{name}] WARNING: no map->odom in /tf; "
              f"treating odom as map (check localisation was running)")
    if m["min_distance"] is None:
        print(f"[{name}] WARNING: no overlapping ground truth; "
              f"min_distance unavailable")

    return m


def print_report(results):
    print()
    print("=" * 78)
    print("RTF-SAFE METRICS (geometric — valid regardless of real-time factor)")
    print("=" * 78)
    hdr = f"{'trial':<28} {'min_dist':>9} {'path_len':>9} {'ratio':>7} {'lat_dev':>8}"
    print(hdr)
    print("-" * 78)
    for m in results:
        md = f"{m['min_distance']:.3f}" if m["min_distance"] is not None else "n/a"
        print(f"{m['trial']:<28} {md:>9} {m['path_length']:>9.3f} "
              f"{m['path_ratio']:>7.3f} {m['max_lateral_dev']:>8.3f}")

    print()
    print("=" * 78)
    print("RTF-SENSITIVE (do NOT compare across runs at different RTF)")
    print("=" * 78)
    print(f"{'trial':<28} {'duration':>9} {'mean_spd':>9} {'n_odom':>8} {'n_person':>9}")
    print("-" * 78)
    for m in results:
        print(f"{m['trial']:<28} {m['duration_s']:>9.2f} {m['mean_speed']:>9.3f} "
              f"{m['n_odom']:>8} {m['n_person']:>9}")

    # --- group by condition, i.e. trial name minus the _tNN suffix ---
    groups = {}
    for m in results:
        key = m["trial"].rsplit("_t", 1)[0]
        groups.setdefault(key, []).append(m)

    if len(groups) > 1 or any(len(v) > 1 for v in groups.values()):
        print()
        print("=" * 78)
        print("AGGREGATE BY CONDITION (mean +/- sample std)")
        print("=" * 78)
        print(f"{'condition':<28} {'n':>3} {'min_dist':>16} {'path_len':>16}")
        print("-" * 78)
        for key in sorted(groups):
            g = groups[key]
            mds = [x["min_distance"] for x in g if x["min_distance"] is not None]
            pls = [x["path_length"] for x in g]
            print(f"{key:<28} {len(g):>3} "
                  f"{fmt_stat(mds):>16} {fmt_stat(pls):>16}")


def fmt_stat(vals):
    if not vals:
        return "n/a"
    mean = sum(vals) / len(vals)
    if len(vals) < 2:
        return f"{mean:.3f}"
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    return f"{mean:.3f}+/-{var ** 0.5:.3f}"


def write_csv(results, out_path):
    cols = ["trial", "min_distance", "path_length", "straight_line",
            "path_ratio", "max_lateral_dev", "duration_s", "mean_speed",
            "n_odom", "n_person"]
    with open(out_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for m in results:
            row = []
            for c in cols:
                v = m.get(c)
                row.append("" if v is None else
                           (f"{v:.4f}" if isinstance(v, float) else str(v)))
            f.write(",".join(row) + "\n")
    print(f"\nWrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("bags", nargs="+", help="bag directories (globs OK)")
    ap.add_argument("--csv", help="also write results to this CSV path")
    args = ap.parse_args()

    paths = []
    for pattern in args.bags:
        expanded = sorted(glob.glob(pattern))
        paths.extend(expanded if expanded else [pattern])

    results = []
    for p in paths:
        if not os.path.isdir(p):
            continue
        r = analyse(p)
        if r is not None:
            results.append(r)

    if not results:
        sys.exit("No bags analysed.")

    print_report(results)

    if args.csv:
        write_csv(results, args.csv)


if __name__ == "__main__":
    main()
