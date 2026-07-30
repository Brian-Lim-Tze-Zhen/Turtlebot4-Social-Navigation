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


# Kept alongside person_tracks_all() as a documented single-person
# reference implementation; analyse() itself now only calls the latter.
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


def person_tracks_all(gt_msgs):
    """Returns {person_idx: [(t, x, y), ...]} for all people.

    /person_ground_truth is a PoseArray whose entries are in a fixed
    order matching the mover script's people list, so pose index ==
    person index and identity is stable across messages."""
    tracks = {}
    for _, m in gt_msgs:
        t = stamp_to_sec(m.header.stamp)
        for idx, pose in enumerate(m.poses):
            tracks.setdefault(idx, []).append((t, pose.position.x, pose.position.y))

    for track in tracks.values():
        track.sort(key=lambda r: r[0])

    return tracks


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

# THESIS ADDITION: separation threshold defining a person's "encounter
# window" for per-person commit_dist (see compute_metrics()).
ENCOUNTER_WINDOW_M = 3.0   # m


def compute_metrics(path, person_tracks):
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

    # --- max lateral deviation from the start->end straight line, plus
    # the full per-sample deviation series (reused below for the global
    # AND per-person commit_dist so both use one identical computation) ---
    max_dev = 0.0
    dev_series = []   # [(t, px, py, dev), ...], aligned with `path`
    if straight > 1e-6:
        ux, uy = (x1 - x0) / straight, (y1 - y0) / straight
        for t, px, py in path:
            rx, ry = px - x0, py - y0
            dev = -rx * uy + ry * ux           # signed; + = left
            dev_series.append((t, px, py, dev))
            max_dev = max(max_dev, abs(dev))

    # --- minimum robot-person distance, and each person's own encounter
    # window (contiguous span with separation < ENCOUNTER_WINDOW_M,
    # containing their own minimum separation) ---
    # THESIS MODIFICATION: min_distance is now the minimum over ALL
    # tracked people, not just whoever was at pose index 0. A 2+-person
    # bag used to silently score min_distance against a single arbitrary
    # person while looking entirely plausible.
    min_dist = None
    min_at = None
    closest_idx = None
    min_distance_per_person = {}
    windows_per_person = {}   # idx -> (window_start, window_end) or None

    for idx, track in person_tracks.items():
        p_min = None
        p_min_at = None
        segments = []
        seg_start = None
        prev_t = None
        for t, px, py in path:
            q = interp_person(track, t)
            sep = math.hypot(px - q[0], py - q[1]) if q is not None else None

            below = sep is not None and sep < ENCOUNTER_WINDOW_M
            if below and seg_start is None:
                seg_start = t
            if not below and seg_start is not None:
                segments.append((seg_start, prev_t))
                seg_start = None
            prev_t = t

            if sep is not None and (p_min is None or sep < p_min):
                p_min = sep
                p_min_at = (t, px, py, q[0], q[1])

        if seg_start is not None:
            segments.append((seg_start, prev_t))

        min_distance_per_person[idx] = p_min
        if p_min is not None and (min_dist is None or p_min < min_dist):
            min_dist = p_min
            min_at = p_min_at
            closest_idx = idx

        window = None
        if p_min_at is not None:
            p_min_t = p_min_at[0]
            for seg in segments:
                if seg[0] <= p_min_t <= seg[1]:
                    window = seg
                    break
        windows_per_person[idx] = window

    # --- THESIS ADDITION: early-commitment + pass-side metrics ---
    COMMIT_DEV_THRESH = 0.20   # m

    commit_dist = None
    commit_t = None
    dev_at_min = None
    # Cross-check against whichever person set the overall min_distance
    # (index 0 in the single-person case - same track as before).
    commit_track = person_tracks.get(closest_idx) if closest_idx is not None else None

    for t, px, py, dev in dev_series:
        if abs(dev) > COMMIT_DEV_THRESH:
            commit_t = t
            if commit_track:
                q = interp_person(commit_track, t)
                if q is not None:
                    commit_dist = math.hypot(px - q[0], py - q[1])
            break

    if min_at is not None and dev_series:
        _, mpx, mpy, _, _ = min_at
        rx, ry = mpx - x0, mpy - y0
        dev_at_min = -rx * uy + ry * ux

    # --- THESIS ADDITION: per-person commit_dist ---
    # Each person gets their own commit_dist, searched only within their
    # own encounter window, lower-bounded by the END of whichever other
    # person's window most recently preceded it (or trial start if none
    # did). This isolates each person's own swerve in a sequential
    # multi-person encounter without an arbitrary fixed lead-in duration.
    #
    # For a single person (or the first person chronologically), the
    # lower bound is simply trial start, so this reduces EXACTLY to the
    # global commit_dist search above - verified against
    # bags/headon_full_hl_t01, where the swerve begins ~9.3s before that
    # person's own encounter window even opens (anticipatory avoidance
    # from the predictive costmap, not a reaction to current proximity).
    # A fixed lead-in shorter than that gap would have missed it; using
    # window boundaries themselves as the bound avoids needing one.
    commit_dist_per_person = {idx: None for idx in person_tracks}
    ordered = sorted(
        (idx for idx in person_tracks if windows_per_person.get(idx) is not None),
        key=lambda idx: windows_per_person[idx][0],
    )
    lower_bound = path[0][0]
    for idx in ordered:
        w_start, w_end = windows_per_person[idx]
        track = person_tracks[idx]
        for t, px, py, dev in dev_series:
            if t < lower_bound:
                continue
            if t > w_end:
                break
            if abs(dev) > COMMIT_DEV_THRESH:
                q = interp_person(track, t)
                if q is not None:
                    commit_dist_per_person[idx] = math.hypot(px - q[0], py - q[1])
                break
        lower_bound = w_end

    duration = path[-1][0] - path[0][0]

    result = {
        "min_distance": min_dist,
        "min_at": min_at,
        "path_length": length,
        "straight_line": straight,
        "path_ratio": length / straight if straight > 1e-6 else float("nan"),
        "max_lateral_dev": max_dev,
        "commit_dist": commit_dist,
        "commit_t": commit_t,
        "dev_at_min": dev_at_min,
        "commit_dist_per_person": commit_dist_per_person,
        "duration_s": duration,
        "mean_speed": length / duration if duration > 0 else float("nan"),
        "n_odom": len(path),
        "n_person": len(person_tracks.get(0, [])),
        "n_people": len(person_tracks),
        "start": (x0, y0),
        "end": (x1, y1),
    }

    if len(person_tracks) > 1:
        result["min_distance_per_person"] = min_distance_per_person
        result["closest_person_idx"] = closest_idx

    return result


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
    person_tracks = person_tracks_all(data["/person_ground_truth"])

    if not robot:
        print(f"[{name}] no /odom messages — skipping")
        return None

    m = compute_metrics(robot, person_tracks)
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
    # THESIS ADDITION: only show the per-trial closest-person column when
    # some trial actually has more than one tracked person, so existing
    # single-person output is unchanged byte-for-byte.
    has_multi = any(m.get("n_people", 1) > 1 for m in results)

    print()
    print("=" * 96)
    print("RTF-SAFE METRICS (geometric — valid regardless of real-time factor)")
    print("=" * 96)
    hdr = (f"{'trial':<28} {'min_dist':>9} {'path_len':>9} {'ratio':>7} {'lat_dev':>8} "
           f"{'commit_d':>9} {'side_dev':>9}")
    if has_multi:
        hdr += f" {'person':>6}"
    print(hdr)
    print("-" * 96)
    for m in results:
        md = f"{m['min_distance']:.3f}" if m["min_distance"] is not None else "n/a"
        cd = f"{m.get('commit_dist'):.3f}" if m.get("commit_dist") is not None else "n/a"
        sd = f"{m.get('dev_at_min'):+.3f}" if m.get("dev_at_min") is not None else "n/a"
        row = (f"{m['trial']:<28} {md:>9} {m['path_length']:>9.3f} "
              f"{m['path_ratio']:>7.3f} {m['max_lateral_dev']:>8.3f} "
              f"{cd:>9} {sd:>9}")
        if has_multi:
            pidx = m.get("closest_person_idx")
            row += f" {(str(pidx) if pidx is not None else '-'):>6}"
        print(row)

    # THESIS ADDITION: per-person commit_dist, printed as its own block
    # rather than widening the table above - the existing commit_d
    # column keeps its current (whole-trial) meaning unchanged.
    if has_multi:
        print()
        print("=" * 96)
        print("PER-PERSON COMMIT DISTANCES (commit_dist_per_person)")
        print("=" * 96)
        for m in results:
            cdp = m.get("commit_dist_per_person")
            if not cdp:
                continue
            parts = [
                f"person{idx}={cdp[idx]:.3f}" if cdp[idx] is not None else f"person{idx}=n/a"
                for idx in sorted(cdp)
            ]
            print(f"{m['trial']:<28} " + "  ".join(parts))

    print()
    print("=" * 78)
    print("RTF-SENSITIVE (do NOT compare across runs at different RTF)")
    print("=" * 78)
    print(f"{'trial':<28} {'duration':>9} {'mean_spd':>9} {'n_odom':>8} {'n_person':>9} "
          f"{'n_people':>9}")
    print("-" * 78)
    for m in results:
        print(f"{m['trial']:<28} {m['duration_s']:>9.2f} {m['mean_speed']:>9.3f} "
              f"{m['n_odom']:>8} {m['n_person']:>9} {m.get('n_people', 1):>9}")

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
            "path_ratio", "max_lateral_dev", "commit_dist", "dev_at_min",
            "duration_s", "mean_speed", "n_odom", "n_person"]
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
