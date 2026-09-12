#!/usr/bin/env python3
"""
compare_perception_error.py

Quantifies WHERE the ~1.2 m lateral offset in SocialCritic's target
positions comes from.

THE QUESTION
------------
SocialCritic logs person positions in the odom frame:

    SocialCritic: 2 target(s) in odom, first=(-5.28, 1.25) scale=1.00

The person walks a straight line along y = 0 in the map frame
(move_person_oneway.py: (8,0) -> (-3,0)). Transformed into odom that
should stay near y = 0, but the logged value is 1.2-1.6 m. The x column
agrees with theory (-8.70 logged vs -8.9 predicted), so the transform
code is not simply mirrored or swapped.

Three candidates, and they separate cleanly by their signature:

  A. AMCL yaw drift in map->odom
       Ground truth lifted into odom via /tf already carries the same
       error, so GT_odom and critic agree, and BOTH differ from the
       naive transform built from the robot's spawn pose. Error grows
       with distance from the robot (it is a lever arm).

  B. YOLO/depth reprojection bias
       Ground truth lifted into odom sits near y=0, the critic does not.
       The residual is then perception, not localisation.

  C. Both.

WHAT THIS SCRIPT DOES
---------------------
1. Reads /person_ground_truth (map) and /tf (map->odom) from the bag.
2. Lifts ground truth into odom using the SAME zero-order-hold logic
   analyse_avoidance.py uses, so the two tools cannot disagree.
3. Parses the SocialCritic log lines for observed positions in odom.
4. Matches each log line to the nearest ground-truth sample in time and
   reports the residual, split into along-track and cross-track.

USAGE
-----
    python3 compare_perception_error.py \
        --bag bags/socialcritic_w40_coast5.0_t01/data \
        --log nav2_121413.log

Add --spawn-x/-y/-yaw (defaults match the usual launch args) to also
print the naive spawn-pose transform, which is what tells A apart from B.
"""

import argparse
import math
import re
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
# Log parsing
# ---------------------------------------------------------------------

# Preferred format — t= carries SIM time, which is what the bag uses:
#   ...: SocialCritic: t=51.234 2 target(s) in odom, first=(-5.28, 1.25) scale=1.00
LOG_RE = re.compile(
    r"SocialCritic: t=(\d+\.\d+) (\d+) target\(s\) in (\S+), "
    r"first=\((-?\d+\.\d+), (-?\d+\.\d+)\)(?: scale=(\d+\.\d+))?"
)

# Legacy format without t=, where the only stamp available is rclcpp's
# wall-clock bracket. Kept so older logs still parse, but they will not
# align with a sim-time bag — the script says so rather than guessing.
LOG_RE_WALL = re.compile(
    r"\[(\d{10}\.\d+)\].*?SocialCritic: (\d+) target\(s\) in (\S+), "
    r"first=\((-?\d+\.\d+), (-?\d+\.\d+)\)(?: scale=(\d+\.\d+))?"
)


def parse_log(path):
    """Returns ([(t_sec, n_targets, frame, x, y, scale), ...], used_wall_clock)."""
    out = []
    wall = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = LOG_RE.search(line)
            if m:
                out.append((
                    float(m.group(1)), int(m.group(2)), m.group(3),
                    float(m.group(4)), float(m.group(5)),
                    float(m.group(6)) if m.group(6) else 1.0,
                ))
                continue
            m = LOG_RE_WALL.search(line)
            if m:
                wall.append((
                    float(m.group(1)), int(m.group(2)), m.group(3),
                    float(m.group(4)), float(m.group(5)),
                    float(m.group(6)) if m.group(6) else 1.0,
                ))

    if out:
        out.sort(key=lambda r: r[0])
        return out, False
    wall.sort(key=lambda r: r[0])
    return wall, True


# ---------------------------------------------------------------------
# Bag reading — deliberately mirrors analyse_avoidance.py
# ---------------------------------------------------------------------

def read_bag(path):
    wanted = {"/person_ground_truth", "/odom", "/tf", "/tf_static"}
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


def stamp_to_sec(h):
    return h.sec + h.nanosec * 1e-9


def yaw_from_quat(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


def extract_map_to_odom(tf_msgs):
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


def person_track(gt_msgs):
    out = []
    for _, m in gt_msgs:
        if not m.poses:
            continue
        p = m.poses[0].position
        out.append((stamp_to_sec(m.header.stamp), p.x, p.y))
    out.sort(key=lambda r: r[0])
    return out


def robot_track_odom(odom_msgs):
    """Robot pose in the odom frame — straight out of /odom, no transform."""
    return [
        (stamp_to_sec(m.header.stamp),
         m.pose.pose.position.x,
         m.pose.pose.position.y)
        for _, m in odom_msgs
    ]


def interp(track, t):
    """Linear interpolation; None outside the recorded span."""
    if not track or t < track[0][0] or t > track[-1][0]:
        return None
    times = [r[0] for r in track]
    i = bisect_right(times, t) - 1
    if i < 0:
        return (track[0][1], track[0][2])
    if i >= len(track) - 1:
        return (track[-1][1], track[-1][2])
    t0, x0, y0 = track[i]
    t1, x1, y1 = track[i + 1]
    if t1 - t0 < 1e-9:
        return (x0, y0)
    a = (t - t0) / (t1 - t0)
    return (x0 + a * (x1 - x0), y0 + a * (y1 - y0))


# ---------------------------------------------------------------------
# Frame maths
# ---------------------------------------------------------------------

def map_to_odom_via_tf(mx, my, m2o, t):
    """Apply the recorded map->odom transform (zero-order hold).

    /tf publishes map->odom, i.e. odom expressed in map. To send a POINT
    from map into odom we need the inverse of that pose.
    """
    if not m2o:
        return None
    times = [r[0] for r in m2o]
    i = bisect_right(times, t) - 1
    if i < 0:
        i = 0
    _, ox, oy, oyaw = m2o[i]
    dx = mx - ox
    dy = my - oy
    c, s = math.cos(-oyaw), math.sin(-oyaw)
    return (c * dx - s * dy, s * dx + c * dy)


def map_to_odom_via_spawn(mx, my, sx, sy, syaw):
    """Naive transform assuming odom origin sits exactly at the spawn pose.

    This is what the transform WOULD be with perfect localisation. Any
    disagreement with map_to_odom_via_tf is AMCL drift.
    """
    dx = mx - sx
    dy = my - sy
    c, s = math.cos(-syaw), math.sin(-syaw)
    return (c * dx - s * dy, s * dx + c * dy)


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("--bag", required=True, help="bag dir (the one with metadata.yaml)")
    ap.add_argument("--log", required=True, help="nav2 log containing SocialCritic lines")
    ap.add_argument("--spawn-x", type=float, default=-1.0)
    ap.add_argument("--spawn-y", type=float, default=0.0)
    ap.add_argument("--spawn-yaw", type=float, default=3.14)
    ap.add_argument("--max-dt", type=float, default=0.3,
                    help="skip log lines with no GT sample this close in time")
    args = ap.parse_args()

    data = read_bag(args.bag)
    m2o = extract_map_to_odom(data["/tf"] + data["/tf_static"])
    gt = person_track(data["/person_ground_truth"])
    robot = robot_track_odom(data["/odom"])
    obs, used_wall = parse_log(args.log)

    if not gt:
        sys.exit("No /person_ground_truth in bag.")
    if not obs:
        sys.exit("No SocialCritic target lines matched in the log.")

    print(f"bag       : {args.bag}")
    print(f"log       : {args.log}")
    print(f"GT samples: {len(gt)}   span {gt[0][0]:.1f} .. {gt[-1][0]:.1f}")
    print(f"log lines : {len(obs)}  span {obs[0][0]:.1f} .. {obs[-1][0]:.1f}")
    print(f"map->odom : {len(m2o)} transforms recorded")

    # Overlap check first — mismatched clocks are the usual reason this
    # kind of comparison silently produces nothing.
    lo = max(gt[0][0], obs[0][0])
    hi = min(gt[-1][0], obs[-1][0])
    if hi <= lo:
        print()
        print("NO TIME OVERLAP between bag and log.")
        if used_wall:
            print()
            print("This log predates the sim-time stamp: its lines carry only")
            print("rclcpp's wall-clock bracket, while the bag is in sim time.")
            print("These two clocks can never align. Rebuild social_critic with")
            print("the t=%.3f field and re-run the trial, then compare that log.")
        else:
            print("Both sides carry sim time but the spans do not intersect —")
            print("this log is almost certainly from a different run than this")
            print("bag. Match them by trial, not by recency.")
        return

    rows = []
    for (t, n, frame, ox, oy, scale) in obs:
        g = interp(gt, t)
        if g is None:
            continue
        gt_odom = map_to_odom_via_tf(g[0], g[1], m2o, t)
        if gt_odom is None:
            continue
        spawn_odom = map_to_odom_via_spawn(
            g[0], g[1], args.spawn_x, args.spawn_y, args.spawn_yaw)
        r = interp(robot, t)

        ex = ox - gt_odom[0]
        ey = oy - gt_odom[1]
        drift_y = gt_odom[1] - spawn_odom[1]

        range_to_robot = (
            math.hypot(gt_odom[0] - r[0], gt_odom[1] - r[1]) if r else float("nan"))

        rows.append((t, n, scale, ox, oy, gt_odom[0], gt_odom[1],
                     ex, ey, drift_y, range_to_robot))

    if not rows:
        print("\nNo log line could be matched to a ground-truth sample.")
        return

    print()
    print("=" * 104)
    print("PER-SAMPLE: critic's target vs ground truth, both in odom")
    print("=" * 104)
    print(f"{'t':>12} {'n':>2} {'scale':>5} "
          f"{'obs_x':>8} {'obs_y':>8} {'gt_x':>8} {'gt_y':>8} "
          f"{'err_x':>8} {'err_y':>8} {'amcl_dy':>8} {'range':>7}")
    print("-" * 104)
    for r in rows:
        print(f"{r[0]:>12.2f} {r[1]:>2} {r[2]:>5.2f} "
              f"{r[3]:>8.2f} {r[4]:>8.2f} {r[5]:>8.2f} {r[6]:>8.2f} "
              f"{r[7]:>+8.2f} {r[8]:>+8.2f} {r[9]:>+8.2f} {r[10]:>7.2f}")

    # --- attribution ---
    ex = [r[7] for r in rows]
    ey = [r[8] for r in rows]
    dy = [r[9] for r in rows]

    def stats(v):
        mean = sum(v) / len(v)
        if len(v) < 2:
            return mean, 0.0
        var = sum((x - mean) ** 2 for x in v) / (len(v) - 1)
        return mean, var ** 0.5

    mex, sex = stats(ex)
    mey, sey = stats(ey)
    mdy, sdy = stats(dy)

    print()
    print("=" * 72)
    print("ATTRIBUTION")
    print("=" * 72)
    print(f"n matched samples          : {len(rows)}")
    print(f"perception err_x (obs-gt)  : {mex:+.3f} +/- {sex:.3f} m")
    print(f"perception err_y (obs-gt)  : {mey:+.3f} +/- {sey:.3f} m")
    print(f"AMCL lateral drift         : {mdy:+.3f} +/- {sdy:.3f} m")
    print()

    # Fresh observations only — coasted targets carry extrapolation error
    # on top of perception error, so they muddy the attribution.
    fresh = [r for r in rows if r[2] >= 0.99]
    if fresh:
        fex, _ = stats([r[7] for r in fresh])
        fey, _ = stats([r[8] for r in fresh])
        print(f"fresh observations only (scale=1.00, n={len(fresh)}):")
        print(f"  err_x {fex:+.3f} m   err_y {fey:+.3f} m")
        print()

    big_perception = abs(mey) > 0.3
    big_drift = abs(mdy) > 0.3

    if big_perception and big_drift:
        print("VERDICT: both layers contribute. Fix localisation first —")
        print("perception error is measured against a frame that is itself")
        print("moving, so the perception number is not yet trustworthy.")
    elif big_drift:
        print("VERDICT: AMCL drift dominates. The critic's targets are")
        print("roughly correct relative to ground truth; it is the odom")
        print("frame itself that is rotated/shifted. Check AMCL convergence")
        print("and whether the initial pose matches the spawn pose.")
    elif big_perception:
        print("VERDICT: perception bias dominates. Localisation is sound,")
        print("so the offset comes from YOLO bbox centring or the depth")
        print("reprojection in yolo_detector.py.")
    else:
        print("VERDICT: no large systematic offset in the matched samples.")
        print("If the critic still misses, the problem is coverage (gaps in")
        print("time) rather than accuracy (error in space).")

    # Lever-arm check: a yaw error shows up as cross-track error growing
    # with range, whereas a depth bias does not.
    far = [r for r in rows if r[10] > 5.0]
    near = [r for r in rows if r[10] < 4.0]
    if far and near:
        mf, _ = stats([r[8] for r in far])
        mn, _ = stats([r[8] for r in near])
        print()
        print(f"err_y at range >5 m : {mf:+.3f} m  (n={len(far)})")
        print(f"err_y at range <4 m : {mn:+.3f} m  (n={len(near)})")
        if abs(mf) > 1.6 * max(abs(mn), 1e-6):
            print("-> error grows with range: consistent with a YAW error")
            print("   (lever arm), not a constant translational offset.")
        else:
            print("-> error roughly constant with range: consistent with a")
            print("   translational offset rather than a yaw error.")


if __name__ == "__main__":
    main()