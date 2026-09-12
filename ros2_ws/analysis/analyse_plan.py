#!/usr/bin/env python3
"""analyse_plan.py - navigation metrics from a recorded /plan bag."""

import argparse
import math
import os
import sys


def read_paths(bag_dir):
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from nav_msgs.msg import Path
    except ImportError as e:
        print(f"ERROR: missing ROS python packages ({e}).")
        sys.exit(1)

    storage_options = rosbag2_py.StorageOptions(uri=bag_dir, storage_id="")
    converter_options = rosbag2_py.ConverterOptions("", "")

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    first_points = None
    count = 0

    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic != "/plan":
            continue
        msg = deserialize_message(data, Path)
        count += 1
        pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        # FIRST plan, not last: Nav2 republishes /plan as the robot
        # advances, so the final message is a short stub near the goal
        # (measured: 10 points, 0.28m) and says nothing about the route
        # chosen. The first message is the planner's decision from the
        # start pose, which is what the ablation compares.
        if first_points is None and pts:
            first_points = pts

    return first_points, count


def path_length(points):
    return sum(math.dist(points[i], points[i + 1]) for i in range(len(points) - 1))


def min_dist_to(points, target):
    return min(math.dist(p, target) for p in points)


def inside_o_space(points, pa, pb, body_clearance=0.25, half_width=0.35):
    cx, cy = (pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0
    dx, dy = pb[0] - pa[0], pb[1] - pa[1]
    sep = math.hypot(dx, dy)
    if sep < 1e-6:
        return False
    a = sep / 2.0 - body_clearance
    if a <= 0:
        return False
    ax, ay = dx / sep, dy / sep
    for px, py in points:
        rx, ry = px - cx, py - cy
        u = rx * ax + ry * ay
        v = -rx * ay + ry * ax
        if (u / a) ** 2 + (v / half_width) ** 2 <= 1.0:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bags", nargs="+")
    ap.add_argument("--person-a", nargs=2, type=float, default=[3.0, -0.75])
    ap.add_argument("--person-b", nargs=2, type=float, default=[3.0, 0.75])
    args = ap.parse_args()

    pa = tuple(args.person_a)
    pb = tuple(args.person_b)
    mid = ((pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0)

    print("=" * 78)
    print("PLAN ANALYSIS")
    print("=" * 78)
    print(f"person A: {pa}   person B: {pb}   midpoint: ({mid[0]:.2f}, {mid[1]:.2f})"
          f"   separation: {math.dist(pa, pb):.2f} m")
    print()
    print(f"{'run':<24} {'pts':>5} {'msgs':>5} {'min_clear':>10} "
          f"{'min_person':>11} {'length':>8} {'cuts through':>13}")
    print("-" * 78)

    for bag in args.bags:
        name = os.path.basename(bag.rstrip("/"))
        if not os.path.isdir(bag):
            print(f"{name:<24} BAG NOT FOUND")
            continue
        points, n_msgs = read_paths(bag)
        if not points:
            print(f"{name:<24} {'-':>5} {n_msgs:>5}  NO /plan MESSAGES")
            continue
        clear = min_dist_to(points, mid)
        near = min(min_dist_to(points, pa), min_dist_to(points, pb))
        length = path_length(points)
        cuts = inside_o_space(points, pa, pb)
        print(f"{name:<24} {len(points):>5} {n_msgs:>5} {clear:>10.3f} "
              f"{near:>11.3f} {length:>8.3f} {'YES' if cuts else 'no':>13}")

    print()
    print("min_clear  = closest approach to the pair's midpoint (higher = more compliant)")
    print("min_person = closest approach to the nearer individual")
    print("length     = path length (detour cost)")
    print("cuts       = did the path enter the o-space")


if __name__ == "__main__":
    main()
