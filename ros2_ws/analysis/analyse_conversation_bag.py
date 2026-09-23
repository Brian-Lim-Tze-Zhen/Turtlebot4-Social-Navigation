#!/usr/bin/env python3
"""
analyse_conversation_bag.py

Offline analysis of a conversation-group avoidance run (ablation F).

Reads a rosbag2 (mcap) and reports, against BOTH ground truth and the
robot's own perception:

  - minimum distance from the robot to each person over the run
  - clearance at closest approach, and when it happened
  - whether the executed trajectory crossed the o-space (the segment
    between the two people) or routed around it
  - speed through the zone
  - perception error: reported person positions vs ground truth

Robot pose: the bag's /tf carries only odom -> base_link (AMCL's
map -> odom was not recorded), so map-frame poses are reconstructed by
composing the dense odom -> base_link stream with the map -> odom
offset implied by each /amcl_pose sample. The offset is held between
AMCL samples, so pose accuracy between them is odometry-accurate -
fine over a one-minute run, and it preserves the full 2 kHz-ish pose
rate instead of dropping to AMCL's ~0.7 Hz.

Usage:
  python3 analyse_conversation_bag.py <bag_dir_or_mcap>

Ground truth poses are the gz model poses for this world; override with
--p1 / --p2 if the world changes.
"""

import argparse
import math
import sys

import rclpy.serialization as ser
import rosbag2_py
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

# Defaults: verified with `gz model -m person_N -p` for conversation_test
P1_DEFAULT = (3.00, -0.75)
P2_DEFAULT = (3.00, 0.75)

# A crossing is counted when the robot passes the pair's connecting line
# BETWEEN the two people, not outside them. Expressed as a fraction of
# the pair separation measured from the midpoint: 1.0 would be exactly
# at a person, so 0.9 keeps the test inside the o-space.
CROSS_SPAN_FRAC = 0.9


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def read_bag(path):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=path, storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        yield topic, data, stamp * 1e-9, types.get(topic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("--p1", nargs=2, type=float, default=P1_DEFAULT)
    ap.add_argument("--p2", nargs=2, type=float, default=P2_DEFAULT)
    args = ap.parse_args()

    p1, p2 = tuple(args.p1), tuple(args.p2)
    mid = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
    sep = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    # Unit vector along the pair (the o-space's long axis) and across it.
    ax = ((p2[0] - p1[0]) / sep, (p2[1] - p1[1]) / sep)
    perp = (-ax[1], ax[0])

    odom_tf = []     # (t, x, y, yaw) odom -> base_link, dense
    amcl = []        # (t, x, y, yaw) map -> base_link, sparse
    traj = []        # (t, x, y) robot pose in map frame, reconstructed
    speeds = []      # (t, linear speed)
    groups = []      # (t, cx, cy, half_length, half_width, buffer)
    perceived = {}   # track id -> list of (t, x, y)
    plans = []       # (t, [(x, y), ...])
    t0 = None

    for topic, data, t, typ in read_bag(args.bag):
        if t0 is None:
            t0 = t
        if topic == "/tf":
            msg = ser.deserialize_message(data, TFMessage)
            for tr in msg.transforms:
                if tr.header.frame_id == "map" and tr.child_frame_id == "base_link":
                    traj.append((t - t0,
                                 tr.transform.translation.x,
                                 tr.transform.translation.y))
                elif tr.header.frame_id == "odom" and tr.child_frame_id == "base_link":
                    odom_tf.append((t - t0,
                                    tr.transform.translation.x,
                                    tr.transform.translation.y,
                                    yaw_of(tr.transform.rotation)))
        elif topic == "/amcl_pose":
            msg = ser.deserialize_message(data, PoseWithCovarianceStamped)
            p = msg.pose.pose
            amcl.append((t - t0, p.position.x, p.position.y,
                         yaw_of(p.orientation)))
        elif topic == "/odom":
            msg = ser.deserialize_message(data, Odometry)
            v = msg.twist.twist.linear
            speeds.append((t - t0, math.hypot(v.x, v.y)))
        elif topic == "/social_groups":
            f = ser.deserialize_message(data, String).data.split(",")
            if len(f) >= 11:
                try:
                    groups.append((t - t0, float(f[2]), float(f[3]),
                                   float(f[6]), float(f[7]), float(f[10])))
                except ValueError:
                    pass
        elif topic == "/person_positions_fused":
            f = ser.deserialize_message(data, String).data.split(",")
            if len(f) >= 4:
                try:
                    perceived.setdefault(int(float(f[0])), []).append(
                        (t - t0, float(f[2]), float(f[3])))
                except ValueError:
                    pass
        elif topic == "/plan":
            msg = ser.deserialize_message(data, Path)
            plans.append((t - t0, [(p.pose.position.x, p.pose.position.y)
                                   for p in msg.poses]))

    if not traj and odom_tf and amcl:
        # Reconstruct map -> base_link. For each AMCL fix, solve the
        # map -> odom offset against the odom pose at the same instant,
        # then apply the most recent offset to every odom pose.
        offsets = []   # (t, dx, dy, dyaw)
        for at, axm, aym, ayaw in amcl:
            ot, ox, oy, oyaw = min(odom_tf, key=lambda p: abs(p[0] - at))
            dyaw = ayaw - oyaw
            c, s_ = math.cos(dyaw), math.sin(dyaw)
            offsets.append((at, axm - (c * ox - s_ * oy),
                            aym - (s_ * ox + c * oy), dyaw))
        offsets.sort()
        oi = 0
        for ot, ox, oy, oyaw in odom_tf:
            while oi + 1 < len(offsets) and offsets[oi + 1][0] <= ot:
                oi += 1
            _, dx, dy, dyaw = offsets[oi]
            c, s_ = math.cos(dyaw), math.sin(dyaw)
            traj.append((ot, dx + c * ox - s_ * oy, dy + s_ * ox + c * oy))
        print(f"Note: map->base_link absent; reconstructed {len(traj)} poses "
              f"from {len(odom_tf)} odom transforms and {len(amcl)} AMCL fixes")

    if not traj:
        print("No robot pose in the bag (need map->base_link, or "
              "odom->base_link plus /amcl_pose) - cannot analyse.")
        return 1

    print(f"Run duration        : {traj[-1][0]:.1f} s, {len(traj)} poses")
    print(f"Ground truth people : {p1} and {p2}, {sep:.2f} m apart")
    print(f"O-space midpoint    : ({mid[0]:.2f}, {mid[1]:.2f})")
    print()

    # ---- minimum distance to each person -----------------------------
    print("=== Clearance (ground truth) ===")
    closest = {}
    for name, p in (("person_1", p1), ("person_2", p2)):
        d = [(math.hypot(x - p[0], y - p[1]), t, x, y) for t, x, y in traj]
        dmin, tmin, xmin, ymin = min(d)
        closest[name] = (dmin, tmin, xmin, ymin)
        print(f"{name}: min distance {dmin:.3f} m at t={tmin:.1f}s, "
              f"robot at ({xmin:.2f}, {ymin:.2f})")
    dmin_any = min(v[0] for v in closest.values())
    print(f"Minimum distance to ANY person: {dmin_any:.3f} m")
    print()

    # ---- did the executed path cross the o-space? --------------------
    # Project each pose onto the pair's frame: u = along the pair from
    # the midpoint, v = across it (the robot's travel direction). A
    # crossing is a sign change in v while |u| stays inside the pair.
    print("=== O-space crossing (executed trajectory) ===")
    crossed = False
    cross_t = cross_u = None
    prev = None
    for t, x, y in traj:
        dx, dy = x - mid[0], y - mid[1]
        u = dx * ax[0] + dy * ax[1]       # along the pair
        v = dx * perp[0] + dy * perp[1]   # across the pair
        if prev is not None:
            v_prev, u_prev = prev
            if v_prev * v < 0:            # crossed the connecting line
                u_cross = (u + u_prev) / 2.0
                if abs(u_cross) <= sep / 2.0 * CROSS_SPAN_FRAC:
                    crossed = True
                    cross_t, cross_u = t, u_cross
                    break
        prev = (v, u)
    if crossed:
        print(f"CROSSED the o-space at t={cross_t:.1f}s, "
              f"{abs(cross_u):.2f} m from the midpoint along the pair axis")
    else:
        print("Did NOT cross between the people - routed around "
              "(or never reached them)")
    # Lateral offset from the midpoint at closest approach to the pair
    d_mid = [(math.hypot(x - mid[0], y - mid[1]), t, x, y) for t, x, y in traj]
    dm, tm, xm, ym = min(d_mid)
    dxm, dym = xm - mid[0], ym - mid[1]
    print(f"Closest approach to o-space midpoint: {dm:.3f} m at t={tm:.1f}s "
          f"(offset along pair {dxm * ax[0] + dym * ax[1]:+.2f} m, "
          f"across {dxm * perp[0] + dym * perp[1]:+.2f} m)")
    print()

    # ---- speed near the pair -----------------------------------------
    print("=== Speed ===")
    if speeds:
        near = []
        for t, v in speeds:
            pose = min(traj, key=lambda p: abs(p[0] - t))
            if math.hypot(pose[1] - mid[0], pose[2] - mid[1]) <= 2.0:
                near.append(v)
        overall = [v for _, v in speeds]
        print(f"Mean speed overall        : {sum(overall) / len(overall):.3f} m/s")
        print(f"Max speed overall         : {max(overall):.3f} m/s")
        if near:
            print(f"Mean speed within 2 m     : {sum(near) / len(near):.3f} m/s "
                  f"({len(near)} samples)")
        else:
            print("Robot never came within 2 m of the o-space midpoint")
    print()

    # ---- what the robot believed -------------------------------------
    print("=== Perception vs ground truth ===")
    if not perceived:
        print("No /person_positions_fused in the bag")
    for tid, pts in sorted(perceived.items()):
        xs = sum(p[1] for p in pts) / len(pts)
        ys = sum(p[2] for p in pts) / len(pts)
        # Match to whichever ground truth person is nearer
        d1 = math.hypot(xs - p1[0], ys - p1[1])
        d2 = math.hypot(xs - p2[0], ys - p2[1])
        name, err = ("person_1", d1) if d1 < d2 else ("person_2", d2)
        spread = max(math.hypot(p[1] - xs, p[2] - ys) for p in pts)
        print(f"id {tid}: {len(pts)} samples, mean ({xs:.3f}, {ys:.3f}) "
              f"-> {name}, bias {err:.3f} m, max spread {spread:.3f} m")
    print()

    # ---- the zone the planner was given ------------------------------
    print("=== Social zone ===")
    if not groups:
        print("No /social_groups in the bag - no zone was ever published")
    else:
        buffers = [g[5] for g in groups]
        narrow = sum(1 for b in buffers if b < 0.399)
        print(f"{len(groups)} group messages, t={groups[0][0]:.1f}s "
              f"to {groups[-1][0]:.1f}s")
        print(f"Buffer: min {min(buffers):.3f} m, max {max(buffers):.3f} m, "
              f"{narrow} sample(s) in the NARROW branch")
        cxs = sum(g[1] for g in groups) / len(groups)
        cys = sum(g[2] for g in groups) / len(groups)
        print(f"Mean zone centre ({cxs:.3f}, {cys:.3f}) vs true midpoint "
              f"({mid[0]:.2f}, {mid[1]:.2f}), error "
              f"{math.hypot(cxs - mid[0], cys - mid[1]):.3f} m")
    print()

    # ---- the plan Nav2 produced --------------------------------------
    print("=== Global plan ===")
    if not plans:
        print("No /plan in the bag")
    else:
        # Does the FINAL plan pass between the people?
        t_last, pts = plans[-1]
        plan_cross = False
        prev = None
        for x, y in pts:
            dx, dy = x - mid[0], y - mid[1]
            u = dx * ax[0] + dy * ax[1]
            v = dx * perp[0] + dy * perp[1]
            if prev is not None and prev[0] * v < 0:
                if abs((u + prev[1]) / 2.0) <= sep / 2.0 * CROSS_SPAN_FRAC:
                    plan_cross = True
                    break
            prev = (v, u)
        print(f"{len(plans)} plans, last at t={t_last:.1f}s with {len(pts)} poses")
        print("Final plan routes "
              + ("THROUGH the o-space" if plan_cross else "AROUND the o-space"))
        if pts:
            dmin_plan = min(
                min(math.hypot(x - p1[0], y - p1[1]),
                    math.hypot(x - p2[0], y - p2[1])) for x, y in pts)
            print(f"Final plan's closest point to a person: {dmin_plan:.3f} m")

    return 0


if __name__ == "__main__":
    sys.exit(main())
