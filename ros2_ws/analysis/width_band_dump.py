#!/usr/bin/env python3
"""
width_band_dump.py

THESIS CALIBRATION - measures the metric-width distribution of WALL
FRAGMENT clusters vs. real LEG clusters, to decide whether a width
floor can separate them at all.

WHY: leg_detector_node's MIN_CLUSTER_METRIC_WIDTH=0.03 was chosen to
sit safely below the narrowest measured leg (0.048m from
cluster_sweep.py). It was never checked against what it is supposed to
REJECT. The occlusion test then produced persistent spurious tracks
(one lasting ~20s) at y ~ -4.91 - a wall behind the occlusion waypoint,
fragmented into short segments by the person standing in front of it.

A floor only works if there is a GAP between the two width
distributions. This script measures both bands so that question can be
answered from data:

  - fragments clearly NARROWER than legs -> raise the floor, done
  - bands OVERLAP                        -> no floor can work; need a
                                            shape/linearity test instead
                                            (walls are collinear, legs
                                            are convex arcs)

METHOD: park person_1 at the occlusion waypoint (3, -4.0) where the
fragments appeared, capture N scans, and report EVERY cluster passing
the current point-count filter - with no width filtering at all - each
labelled by map region so the two populations can be compared directly.
"""

import math
import subprocess
import time

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

import tf2_ros
import tf2_geometry_msgs  # noqa: F401

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PointStamped


WORLD_NAME = "conversation_test"
MODEL_NAME = "person_1"

# Park at the occlusion waypoint - this is where the spurious wall
# fragments appeared during the occlusion test.
PARK_XY = (3.0, -4.0)
PARK_YAW = 0.0          # facing -Y, matching the mover's walk-out facing

PERSON_2_XY = (3.0, 0.5)

SCANS = 5
SETTLE_SECONDS = 3.0

# Mirror leg_detector_node's clustering exactly.
CLUSTER_JUMP_THRESHOLD = 0.5
MIN_CLUSTER_POINTS = 2
MAX_CLUSTER_POINTS = 20
MIN_VALID_RANGE = 0.164
MAX_VALID_RANGE = 8.0

# Current floor under evaluation - reported, NOT applied.
CURRENT_MIN_WIDTH = 0.03
CURRENT_MAX_WIDTH = 0.40

# Region attribution radii (map frame)
PERSON_RADIUS = 0.8


def set_pose(x, y, yaw):
    qz, qw = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
    req = (
        f"name: '{MODEL_NAME}', position: {{x: {x}, y: {y}, z: 0}}, "
        f"orientation: {{x: 0, y: 0, z: {qz}, w: {qw}}}"
    )
    subprocess.run(
        ["gz", "service", "-s", f"/world/{WORLD_NAME}/set_pose",
         "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
         "--timeout", "3000", "--req", req],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5.0
    )


class WidthBandDump(Node):
    def __init__(self):
        super().__init__("width_band_dump")
        self.scan = None
        self.buf = tf2_ros.Buffer()
        self.lis = tf2_ros.TransformListener(self.buf, self)
        self.create_subscription(LaserScan, "/scan", self.cb, 10)

    def cb(self, msg):
        self.scan = msg

    def cluster_scan(self, msg):
        out, cur, prev = [], [], None
        for i, r in enumerate(msg.ranges):
            if not math.isfinite(r) or not (MIN_VALID_RANGE <= r <= MAX_VALID_RANGE):
                if cur:
                    out.append(list(cur))
                    cur.clear()
                prev = None
                continue
            if prev is not None and abs(r - prev) > CLUSTER_JUMP_THRESHOLD:
                if cur:
                    out.append(list(cur))
                    cur.clear()
            cur.append((i, r))
            prev = r
        if cur:
            out.append(list(cur))
        return out

    def to_map(self, cx, cy, frame_id, stamp):
        pt = PointStamped()
        pt.header.frame_id = frame_id
        pt.point.x, pt.point.y = cx, cy
        for s in (stamp, rclpy.time.Time().to_msg()):
            pt.header.stamp = s
            try:
                o = self.buf.transform(pt, "map", timeout=Duration(seconds=0.1))
                return o.point.x, o.point.y
            except Exception:
                continue
        return None

    def linearity(self, msg, cluster):
        """RMS perpendicular deviation from the best-fit line through the
        cluster's Cartesian points, in metres. Walls -> near 0; convex
        arcs (legs/torso) -> clearly positive."""
        pts = []
        for idx, r in cluster:
            a = msg.angle_min + idx * msg.angle_increment
            pts.append((r * math.cos(a), r * math.sin(a)))

        n = len(pts)
        if n < 3:
            return 0.0

        mx = sum(p[0] for p in pts) / n
        my = sum(p[1] for p in pts) / n
        sxx = sum((p[0] - mx) ** 2 for p in pts)
        syy = sum((p[1] - my) ** 2 for p in pts)
        sxy = sum((p[0] - mx) * (p[1] - my) for p in pts)

        theta = 0.5 * math.atan2(2 * sxy, sxx - syy)
        ax, ay = math.cos(theta), math.sin(theta)

        sq = 0.0
        for px, py in pts:
            rx, ry = px - mx, py - my
            perp = -rx * ay + ry * ax
            sq += perp * perp
        return math.sqrt(sq / n)


def region_of(mx, my, park_xy):
    if math.hypot(mx - park_xy[0], my - park_xy[1]) <= PERSON_RADIUS:
        return "PERSON_1"
    if math.hypot(mx - PERSON_2_XY[0], my - PERSON_2_XY[1]) <= PERSON_RADIUS:
        return "PERSON_2"
    return "other/wall"


def main():
    rclpy.init()
    node = WidthBandDump()

    print("=" * 78)
    print("WIDTH BAND DUMP - wall fragments vs. real legs")
    print("=" * 78)
    print(f"Parking {MODEL_NAME} at {PARK_XY}, yaw={PARK_YAW}")
    print(f"Current floor under evaluation: {CURRENT_MIN_WIDTH} m "
          f"(reported, NOT applied)")
    set_pose(PARK_XY[0], PARK_XY[1], PARK_YAW)
    time.sleep(SETTLE_SECONDS)

    by_region = {}
    seen, last = 0, None
    deadline = time.time() + 40.0

    while seen < SCANS and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.5)
        msg = node.scan
        if msg is None:
            continue
        key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        if key == last:
            continue
        last = key
        seen += 1

        print(f"\n=== scan {seen} ===")
        for cl in node.cluster_scan(msg):
            n = len(cl)
            if not (MIN_CLUSTER_POINTS <= n <= MAX_CLUSTER_POINTS):
                continue

            aw = (cl[-1][0] - cl[0][0]) * msg.angle_increment
            mr = sum(r for _, r in cl) / n
            mw = aw * mr
            lin = node.linearity(msg, cl)

            xs = [r * math.cos(msg.angle_min + i * msg.angle_increment) for i, r in cl]
            ys = [r * math.sin(msg.angle_min + i * msg.angle_increment) for i, r in cl]
            mp = node.to_map(sum(xs) / n, sum(ys) / n, msg.header.frame_id,
                             msg.header.stamp)
            if mp is None:
                continue

            reg = region_of(mp[0], mp[1], PARK_XY)
            passes = CURRENT_MIN_WIDTH <= mw <= CURRENT_MAX_WIDTH

            print(f"  {reg:<11} pts={n:2d} width={mw:.4f}m "
                  f"lin={lin:.4f}m range={mr:5.2f}m "
                  f"map=({mp[0]:6.2f},{mp[1]:6.2f}) "
                  f"{'[passes floor]' if passes else '[rejected]'}")

            by_region.setdefault(reg, []).append((mw, lin, n))

    print("\n" + "=" * 78)
    print("WIDTH BANDS BY REGION")
    print("=" * 78)
    print(f"{'region':<12} {'count':>6} {'width_min':>10} {'width_max':>10} "
          f"{'lin_min':>9} {'lin_max':>9}")
    for reg, vals in sorted(by_region.items()):
        ws = [v[0] for v in vals]
        ls = [v[1] for v in vals]
        print(f"{reg:<12} {len(vals):>6} {min(ws):>10.4f} {max(ws):>10.4f} "
              f"{min(ls):>9.4f} {max(ls):>9.4f}")

    person = [v for r, vs in by_region.items() if r.startswith("PERSON") for v in vs]
    wall = by_region.get("other/wall", [])

    print("\n" + "-" * 78)
    print("VERDICT")
    print("-" * 78)
    if not wall:
        print("  No wall/other clusters captured - fragments may be")
        print("  position-dependent. Try a different park position.")
    elif not person:
        print("  No person clusters captured - check park position/TF.")
    else:
        pw_min = min(v[0] for v in person)
        ww_max = max(v[0] for v in wall)
        print(f"  narrowest PERSON cluster : {pw_min:.4f} m")
        print(f"  widest    WALL   cluster : {ww_max:.4f} m")
        if ww_max < pw_min:
            mid = (ww_max + pw_min) / 2.0
            print(f"  -> GAP EXISTS. A width floor of ~{mid:.4f} m separates them.")
        else:
            print("  -> BANDS OVERLAP. No width floor can separate these.")
            pl_min = min(v[1] for v in person)
            wl_max = max(v[1] for v in wall)
            print(f"     linearity: person min={pl_min:.4f}  wall max={wl_max:.4f}")
            if wl_max < pl_min:
                mid = (wl_max + pl_min) / 2.0
                print(f"     -> but LINEARITY separates them at ~{mid:.4f} m.")
            else:
                print("     -> linearity does not separate them either;")
                print("        needs a different discriminator.")

    print(f"\nRestoring {MODEL_NAME} to (3, -0.5) yaw=pi ...")
    set_pose(3.0, -0.5, math.pi)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
