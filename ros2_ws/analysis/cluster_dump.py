#!/usr/bin/env python3
"""
cluster_dump.py — parks person_1 and dumps EVERY cluster passing
leg_detector_node's current filter, with map coords, so a spurious
detection can be told apart from an unmerged leg pair.
"""
import math, subprocess, time
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import tf2_ros
import tf2_geometry_msgs  # noqa: F401
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PointStamped

WORLD, MODEL = "conversation_test", "person_1"
PARK_XY, PARK_YAW = (3.0, -3.0), 0.0

# must mirror leg_detector_node.py exactly
JUMP = 0.5
MIN_PTS, MAX_PTS = 2, 20
MIN_W, MAX_W = 0.03, 0.40
MIN_R, MAX_R = 0.164, 8.0
PAIR_DIST = 0.40


def set_pose(x, y, yaw):
    qz, qw = math.sin(yaw / 2), math.cos(yaw / 2)
    req = (f"name: '{MODEL}', position: {{x: {x}, y: {y}, z: 0}}, "
           f"orientation: {{x: 0, y: 0, z: {qz}, w: {qw}}}")
    subprocess.run(["gz", "service", "-s", f"/world/{WORLD}/set_pose",
                    "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
                    "--timeout", "3000", "--req", req],
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)


class Dump(Node):
    def __init__(self):
        super().__init__("cluster_dump")
        self.scan = None
        self.buf = tf2_ros.Buffer()
        self.lis = tf2_ros.TransformListener(self.buf, self)
        self.create_subscription(LaserScan, "/scan", self.cb, 10)

    def cb(self, m):
        self.scan = m

    def clusters(self, m):
        out, cur, prev = [], [], None
        for i, r in enumerate(m.ranges):
            if not math.isfinite(r) or not (MIN_R <= r <= MAX_R):
                if cur: out.append(list(cur)); cur.clear()
                prev = None; continue
            if prev is not None and abs(r - prev) > JUMP:
                if cur: out.append(list(cur)); cur.clear()
            cur.append((i, r)); prev = r
        if cur: out.append(list(cur))
        return out

    def to_map(self, cx, cy, fid, stamp):
        p = PointStamped(); p.header.frame_id = fid
        p.point.x, p.point.y = cx, cy
        for s in (stamp, rclpy.time.Time().to_msg()):
            p.header.stamp = s
            try:
                o = self.buf.transform(p, "map", timeout=Duration(seconds=0.1))
                return o.point.x, o.point.y
            except Exception:
                continue
        return None


def main():
    rclpy.init(); n = Dump()
    print(f"Parking {MODEL} at {PARK_XY}, yaw={PARK_YAW} ...")
    set_pose(PARK_XY[0], PARK_XY[1], PARK_YAW)
    time.sleep(3.0)

    shown, last = 0, None
    deadline = time.time() + 30
    while shown < 3 and time.time() < deadline:
        rclpy.spin_once(n, timeout_sec=0.5)
        m = n.scan
        if m is None: continue
        key = (m.header.stamp.sec, m.header.stamp.nanosec)
        if key == last: continue
        last = key; shown += 1

        print(f"\n=== scan {shown} ===")
        passed = []
        for cl in n.clusters(m):
            np_ = len(cl)
            aw = (cl[-1][0] - cl[0][0]) * m.angle_increment
            mr = sum(r for _, r in cl) / np_
            mw = aw * mr
            ok = (MIN_PTS <= np_ <= MAX_PTS) and (MIN_W <= mw <= MAX_W)
            if not ok: continue
            xs = [r * math.cos(m.angle_min + i * m.angle_increment) for i, r in cl]
            ys = [r * math.sin(m.angle_min + i * m.angle_increment) for i, r in cl]
            mp = n.to_map(sum(xs)/np_, sum(ys)/np_, m.header.frame_id, m.header.stamp)
            if mp is None: continue
            passed.append(mp)
            print(f"  PASS pts={np_:2d} width={mw:.3f}m range={mr:5.2f}m "
                  f"map=({mp[0]:6.2f},{mp[1]:6.2f})")

        print("  pairwise distances between passing clusters:")
        for i in range(len(passed)):
            for j in range(i + 1, len(passed)):
                d = math.hypot(passed[i][0]-passed[j][0], passed[i][1]-passed[j][1])
                flag = "MERGE" if d <= PAIR_DIST else "     "
                print(f"    [{i}]-[{j}] {d:6.3f}m {flag}")

    print(f"\nGround truth: person_1=({PARK_XY[0]}, {PARK_XY[1]})  person_2=(3.0, 0.5)")
    n.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
