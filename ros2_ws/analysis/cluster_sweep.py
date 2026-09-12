#!/usr/bin/env python3
"""
cluster_sweep.py

THESIS CALIBRATION HARNESS - measures raw lidar cluster properties for a
single person across a range/orientation sweep, so leg_detector_node.py's
filter thresholds can be picked from evidence instead of guessed.

WHY: leg_detector_node's MIN/MAX_CLUSTER_POINTS and MIN/MAX_CLUSTER_
ANGULAR_WIDTH were seeded from ONE data point (person at ~3.1m, facing
the robot). The occlusion test then failed at 5m with passed=1, i.e.
the moving person vanished from the filter entirely. Two suspected
causes, both untested:

  1. RANGE. A fixed angular-width floor is range-dependent in disguise:
     a target of fixed physical width subtends fewer beams the further
     away it is. MIN_CLUSTER_ANGULAR_WIDTH=0.02 rad needs >=3 beam gaps
     (4 points) to pass, so any target reduced to 2-3 returns is
     silently dropped no matter how real it is. Note this also makes
     MIN_CLUSTER_POINTS=2 dead code - the width floor is the binding
     constraint.

  2. ORIENTATION. The simulated RPLidar raycasts the VISUAL MESH
     (standing.dae, two legs), not the SDF collision cylinder. Apparent
     leg separation scales with the sine of the angle between the
     leg-separation axis and the line of sight, so the same person can
     present ONE merged cluster or TWO resolved ones depending purely
     on facing - splitting/merging mid-walk, spawning and orphaning
     tracks.

This script measures both axes directly. It applies NO width or
point-count filtering - it reports what is actually there, unfiltered,
so the thresholds can be read off the numbers.

METHOD: park person_1 at each (y, yaw) condition via gz set_pose, let
it settle, capture N scans, break-point cluster each scan with the SAME
jump threshold leg_detector_node uses (so results transfer directly),
transform centroids to map, and keep only clusters near the known
ground-truth position (rejecting walls and person_2 without relying on
hand-computed bearings, which have already produced two wrong
predictions this session).

The y positions trace the actual occlusion-test path (x=3, y=-0.5 ->
-4.0). Both orientations are swept at every position:
  - yaw = pi  : original SDF facing (toward person_2, +Y)
  - yaw = 0   : facing -Y, i.e. the travel direction during walk-out,
                which is what the fixed mover actually commands
"""

import math
import subprocess
import time

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

import tf2_ros
import tf2_geometry_msgs  # noqa: F401  (registers PointStamped support)

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PointStamped


WORLD_NAME = "conversation_test"
MODEL_NAME = "person_1"

PERSON_X = 3.0

# Sweep positions along the occlusion-test path, and both facings.
SWEEP_Y = [-0.5, -1.5, -2.5, -3.5, -4.0]
SWEEP_YAW = [
    (math.pi, "facing +Y (original SDF)"),
    (0.0, "facing -Y (walk-out travel dir)"),
]

SETTLE_SECONDS = 2.0   # wall-clock; let the pose apply and scans refresh
SCANS_PER_CONDITION = 5

# Must match leg_detector_node.py so measurements transfer directly.
CLUSTER_JUMP_THRESHOLD = 0.5

# Deliberately permissive - we want to SEE small/far clusters, not
# silently drop the very ones under investigation.
MIN_VALID_RANGE = 0.164
MAX_VALID_RANGE = 12.0

# Radius around ground truth within which a cluster is attributed to
# person_1. Generous enough to catch both legs when they resolve
# separately, tight enough to exclude person_2 (>=1.0m away) and walls.
ATTRIBUTION_RADIUS = 0.8


def set_pose(x, y, z, yaw):
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)
    req = (
        f"name: '{MODEL_NAME}', "
        f"position: {{x: {x}, y: {y}, z: {z}}}, "
        f"orientation: {{x: 0, y: 0, z: {qz}, w: {qw}}}"
    )
    cmd = [
        "gz", "service",
        "-s", f"/world/{WORLD_NAME}/set_pose",
        "--reqtype", "gz.msgs.Pose",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", "3000",
        "--req", req,
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=5.0
    )
    if result.returncode != 0:
        print(f"  WARN set_pose failed: {result.stderr.strip()}")


class ClusterSweep(Node):
    def __init__(self):
        super().__init__("cluster_sweep")
        self.latest_scan = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.create_subscription(LaserScan, "/scan", self.scan_cb, 10)

    def scan_cb(self, msg):
        self.latest_scan = msg

    # --- same break-point clustering as leg_detector_node -------------
    def cluster_scan(self, msg):
        clusters = []
        current = []
        prev_range = None

        def flush():
            if current:
                clusters.append(list(current))
                current.clear()

        for i in range(len(msg.ranges)):
            r = msg.ranges[i]
            if not math.isfinite(r) or not (MIN_VALID_RANGE <= r <= MAX_VALID_RANGE):
                flush()
                prev_range = None
                continue
            if prev_range is not None and abs(r - prev_range) > CLUSTER_JUMP_THRESHOLD:
                flush()
            current.append((i, r))
            prev_range = r

        flush()
        return clusters

    def cluster_stats(self, msg, cluster):
        n = len(cluster)
        idx_span = cluster[-1][0] - cluster[0][0]
        angular_width = idx_span * msg.angle_increment

        xs, ys = [], []
        for idx, r in cluster:
            ang = msg.angle_min + idx * msg.angle_increment
            xs.append(r * math.cos(ang))
            ys.append(r * math.sin(ang))

        cx = sum(xs) / n
        cy = sum(ys) / n
        mean_range = sum(r for _, r in cluster) / n
        metric_width = angular_width * mean_range

        return {
            "n_points": n,
            "angular_width": angular_width,
            "metric_width": metric_width,
            "mean_range": mean_range,
            "lidar_xy": (cx, cy),
        }

    def to_map(self, cx, cy, frame_id, stamp):
        pt = PointStamped()
        pt.header.frame_id = frame_id
        pt.point.x = cx
        pt.point.y = cy
        pt.point.z = 0.0
        for candidate_stamp in (stamp, rclpy.time.Time().to_msg()):
            pt.header.stamp = candidate_stamp
            try:
                out = self.tf_buffer.transform(
                    pt, "map", timeout=Duration(seconds=0.1)
                )
                return out.point.x, out.point.y
            except Exception:
                continue
        return None

    def capture(self, truth_x, truth_y):
        """Capture SCANS_PER_CONDITION scans, return per-scan lists of
        clusters attributed to the person."""
        results = []
        seen = 0
        deadline = time.time() + 15.0
        last_stamp = None

        while seen < SCANS_PER_CONDITION and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.5)
            msg = self.latest_scan
            if msg is None:
                continue
            stamp_key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
            if stamp_key == last_stamp:
                continue
            last_stamp = stamp_key
            seen += 1

            attributed = []
            for cl in self.cluster_scan(msg):
                st = self.cluster_stats(msg, cl)
                mapped = self.to_map(
                    st["lidar_xy"][0], st["lidar_xy"][1],
                    msg.header.frame_id, msg.header.stamp
                )
                if mapped is None:
                    continue
                d = math.hypot(mapped[0] - truth_x, mapped[1] - truth_y)
                if d <= ATTRIBUTION_RADIUS:
                    st["map_xy"] = mapped
                    st["offset"] = d
                    attributed.append(st)

            results.append(attributed)

        return results


def main():
    rclpy.init()
    node = ClusterSweep()

    print("=" * 72)
    print("LIDAR CLUSTER SWEEP - unfiltered cluster properties vs range/facing")
    print("=" * 72)
    print(f"Jump threshold: {CLUSTER_JUMP_THRESHOLD} m (matches leg_detector_node)")
    print(f"Attribution radius: {ATTRIBUTION_RADIUS} m around ground truth")
    print(f"Scans per condition: {SCANS_PER_CONDITION}")

    rows = []

    for yaw, yaw_label in SWEEP_YAW:
        for y in SWEEP_Y:
            print("\n" + "-" * 72)
            print(f"CONDITION: y={y:+.1f}  |  {yaw_label}")
            set_pose(PERSON_X, y, 0.0, yaw)
            time.sleep(SETTLE_SECONDS)

            scans = node.capture(PERSON_X, y)

            counts = [len(s) for s in scans]
            flat = [c for s in scans for c in s]

            if not flat:
                print("  NO CLUSTERS ATTRIBUTED - person invisible to lidar here")
                rows.append((y, yaw_label, 0, None, None, None, None))
                continue

            n_pts = [c["n_points"] for c in flat]
            widths = [c["angular_width"] for c in flat]
            metric = [c["metric_width"] for c in flat]
            rng = [c["mean_range"] for c in flat]
            offs = [c["offset"] for c in flat]

            print(f"  clusters per scan: {counts}")
            print(f"  points/cluster : min={min(n_pts)} max={max(n_pts)} "
                  f"mean={sum(n_pts)/len(n_pts):.1f}")
            print(f"  angular width  : min={min(widths):.4f} max={max(widths):.4f} "
                  f"mean={sum(widths)/len(widths):.4f} rad")
            print(f"  metric width   : min={min(metric):.3f} max={max(metric):.3f} m")
            print(f"  range          : mean={sum(rng)/len(rng):.2f} m")
            print(f"  centroid offset: mean={sum(offs)/len(offs):.3f} m from truth")

            rows.append((
                y, yaw_label,
                sum(counts) / len(counts),
                min(n_pts),
                min(widths),
                sum(rng) / len(rng),
                sum(offs) / len(offs),
            ))

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'y':>6} {'facing':<28} {'clus/scan':>9} {'min_pts':>8} "
          f"{'min_width':>10} {'range':>7} {'offset':>7}")
    for y, lbl, cps, mp, mw, rg, off in rows:
        if mp is None:
            print(f"{y:>6.1f} {lbl:<28} {'0':>9} {'-':>8} {'-':>10} "
                  f"{'-':>7} {'-':>7}")
        else:
            print(f"{y:>6.1f} {lbl:<28} {cps:>9.1f} {mp:>8d} "
                  f"{mw:>10.4f} {rg:>7.2f} {off:>7.3f}")

    print("\nRead thresholds off this table:")
    print("  - MIN_CLUSTER_POINTS      <= smallest min_pts you need to keep")
    print("  - MIN_CLUSTER_ANGULAR_WIDTH <= smallest min_width you need to keep")
    print("  - clus/scan ~2 indicates legs resolving separately at that condition")

    print(f"\nRestoring {MODEL_NAME} to (3, -0.5) yaw=pi ...")
    set_pose(PERSON_X, -0.5, 0.0, math.pi)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
