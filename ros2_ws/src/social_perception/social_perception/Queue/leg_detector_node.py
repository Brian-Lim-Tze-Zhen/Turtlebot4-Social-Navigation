#!/usr/bin/env python3
"""
leg_detector_node.py

THESIS ADDITION — lidar-based person-cylinder cluster detector.

-----------------------------------------------------------------------
WHY THIS EXISTS
-----------------------------------------------------------------------
Camera-based tracking (yolo_detector.py -> human_kf_predictor.py) loses
identity across ~2s+ occlusions: ByteTrack assigns a new track_id, which
resets the KF (human_kf_predictor.py) and the close_since timers
(group_formation_detector.py). This node provides a continuous,
occlusion-independent position anchor via the 2D lidar, so a later
fusion step can re-associate a new camera track_id with an old lost one
by proximity to a continuously-tracked lidar cluster.

This node does NOT do that re-association itself yet - it only detects
and tracks lidar clusters continuously. Camera-to-lidar association is
the next piece, built on top of this node's output.

-----------------------------------------------------------------------
GEOMETRY, CONFIRMED EMPIRICALLY THIS SESSION (conversation_test.sdf)
-----------------------------------------------------------------------
- RPLidar mount: base_link -> rplidar_link translation z=0.193m, zero
  pitch/roll (pure 90deg yaw). Scan plane is horizontal at ankle/shin
  height - within the standard range used in 2D-lidar leg/person
  detection literature.
- person_standing model (simulation_models/person_standing/model.sdf)
  has a SINGLE cylinder collision (radius=0.25m, length=1.7m, spanning
  z=0 to 1.7m) per person - NOT two separate leg colliders. Confirmed
  via direct /scan capture: each person produces exactly ONE contiguous
  cluster (7 beams at ~3.1m range), not two. This means classic "two
  leg clusters paired into one person" literature algorithms (e.g.
  Leigh et al. leg_tracker) do not apply to this synthetic model as-is
  - we detect one cluster per person directly instead.
- At ~3.1m range: cluster width ~7 beams (~0.069 rad angular,
  ~0.20m metric - narrower than the cylinder's geometric 0.5m diameter,
  likely grazing-incidence dropout at the cylinder's tangent edges).
  Background (wall) in the test world reads ~10.5-11.3m, giving a clean
  ~7-8m step at the person's leading edge - trivial break-point signal.

Cluster width/jump thresholds below are seeded from this single-range
data point. NOT yet validated across multiple ranges - re-tune
CLUSTER_JUMP_THRESHOLD and the width filter bounds if detection is
unreliable at ranges much closer/farther than ~3m, and prefer widening
the width filter bounds over narrowing them until more range samples
are collected.

-----------------------------------------------------------------------
ALGORITHM
-----------------------------------------------------------------------
1. Break-point clustering: walk consecutive beams, start a new cluster
   whenever the range jump between adjacent valid beams exceeds
   CLUSTER_JUMP_THRESHOLD. This is the classic, simple approach used
   as the baseline in the 2D-lidar leg/person detection literature
   before any learned classifier is layered on top.
2. Filter clusters by angular width and point count to reject long
   flat surfaces (walls) and single-beam noise spikes.
3. Compute each surviving cluster's centroid in the lidar frame,
   transform to map via TF.
4. Track centroids across scans with simple nearest-neighbour gating
   (constant-position association within MAX_ASSOCIATION_DIST) so a
   cluster keeps a stable internal ID across consecutive scans even
   through brief lidar-side gaps (self-occlusion, momentary dropout).
   This internal ID is NOT the camera's track_id - it's a separate,
   lidar-only identity. Fusing the two is the next piece of work.

-----------------------------------------------------------------------
OUTPUT
-----------------------------------------------------------------------
Publishes /lidar_person_clusters (String), one line per active cluster:

    lidar_id,map_x,map_y,last_seen_age

Intentionally simple CSV, matching the style of the existing pipeline's
topics (/person_positions_map, /predicted_person_positions) so a future
fusion node can parse it the same way.

-----------------------------------------------------------------------
WHAT THIS FILE DOES NOT DO YET
-----------------------------------------------------------------------
- No camera-to-lidar re-association (the actual re-ID fix). This node
  only produces continuous lidar-side identities as an input to that.
- No distinction between "person" clusters and other similarly-sized
  static objects (chair legs, bins, etc.) - there are none in the
  current test worlds, but a real/more cluttered world would need a
  static-vs-dynamic filter (e.g. persistent occupancy check) before
  this is trustworthy as a person detector rather than a "roughly
  cylindrical object ~0.5m wide" detector.
- Width/jump thresholds are single-range-point estimates - see note
  above.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

import tf2_ros
import tf2_geometry_msgs  # noqa: F401  (registers PointStamped transform support)

from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped


# =======================================================================
# Tunable thresholds - seeded from this session's single-range data
# point (~3.1m). Re-validate against closer/farther ranges before
# trusting these broadly - see module docstring.
# =======================================================================

# --- Break-point clustering ---
CLUSTER_JUMP_THRESHOLD = 0.5      # m; range delta between adjacent beams
                                   # that starts a new cluster. Seeded from
                                   # the observed ~7-8m step at the
                                   # person/wall boundary - a real target
                                   # edge will vastly exceed this, while
                                   # noise on a flat wall should not.
MIN_CLUSTER_POINTS = 2            # reject single-beam spikes
MAX_CLUSTER_POINTS = 20           # reject long flat surfaces (walls)

# --- Cluster shape filter (METRIC width, metres) ---
# CALIBRATED from cluster_sweep.py across y=-0.5..-4.0 (range 2.9-4.9m).
#
# The previous ANGULAR width filter was removed: for a contiguous
# cluster, angular_width == (n_points - 1) * angle_increment exactly,
# so it was the point-count filter expressed a second time, and its
# floor of 0.02 rad silently rejected every genuine 3-point cluster
# (0.01967 rad) at range >= ~3.9m. That single off-by-a-hair threshold
# is why the person vanished from the filter during the occlusion
# test's hold phase.
#
# Metric width is the physically meaningful quantity. Measured:
#   merged body (near, shallow bearing) : 0.145 - 0.229 m
#   single leg  (far, oblique bearing)  : 0.048 - 0.118 m
# Bounds below add margin on both sides.
MIN_CLUSTER_METRIC_WIDTH = 0.03   # m
MAX_CLUSTER_METRIC_WIDTH = 0.40   # m

# --- Leg pairing ---
# Whether a person presents ONE cluster or TWO depends on the angle
# between their leg-separation axis and the line of sight: the
# separation projects across the beam as sin(that angle). Measured
# transition on the occlusion path is between y=-1.5 (1 cluster, ~0.23m
# wide) and y=-2.5 (2 clusters, ~0.09m each) - i.e. as the bearing
# swings from ~10deg to ~53deg. A person WALKING therefore splits and
# merges mid-track, which spawns and orphans tracks if association runs
# on raw clusters.
#
# Fix: merge clusters within LEG_PAIR_MAX_DIST into one detection
# BEFORE association, so the tracker sees a stable one-detection-per-
# person stream regardless of presented profile. This is the leg-
# pairing step from the classic 2D-lidar literature, which an earlier
# revision of this file wrongly dismissed as inapplicable - that call
# was based on a single close-range sample where the legs happened to
# be occluding each other.
#
# COUPLING - this value is bounded on BOTH sides and must not be tuned
# freely: it must exceed the leg separation (~0.26m measured) but stay
# well under the smallest person-to-person separation in the scene
# (1.0m in conversation_test.sdf), or two people merge into one
# detection. Re-check this bound before using denser scenes.
LEG_PAIR_MAX_DIST = 0.40   # m

# --- Range validity ---
MIN_VALID_RANGE = 0.164  # matches sensor's own range_min
MAX_VALID_RANGE = 8.0    # people beyond this are unlikely to matter yet;
                          # re-tune if evaluation scenarios need farther
                          # detection range

# --- Cross-scan tracking (nearest-neighbour association) ---
MAX_ASSOCIATION_DIST = 0.5   # m; max centroid movement between scans to
                              # keep the same lidar_id. Deliberately
                              # generous vs. camera-side max_jump (0.8m)
                              # since lidar runs at full scan rate with
                              # no depth-frame-timing wrinkles.
TRACK_TIMEOUT = 1.0          # s; drop a lidar track if unseen this long.
                              # NOTE: deliberately looser than
                              # human_kf_predictor's coast_timeout (1.5s)
                              # is NOT the target here - this timeout is
                              # about the LIDAR losing the cluster
                              # (self-occlusion, momentary dropout), a
                              # different failure mode than the CAMERA
                              # losing the person. Re-tune independently.


class LidarTrack:
    def __init__(self, track_id, x, y, timestamp):
        self.id = track_id
        self.x = x
        self.y = y
        self.last_seen = timestamp


class LegDetectorNode(Node):
    def __init__(self):
        super().__init__("leg_detector_node")

        self.map_frame = "map"

        self.tracks = {}          # lidar_id -> LidarTrack
        self.next_id = 0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.sub = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 10
        )

        self.pub = self.create_publisher(
            String, "/lidar_person_clusters", 10
        )

        self.get_logger().info("Lidar leg/person cluster detector started")
        self.get_logger().info(
            f"Jump threshold: {CLUSTER_JUMP_THRESHOLD:.2f} m | "
            f"Metric width filter: [{MIN_CLUSTER_METRIC_WIDTH:.3f}, "
            f"{MAX_CLUSTER_METRIC_WIDTH:.3f}] m"
        )
        self.get_logger().info(
            f"Leg-pair merge distance: {LEG_PAIR_MAX_DIST:.2f} m"
        )
        self.get_logger().info(
            f"Association dist: {MAX_ASSOCIATION_DIST:.2f} m | "
            f"Track timeout: {TRACK_TIMEOUT:.2f} s"
        )

    def get_ros_time_seconds(self):
        return self.get_clock().now().nanoseconds / 1e9

    # -------------------------------------------------------------
    # Break-point clustering over one LaserScan message
    # -------------------------------------------------------------
    def cluster_scan(self, msg):
        n = len(msg.ranges)
        clusters = []
        current = []

        def flush():
            if current:
                clusters.append(list(current))
                current.clear()

        prev_range = None

        for i in range(n):
            r = msg.ranges[i]

            if not (MIN_VALID_RANGE <= r <= MAX_VALID_RANGE) or not math.isfinite(r):
                flush()
                prev_range = None
                continue

            if prev_range is not None and abs(r - prev_range) > CLUSTER_JUMP_THRESHOLD:
                flush()

            current.append((i, r))
            prev_range = r

        flush()

        # NOTE: does not currently merge a cluster that wraps across the
        # 0/2pi boundary (index 0 and index n-1 being the same physical
        # object). Not an issue for the test worlds used this session
        # (people were well inside the middle of the angular range), but
        # worth fixing if a person can appear near angle_min/angle_max.

        return clusters

    def filter_and_centroid(self, msg, cluster):
        n_points = len(cluster)
        if not (MIN_CLUSTER_POINTS <= n_points <= MAX_CLUSTER_POINTS):
            return None

        angular_width = (cluster[-1][0] - cluster[0][0]) * msg.angle_increment
        mean_range = sum(r for _, r in cluster) / n_points
        metric_width = angular_width * mean_range
        if not (MIN_CLUSTER_METRIC_WIDTH <= metric_width <= MAX_CLUSTER_METRIC_WIDTH):
            return None

        # Centroid in the lidar frame (Cartesian mean of the cluster's points)
        xs, ys = [], []
        for idx, r in cluster:
            angle = msg.angle_min + idx * msg.angle_increment
            xs.append(r * math.cos(angle))
            ys.append(r * math.sin(angle))

        cx = sum(xs) / n_points
        cy = sum(ys) / n_points

        return cx, cy

    def transform_to_map(self, cx, cy, frame_id, stamp):
        # THESIS FIX (TF lag fallback) - same pattern as
        # yolo_detector.py's transform_camera_to_target(). This sim's TF
        # publishing lags the scan message's own header.stamp by
        # ~0.2-0.3s, which exceeds the lookup timeout below and would
        # otherwise cause every single transform to fail with
        # "extrapolation into the future" - confirmed empirically this
        # session, not a hypothetical. Try the accurate message-stamp
        # transform first (best accuracy when TF has caught up); if it
        # fails, retry once against "latest available" rather than
        # dropping the cluster entirely.
        point = PointStamped()
        point.header.frame_id = frame_id
        point.header.stamp = stamp
        point.point.x = cx
        point.point.y = cy
        point.point.z = 0.0

        try:
            transformed = self.tf_buffer.transform(
                point, self.map_frame, timeout=Duration(seconds=0.1)
            )
            return transformed.point.x, transformed.point.y
        except Exception as e_stamped:
            try:
                point.header.stamp = rclpy.time.Time().to_msg()
                transformed = self.tf_buffer.transform(
                    point, self.map_frame, timeout=Duration(seconds=0.1)
                )
                return transformed.point.x, transformed.point.y
            except Exception as e_latest:
                self.get_logger().warn(
                    f"TF transform failed (both stamps): {e_latest}"
                )
                return None

    # -------------------------------------------------------------
    # Leg pairing: merge co-located clusters into one detection
    # -------------------------------------------------------------
    def merge_leg_clusters(self, centroids):
        """Single-linkage merge of centroids within LEG_PAIR_MAX_DIST.

        Single-linkage (chain) rather than pairwise so a person showing
        3 fragments merges as one detection, not one pair plus an
        orphan. See LEG_PAIR_MAX_DIST for the two-sided bound on the
        gate."""
        merged = []
        used = [False] * len(centroids)

        for i in range(len(centroids)):
            if used[i]:
                continue
            group = [centroids[i]]
            used[i] = True

            grew = True
            while grew:
                grew = False
                for j in range(len(centroids)):
                    if used[j]:
                        continue
                    for gx, gy in group:
                        if math.hypot(centroids[j][0] - gx,
                                      centroids[j][1] - gy) <= LEG_PAIR_MAX_DIST:
                            group.append(centroids[j])
                            used[j] = True
                            grew = True
                            break

            mx = sum(p[0] for p in group) / len(group)
            my = sum(p[1] for p in group) / len(group)
            merged.append((mx, my))

        return merged

    # -------------------------------------------------------------
    # Cross-scan nearest-neighbour association
    # -------------------------------------------------------------
    def associate(self, centroids_map, now):
        unmatched_tracks = set(self.tracks.keys())

        for mx, my in centroids_map:
            best_id = None
            best_dist = MAX_ASSOCIATION_DIST

            for tid in unmatched_tracks:
                t = self.tracks[tid]
                d = math.hypot(mx - t.x, my - t.y)
                if d < best_dist:
                    best_dist = d
                    best_id = tid

            if best_id is not None:
                t = self.tracks[best_id]
                t.x, t.y = mx, my
                t.last_seen = now
                unmatched_tracks.discard(best_id)
            else:
                new_id = self.next_id
                self.next_id += 1
                self.tracks[new_id] = LidarTrack(new_id, mx, my, now)

        stale_ids = [
            tid for tid, t in self.tracks.items()
            if now - t.last_seen > TRACK_TIMEOUT
        ]
        for tid in stale_ids:
            del self.tracks[tid]
            self.get_logger().info(f"Pruned stale lidar track id:{tid}")

    def publish_tracks(self, now):
        for tid, t in self.tracks.items():
            out = String()
            age = now - t.last_seen
            out.data = f"{tid},{t.x:.3f},{t.y:.3f},{age:.3f}"
            self.pub.publish(out)

    # -------------------------------------------------------------
    # Main callback
    # -------------------------------------------------------------
    def scan_callback(self, msg):
        now = self.get_ros_time_seconds()

        raw_clusters = self.cluster_scan(msg)

        centroids_map = []
        for cluster in raw_clusters:
            result = self.filter_and_centroid(msg, cluster)
            if result is None:
                continue

            cx, cy = result
            transformed = self.transform_to_map(cx, cy, msg.header.frame_id, msg.header.stamp)
            if transformed is None:
                continue

            centroids_map.append(transformed)

        detections = self.merge_leg_clusters(centroids_map)

        self.associate(detections, now)
        self.publish_tracks(now)

        # Logged unconditionally (the old version logged only when
        # clusters existed, so total-dropout frames left NO trace - the
        # exact failure mode under investigation was invisible in the
        # log).
        self.get_logger().info(
            f"Frame clusters: {len(raw_clusters)} raw -> "
            f"{len(centroids_map)} passed filter -> "
            f"{len(detections)} merged -> "
            f"{len(self.tracks)} active track(s)"
        )


def main(args=None):
    rclpy.init(args=args)
    node = LegDetectorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
