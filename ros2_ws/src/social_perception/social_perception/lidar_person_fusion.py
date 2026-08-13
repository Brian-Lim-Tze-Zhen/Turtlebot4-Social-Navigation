#!/usr/bin/env python3
"""
lidar_person_fusion.py

Bridges the camera's terminal-approach blind spot using the 360 deg LIDAR.

-----------------------------------------------------------------------
WHY
-----------------------------------------------------------------------
Measured across eight ablations, min_distance stayed locked in the
0.47-0.58 m band no matter what changed in the cost function:

    baseline (ObstaclesCritic only)  0.582 +/- 0.025
    local inflation 1.3              0.543 +/- 0.004
    repulsion_weight 5.0             0.523 +/- 0.115
    person disk radius 0.55          0.494 +/- 0.070
    SocialCritic w12 coast6.0        0.510 +/- 0.038
    social_distance 1.40             0.471
    global cost_scaling_factor 1.0   0.503

Lateral deviation rose from 0.85 to 1.14 and path ratio from 1.13 to
1.22 across that set - the robot was made to yield progressively more,
and it did, but the clearance never moved. The binding constraint is
not how hard the planner is pushed; it is WHEN it learns to push.

The OAK-D has ~72 deg HFOV (1.25 rad in the stock xacro; ~78 deg on
real hardware). In a head-on pass the person exits frame laterally at
about 3.5 m, and the closest approach happens roughly 2.4 s later at
1.46 m/s closing speed. Every controller decision in that window runs
on extrapolated state.

The RPLidar has no such blind spot. A separate experiment already
confirmed the robot avoids a stationary person on LIDAR alone. This
node therefore keeps a LIDAR lock on the person that the camera
established, and keeps publishing real observations through the
interval where the camera has nothing.

-----------------------------------------------------------------------
HOW
-----------------------------------------------------------------------
    /predicted_person_positions  (YOLO -> ByteTrack -> KF)
                |
                |  seeds and re-seeds the association
                v
    /scan  -->  gate  -->  cluster  -->  nearest cluster to prediction
                |
                v
    /fused_person_positions   (identical field layout)

Downstream consumers need no code change:

    SocialCritic:
      topic: /fused_person_positions      # was /predicted_person_positions

    predicted_person_cloud_node.py:
      change the subscription topic, or remap on launch.

While the camera track is fresh the node republishes it verbatim, so
behaviour in that regime is bit-identical to the camera-only pipeline -
which keeps the ablation honest. Only once the camera goes silent does
the LIDAR estimate take over.

-----------------------------------------------------------------------
WHAT KEEPS IT FROM LOCKING ONTO A WALL
-----------------------------------------------------------------------
Three independent gates, all of which must pass:

  1. Position gate - the cluster centroid must lie within assoc_radius
     of where constant-velocity motion says the person should be. The
     gate travels with the prediction, so it does not need to be wide.

  2. Size gate - a standing person subtends roughly 0.2-0.7 m of chord.
     Walls produce clusters metres long; furniture legs produce clusters
     centimetres long.

  3. Static-map gate - a cluster that sits on a wall in the static map
     is rejected. Without a map this degrades gracefully to gates 1-2.

Run with:
    python3 lidar_person_fusion.py --ros-args -p use_sim_time:=true
"""

import math

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

import tf2_ros
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


class LidarPersonFusion(Node):
    def __init__(self):
        super().__init__("lidar_person_fusion")

        # --- parameters ---------------------------------------------
        self.declare_parameter("input_topic", "/predicted_person_positions")
        self.declare_parameter("output_topic", "/fused_person_positions")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("publish_rate", 10.0)

        # Camera track older than this is considered lost, and the LIDAR
        # estimate takes over. Matches SocialCritic's track_timeout so the
        # handover and the critic's own freshness test agree.
        self.declare_parameter("camera_timeout", 0.5)

        # Give up entirely this long after the last camera sighting. The
        # LIDAR lock is only trustworthy while it can be traced back to a
        # camera-confirmed identity; past this it is just "some object".
        self.declare_parameter("max_lidar_only_time", 8.0)

        # Association gate. 0.6 m at 10 Hz tolerates a 6 m/s association
        # error, far above the 1.2 m/s the person actually walks, while
        # staying tight enough to reject a wall half a metre behind them.
        self.declare_parameter("assoc_radius", 0.6)

        # Cluster geometry gates.
        self.declare_parameter("cluster_gap", 0.15)      # range jump splitting clusters
        self.declare_parameter("min_cluster_width", 0.08)
        self.declare_parameter("max_cluster_width", 0.80)
        self.declare_parameter("min_cluster_points", 2)

        # Static-map rejection.
        self.declare_parameter("use_static_map", True)
        self.declare_parameter("map_clearance", 0.25)    # m from an occupied cell

        # Velocity smoothing on the LIDAR-derived track.
        self.declare_parameter("velocity_alpha", 0.35)
        self.declare_parameter("prediction_horizon", 1.0)  # matches the KF's 1 s

        self.declare_parameter("publish_markers", True)

        g = self.get_parameter
        self.input_topic = g("input_topic").value
        self.output_topic = g("output_topic").value
        self.scan_topic = g("scan_topic").value
        self.map_frame = g("map_frame").value
        self.publish_rate = g("publish_rate").value
        self.camera_timeout = g("camera_timeout").value
        self.max_lidar_only_time = g("max_lidar_only_time").value
        self.assoc_radius = g("assoc_radius").value
        self.cluster_gap = g("cluster_gap").value
        self.min_cluster_width = g("min_cluster_width").value
        self.max_cluster_width = g("max_cluster_width").value
        self.min_cluster_points = g("min_cluster_points").value
        self.use_static_map = g("use_static_map").value
        self.map_clearance = g("map_clearance").value
        self.velocity_alpha = g("velocity_alpha").value
        self.prediction_horizon = g("prediction_horizon").value
        self.publish_markers = g("publish_markers").value

        # --- state --------------------------------------------------
        # Last full camera message, split into fields. Fields this node
        # does not understand (indices 1 and 8) are carried through
        # unchanged rather than invented, so the wire format stays
        # whatever human_kf_predictor decided it is.
        self.last_camera_parts = None
        self.last_camera_time = None
        self.last_camera_track_id = None

        # LIDAR-maintained estimate, map frame.
        self.lidar_xy = None
        self.lidar_vx = 0.0
        self.lidar_vy = 0.0
        self.lidar_time = None

        self.latest_scan = None
        self.map_msg = None

        # --- ROS wiring ---------------------------------------------
        self.sub_camera = self.create_subscription(
            String, self.input_topic, self.camera_callback, 10)
        self.sub_scan = self.create_subscription(
            LaserScan, self.scan_topic, self.scan_callback,
            rclpy.qos.qos_profile_sensor_data)

        self.pub = self.create_publisher(String, self.output_topic, 10)

        if self.use_static_map:
            map_qos = QoSProfile(
                depth=1,
                reliability=QoSReliabilityPolicy.RELIABLE,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                history=QoSHistoryPolicy.KEEP_LAST,
            )
            self.sub_map = self.create_subscription(
                OccupancyGrid, "/map", self.map_callback, map_qos)

        self.marker_pub = (
            self.create_publisher(MarkerArray, "/lidar_person_markers", 10)
            if self.publish_markers else None)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_fused)

        # Counters for the end-of-run summary — the number that matters
        # is how many cycles were served by LIDAR that the camera-only
        # pipeline would have had to extrapolate through.
        self.n_camera = 0
        self.n_lidar = 0
        self.n_none = 0

        self.get_logger().info("lidar_person_fusion started")
        self.get_logger().info(f"  in : {self.input_topic}")
        self.get_logger().info(f"  out: {self.output_topic}")
        self.get_logger().info(
            f"  camera_timeout={self.camera_timeout:.2f}s  "
            f"assoc_radius={self.assoc_radius:.2f}m  "
            f"cluster width {self.min_cluster_width:.2f}-{self.max_cluster_width:.2f}m")

    # ----------------------------------------------------------------
    def now_sec(self):
        return self.get_clock().now().nanoseconds / 1e9

    def camera_callback(self, msg):
        parts = msg.data.split(",")
        if len(parts) < 9:
            return
        try:
            track_id = int(float(parts[0]))
            cx = float(parts[2])
            cy = float(parts[3])
            vx = float(parts[4])
            vy = float(parts[5])
        except ValueError:
            return

        # track_id -1 means ByteTrack could not associate this detection.
        # Its position is still usable, but it must not reset the
        # identity the LIDAR lock is anchored to.
        self.last_camera_parts = parts
        self.last_camera_time = self.now_sec()
        if track_id >= 0:
            self.last_camera_track_id = track_id

        # Re-seed the LIDAR estimate from every fresh camera fix. This
        # keeps the association warm so the handover at loss-of-track is
        # continuous rather than a cold start.
        self.lidar_xy = (cx, cy)
        self.lidar_vx = vx
        self.lidar_vy = vy
        self.lidar_time = self.last_camera_time

    def scan_callback(self, msg):
        self.latest_scan = msg

    def map_callback(self, msg):
        self.map_msg = msg
        self.get_logger().info(
            f"Static map received: {msg.info.width}x{msg.info.height} "
            f"@ {msg.info.resolution:.3f} m/cell")

    # ----------------------------------------------------------------
    def scan_to_map_points(self, scan):
        """Return scan returns as an (N, 2) array of map-frame points.

        Invalid, out-of-range and non-finite returns are dropped here so
        every downstream stage can assume clean data.
        """
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, scan.header.frame_id,
                rclpy.time.Time(), timeout=Duration(seconds=0.05))
        except Exception:
            return None, None

        ranges = np.asarray(scan.ranges, dtype=np.float64)
        n = ranges.size
        if n == 0:
            return None, None

        angles = scan.angle_min + np.arange(n) * scan.angle_increment

        valid = np.isfinite(ranges) & (ranges > scan.range_min) & (ranges < scan.range_max)
        if not np.any(valid):
            return None, None

        idx = np.nonzero(valid)[0]
        r = ranges[idx]
        a = angles[idx]

        # Sensor frame, then a planar rigid transform into map. The LIDAR
        # is mounted level, so the 2D reduction is exact.
        sx = r * np.cos(a)
        sy = r * np.sin(a)

        t = tf.transform.translation
        q = tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        c, s = math.cos(yaw), math.sin(yaw)

        mx = t.x + c * sx - s * sy
        my = t.y + s * sx + c * sy

        return np.column_stack((mx, my)), (r, idx)

    def cluster_points(self, pts, r_and_idx):
        """Split angularly-ordered returns into clusters.

        Adjacent returns belong to the same object when both the range
        step and the index step are small. The index check matters at the
        scan wrap-around and across dropped invalid returns, where two
        angularly distant points can share a similar range.
        """
        r, idx = r_and_idx
        clusters = []
        start = 0
        for i in range(1, len(r)):
            range_jump = abs(r[i] - r[i - 1]) > self.cluster_gap
            index_jump = (idx[i] - idx[i - 1]) > 2
            if range_jump or index_jump:
                if i - start >= self.min_cluster_points:
                    clusters.append(pts[start:i])
                start = i
        if len(r) - start >= self.min_cluster_points:
            clusters.append(pts[start:])
        return clusters

    def on_static_obstacle(self, x, y):
        """True if (x, y) sits within map_clearance of an occupied cell."""
        if self.map_msg is None:
            return False

        info = self.map_msg.info
        res = info.resolution
        ox = info.origin.position.x
        oy = info.origin.position.y

        pad = max(0, int(math.ceil(self.map_clearance / res)))
        cx = int((x - ox) / res)
        cy = int((y - oy) / res)

        data = self.map_msg.data
        for gy in range(cy - pad, cy + pad + 1):
            if gy < 0 or gy >= info.height:
                continue
            row = gy * info.width
            for gx in range(cx - pad, cx + pad + 1):
                if gx < 0 or gx >= info.width:
                    continue
                if data[row + gx] > 50:
                    return True
        return False

    def find_person_cluster(self, expected_xy):
        """Nearest cluster to expected_xy that passes every gate."""
        if self.latest_scan is None:
            return None

        pts, r_and_idx = self.scan_to_map_points(self.latest_scan)
        if pts is None or len(pts) < self.min_cluster_points:
            return None

        best = None
        best_d = self.assoc_radius

        for cl in self.cluster_points(pts, r_and_idx):
            centroid = cl.mean(axis=0)
            d = math.hypot(centroid[0] - expected_xy[0], centroid[1] - expected_xy[1])
            if d >= best_d:
                continue

            # Size gate — chord length across the cluster.
            width = float(np.linalg.norm(cl[-1] - cl[0]))
            if width < self.min_cluster_width or width > self.max_cluster_width:
                continue

            # Static-map gate.
            if self.use_static_map and self.on_static_obstacle(centroid[0], centroid[1]):
                continue

            best = centroid
            best_d = d

        return best

    # ----------------------------------------------------------------
    def publish_fused(self):
        if self.last_camera_parts is None:
            self.n_none += 1
            return

        now = self.now_sec()
        cam_age = now - self.last_camera_time

        if cam_age > self.max_lidar_only_time:
            # Too long since any camera confirmation. Publishing here
            # would assert an identity nothing has verified in eight
            # seconds, so stop instead and let downstream time out.
            self.n_none += 1
            return

        if cam_age <= self.camera_timeout:
            # Camera is live. Republish verbatim so this regime is
            # byte-identical to the camera-only pipeline.
            out = String()
            out.data = ",".join(self.last_camera_parts)
            self.pub.publish(out)
            self.n_camera += 1
            self.publish_marker(
                (float(self.last_camera_parts[2]), float(self.last_camera_parts[3])),
                source="camera")
            return

        # --- camera lost: LIDAR takes over --------------------------
        if self.lidar_xy is None or self.lidar_time is None:
            self.n_none += 1
            return

        dt = now - self.lidar_time
        if dt <= 1e-3:
            return

        expected = (self.lidar_xy[0] + self.lidar_vx * dt,
                    self.lidar_xy[1] + self.lidar_vy * dt)

        found = self.find_person_cluster(expected)

        if found is None:
            # No cluster passed the gates. Fall back to the constant
            # velocity estimate for this cycle and try again next tick;
            # a single dropout is common when the legs cross and the two
            # clusters merge.
            self.n_none += 1
            return

        vx_raw = (found[0] - self.lidar_xy[0]) / dt
        vy_raw = (found[1] - self.lidar_xy[1]) / dt
        a = self.velocity_alpha
        self.lidar_vx = a * vx_raw + (1.0 - a) * self.lidar_vx
        self.lidar_vy = a * vy_raw + (1.0 - a) * self.lidar_vy
        self.lidar_xy = (float(found[0]), float(found[1]))
        self.lidar_time = now

        parts = list(self.last_camera_parts)
        parts[2] = f"{self.lidar_xy[0]:.4f}"
        parts[3] = f"{self.lidar_xy[1]:.4f}"
        parts[4] = f"{self.lidar_vx:.4f}"
        parts[5] = f"{self.lidar_vy:.4f}"
        parts[6] = f"{self.lidar_xy[0] + self.lidar_vx * self.prediction_horizon:.4f}"
        parts[7] = f"{self.lidar_xy[1] + self.lidar_vy * self.prediction_horizon:.4f}"

        out = String()
        out.data = ",".join(parts)
        self.pub.publish(out)
        self.n_lidar += 1

        self.publish_marker(self.lidar_xy, source="lidar")

        self.get_logger().info(
            f"LIDAR lock t={now:.2f} pos=({self.lidar_xy[0]:.2f},{self.lidar_xy[1]:.2f}) "
            f"v=({self.lidar_vx:.2f},{self.lidar_vy:.2f}) cam_age={cam_age:.2f}s",
            throttle_duration_sec=1.0)

    def publish_marker(self, xy, source):
        if self.marker_pub is None:
            return
        m = Marker()
        m.header.frame_id = self.map_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "fused_person"
        m.id = 0
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.pose.position.x = float(xy[0])
        m.pose.position.y = float(xy[1])
        m.pose.position.z = 0.5
        m.pose.orientation.w = 1.0
        m.scale.x = 0.4
        m.scale.y = 0.4
        m.scale.z = 1.0
        # Green while the camera is driving, magenta once the LIDAR is.
        m.color.r = 0.0 if source == "camera" else 1.0
        m.color.g = 1.0 if source == "camera" else 0.0
        m.color.b = 0.0 if source == "camera" else 1.0
        m.color.a = 0.5
        m.lifetime = Duration(seconds=0.5).to_msg()
        arr = MarkerArray()
        arr.markers.append(m)
        self.marker_pub.publish(arr)

    def destroy_node(self):
        total = self.n_camera + self.n_lidar + self.n_none
        if total:
            self.get_logger().info(
                f"Fusion summary: camera {self.n_camera} "
                f"({100.0 * self.n_camera / total:.1f}%), "
                f"LIDAR {self.n_lidar} ({100.0 * self.n_lidar / total:.1f}%), "
                f"none {self.n_none} ({100.0 * self.n_none / total:.1f}%)")
            self.get_logger().info(
                f"The LIDAR share is the fraction of control cycles that the "
                f"camera-only pipeline had to extrapolate through.")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LidarPersonFusion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
