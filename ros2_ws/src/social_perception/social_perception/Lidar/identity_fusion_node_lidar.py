#!/usr/bin/env python3
"""
identity_fusion_node.py - bearing-based camera/LIDAR fusion, computed
locally (matches real-robot architecture). See lidar_person_detector.py
and yolo_detector_lidar.py - neither publishes a bearing anymore; both
bearings are derived here instead.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener


ANGLE_GATE_DEG = 12.0          # PROVISIONAL - calibrate from 'pair-bearing' logs
CAMERA_HFOV_DEG = 69.0         # unverified assumption, see yolo_detector_lidar.py history
CAMERA_PREVIEW_WIDTH = 640   # VERIFY against sim's actual image width before trusting

MAP_FRAME = "map"
BASE_FRAME = "base_link"

CAMERA_TRACK_TIMEOUT = 2.0
CAMERA_ACTIVE_WINDOW = 1.0
LIDAR_BINDING_TIMEOUT = 300.0
LIDAR_TRACK_TIMEOUT = 2.0
LIDAR_ONLY_PUBLISH_PERIOD = 0.1


class IdentityFusionNode(Node):
    def __init__(self):
        super().__init__("identity_fusion_node")

        self.declare_parameter("camera_topic", "/person_positions_map")
        self.declare_parameter("lidar_topic", "/lidar_person_clusters")
        self.declare_parameter("output_topic", "/person_positions_fused")
        self.declare_parameter("angle_gate_deg", ANGLE_GATE_DEG)
        self.declare_parameter("camera_hfov_deg", CAMERA_HFOV_DEG)
        self.declare_parameter("camera_preview_width", CAMERA_PREVIEW_WIDTH)

        self.camera_topic = self.get_parameter("camera_topic").value
        self.lidar_topic = self.get_parameter("lidar_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.angle_gate_rad = math.radians(float(self.get_parameter("angle_gate_deg").value))
        self.hfov_rad = math.radians(float(self.get_parameter("camera_hfov_deg").value))
        self.preview_width = float(self.get_parameter("camera_preview_width").value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.stable_of_camera = {}
        self.stable_of_lidar = {}
        self.lidar_tracks = {}
        self.next_stable_id = 0
        self.reid_count = 0
        self.last_published = {}

        self.create_subscription(String, self.lidar_topic, self.lidar_callback, 10)
        self.create_subscription(String, self.camera_topic, self.camera_callback, 10)
        self.pub = self.create_publisher(String, self.output_topic, 10)

        self.create_timer(1.0, self.prune)
        self.create_timer(LIDAR_ONLY_PUBLISH_PERIOD, self.publish_lidar_only)

        self.get_logger().info("Identity fusion node started (local bearing computation)")
        self.get_logger().info(f"Camera in : {self.camera_topic}")
        self.get_logger().info(f"Lidar  in : {self.lidar_topic}")
        self.get_logger().info(f"Fused out: {self.output_topic}")
        self.get_logger().info(
            f"Camera-lidar angle gate: {math.degrees(self.angle_gate_rad):.1f} deg "
            f"(PROVISIONAL - calibrate from 'pair-bearing' log lines)")

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def lidar_callback(self, msg):
        parts = msg.data.split(",")
        if len(parts) < 4:
            return
        try:
            lid = int(float(parts[0]))
            x = float(parts[2])
            y = float(parts[3])
        except ValueError:
            return
        self.lidar_tracks[lid] = (x, y, self.now())

    def robot_pose(self):
        """(x, y, yaw) of base_link in map, or None if TF isn't ready.

        THESIS FIX (vs. real-robot original): added a real timeout instead
        of an instant-fail lookup. Without this, any single missed TF
        cycle silently drops that camera detection entirely - see
        conversation notes on this exact bug in the real-robot version.
        """
        try:
            tf = self.tf_buffer.lookup_transform(
                MAP_FRAME, BASE_FRAME, Time(), timeout=Duration(seconds=0.05))
        except Exception:
            return None

        rx = tf.transform.translation.x
        ry = tf.transform.translation.y
        q = tf.transform.rotation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return rx, ry, yaw

    @staticmethod
    def wrap_angle(a):
        while a > math.pi:
            a -= 2 * math.pi
        while a < -math.pi:
            a += 2 * math.pi
        return a

    def pixel_center_to_bearing(self, x1, x2):
        u = (x1 + x2) / 2.0
        normalized = (u - self.preview_width / 2.0) / (self.preview_width / 2.0)
        return -normalized * (self.hfov_rad / 2.0)

    def nearest_lidar(self, cam_bearing_rel):
        pose = self.robot_pose()
        if pose is None:
            return None, None, None
        rx, ry, ryaw = pose

        best_id, best_diff = None, float('inf')
        for lid, (lx, ly, _) in self.lidar_tracks.items():
            bearing_to_track = math.atan2(ly - ry, lx - rx)
            track_bearing_rel = self.wrap_angle(bearing_to_track - ryaw)
            diff = abs(self.wrap_angle(track_bearing_rel - cam_bearing_rel))
            if diff < best_diff:
                best_id, best_diff = lid, diff

        matched_id = best_id if best_id is not None and best_diff < self.angle_gate_rad else None
        return matched_id, best_id, best_diff

    def stable_id_in_use(self, stable_id, exclude_cam_id, now):
        for cam_id, (sid, last) in self.stable_of_camera.items():
            if cam_id == exclude_cam_id:
                continue
            if sid == stable_id and (now - last) < CAMERA_ACTIVE_WINDOW:
                return True
        return False

    def allocate(self):
        sid = self.next_stable_id
        self.next_stable_id += 1
        return sid

    def camera_callback(self, msg):
        parts = msg.data.split(",")
        if len(parts) < 6:
            self.get_logger().warn(f"Invalid camera msg: {msg.data}")
            return

        try:
            cam_id = int(float(parts[0]))
            conf = parts[1]
            x1 = float(parts[2])
            y1 = float(parts[3])
            x2 = float(parts[4])
            y2 = float(parts[5])
        except ValueError:
            self.get_logger().warn(f"Parse failed: {msg.data}")
            return

        now = self.now()
        cam_bearing = self.pixel_center_to_bearing(x1, x2)
        lid, nearest_id, diff = self.nearest_lidar(cam_bearing)

        if nearest_id is not None:
            status = "MATCH" if lid is not None else "no match (outside gate)"
            self.get_logger().info(
                f"pair-bearing cam:{cam_id} <-> lidar:{nearest_id} = "
                f"{math.degrees(diff):.1f} deg [{status}]")

        known = self.stable_of_camera.get(cam_id)

        if known is not None:
            stable_id = known[0]
        else:
            stable_id = None
            if lid is not None and lid in self.stable_of_lidar:
                candidate = self.stable_of_lidar[lid][0]
                if not self.stable_id_in_use(candidate, cam_id, now):
                    stable_id = candidate
                    self.reid_count += 1
                    self.get_logger().info(
                        f"RE-ID #{self.reid_count}: new camera track "
                        f"{cam_id} adopted stable id {stable_id} via "
                        f"lidar track {lid} (bearing diff "
                        f"{math.degrees(diff):.1f} deg)")
                else:
                    self.get_logger().info(
                        f"Lidar {lid} suggests stable {candidate} for "
                        f"camera {cam_id}, but that identity is already "
                        f"active on another camera track - allocating new")

            if stable_id is None:
                stable_id = self.allocate()
                self.get_logger().info(
                    f"New identity: camera track {cam_id} -> stable "
                    f"{stable_id}"
                    + (f" (anchored on lidar {lid})" if lid is not None
                       else " (no lidar anchor in range)"))

        self.stable_of_camera[cam_id] = [stable_id, now]
        if lid is not None:
            self.stable_of_lidar[lid] = [stable_id, now]

        if lid is None:
            return

        lx, ly, _ = self.lidar_tracks[lid]
        fields = [
            str(stable_id), conf, f"{lx:.3f}", f"{ly:.3f}",
            "0.0", "0", "0",
            f"{int(x1)}", f"{int(y1)}", f"{int(x2)}", f"{int(y2)}",
            "camera_confirmed",
        ]
        out = String()
        out.data = ",".join(fields)
        self.pub.publish(out)
        self.last_published[stable_id] = now

    def publish_lidar_only(self):
        now = self.now()
        active_stable_ids = {
            sid for sid, last in self.stable_of_camera.values()
            if (now - last) < CAMERA_ACTIVE_WINDOW
        }
        for lid, (stable_id, _confirmed_at) in list(self.stable_of_lidar.items()):
            if stable_id in active_stable_ids:
                continue
            if lid not in self.lidar_tracks:
                continue
            last_pub = self.last_published.get(stable_id, 0.0)
            if (now - last_pub) < LIDAR_ONLY_PUBLISH_PERIOD * 0.5:
                continue
            lx, ly, _ = self.lidar_tracks[lid]
            fields = [
                str(stable_id), "0.00", f"{lx:.3f}", f"{ly:.3f}",
                "0.0", "0", "0", "0", "0", "0", "0", "lidar_only",
            ]
            out = String()
            out.data = ",".join(fields)
            self.pub.publish(out)
            self.last_published[stable_id] = now

    def prune(self):
        now = self.now()
        for cam_id in [c for c, (_, t) in self.stable_of_camera.items()
                       if now - t > CAMERA_TRACK_TIMEOUT]:
            del self.stable_of_camera[cam_id]
        for lid in [l for l, v in self.lidar_tracks.items()
                    if now - v[2] > LIDAR_TRACK_TIMEOUT]:
            del self.lidar_tracks[lid]
        for lid in [l for l, (_, t) in self.stable_of_lidar.items()
                    if now - t > LIDAR_BINDING_TIMEOUT or l not in self.lidar_tracks]:
            sid = self.stable_of_lidar[lid][0]
            del self.stable_of_lidar[lid]
            self.get_logger().info(f"Dropped lidar binding {lid} -> stable {sid}")


def main(args=None):
    rclpy.init(args=args)
    node = IdentityFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()