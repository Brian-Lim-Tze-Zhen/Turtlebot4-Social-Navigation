#!/usr/bin/env python3
"""
camera_ray_identity_node.py

THESIS ADDITION - identity layer for the CAMERA-PRIMARY pipeline.
Replaces identity_fusion_node_lidar.py when camera_ray_person_node.py
is the detector.

WHY A NEW NODE (not a patch of identity_fusion_node_lidar.py)
  The old node binds camera tracks to LiDAR tracks. Measured 15 Sep
  (proto_B / proto_C): a walking person burns ~1 LiDAR track id per
  second, so bindings churn, and re-entry after a frame exit finds no
  anchor -> new stable id. Its lidar_only path also created phantoms
  (stable 6 at 8.59 m in proto_C; wall fragments while the robot moves).
  Here LiDAR never creates or anchors a person. The camera decides who
  exists; identity is carried by POSITION continuity in the map frame.

INPUT
  /camera_ray_clusters  "cam_id,conf,map_x,map_y"   (camera_ray_person_node)
  /person_positions_map "track_id,conf,x1,y1,x2,y2,keypoints,img_stamp"
                        (yolo_leg_detector_lidar) - bboxes only, see below
  /scan                 LaserScan, only for short coasting (below)

OUTPUT
  /person_positions_fused  - SAME 13-field format as the old node, so
  human_kf_predictor_lidar.py / person_marker_publisher.py are unchanged:
    stable_id,conf,x,y,depth,u,v,x1,y1,x2,y2,source,cam_id
  source = "camera_confirmed" (camera this cycle) or "lidar_only" (coast).

IDENTITY RULES
  1. Known cam_id -> keep its stable id.
  2. New cam_id (ByteTrack switch / frame re-entry) -> adopt the nearest
     stable id seen within REATTACH_WINDOW_S whose last position is within
     JITTER_M + MAX_SPEED * gap, and which no other live cam_id holds.
  3. Otherwise allocate a new stable id.

COASTING (camera lost the person)
  For up to COAST_S, follow the nearest /scan return within the same
  motion gate of the last position and publish it as lidar_only. After
  COAST_S the id goes silent (the KF coasts its own prediction, then
  prunes). Short on purpose: long coasts are how LiDAR latches onto walls.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PointStamped
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point

MAP_FRAME = "map"
SCAN_FRAME = "rplidar_link"      # scan header frame does not match TF tree

CAM_ID_TIMEOUT_S = 0.35      # cam_id unseen this long -> no longer "live"
REATTACH_WINDOW_S = 8.0      # stable id reclaimable by a new cam_id for this long
JITTER_M = 0.5               # fixed position allowance (ray hop across body)
MAX_REATTACH_M = 2.5         # cap on reattach distance
MAX_SPEED = 1.5              # m/s, walking-pace allowance per second of gap

# THESIS FIX (15 Sep, switch vs second object): a ByteTrack switch can
# arrive only ~0.17 s after the old id's last frame (measured: cam 1
# last 965.973, cam 2 first 966.145), so the old id still looks "live"
# and the switch was refused -> duplicate stable id on one person.
# Time alone cannot separate a switch from a second object. So when the
# nearest candidate is held by a recently-seen cam id, hold the new cam
# id PENDING (unpublished) for PENDING_S: if the holder publishes again
# meanwhile, both are real -> new id; if not, it was a switch -> reattach.
PENDING_S = 0.4

# THESIS FIX (16 Sep, flip-flop vs second object): "holder published
# again during PENDING" was NOT proof of a second object. Measured on
# proto_F (one person, robot turning): ByteTrack alternated between
# ids 2 and 3 in blocks (3,3,3,2x7,3), boxes never in the same frame,
# i.e. it revived a lost track - one person, two ids. Two real objects
# appear in the SAME image frame; yolo_leg_detector publishes all
# detections of one frame within ~1-2 ms (live C: 270.0414 / 270.0426).
# So a second object now requires the two cam ids to be seen within
# SAME_FRAME_S of each other.
SAME_FRAME_S = 0.03

COAST_START_S = 0.3          # camera silent this long before coasting starts
COAST_S = 1.5                # max coast duration after last camera update
COAST_GATE_M = 0.4           # fixed allowance per coast step
COAST_PERIOD_S = 0.1

# THESIS FIX (18 Sep, camray bbox loss). publish() hardcoded
# x1,y1,x2,y2 to zero. Harmless for human_kf_predictor_lidar and
# person_marker_publisher, which read only x/y - but
# social_group_detector_node crops the UNION of two members' boxes and
# runs YOLOv8-pose on that crop. With zero boxes the crop collapsed to
# 20x20 px (measured: every debug_crops file ~925 bytes, all labelled
# "_none"), so NO pair could ever be classified "conversation" and the
# whole ablation F chain was dead on this pipeline.
#
# The boxes exist upstream: /person_positions_map carries them keyed by
# ByteTrack track_id, which is the same id this node already holds as
# cam_id. So cache them here and fill the fields on publish. Verified
# field order in yolo_leg_detector_lidar.py:
#   track_id, conf, x1, y1, x2, y2, keypoints, img_stamp
# Corners, not x/y/w/h.
BBOX_TOPIC = "/person_positions_map"


class CameraRayIdentityNode(Node):
    def __init__(self):
        super().__init__("camera_ray_identity_node")

        self.declare_parameter("input_topic", "/camera_ray_clusters")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("output_topic", "/person_positions_fused")
        self.declare_parameter("bbox_topic", BBOX_TOPIC)
        self.declare_parameter("coast_enable", True)

        self.input_topic = self.get_parameter("input_topic").value
        self.scan_topic = self.get_parameter("scan_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.bbox_topic = self.get_parameter("bbox_topic").value
        self.coast_enable = bool(self.get_parameter("coast_enable").value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.latest_scan = None
        self.next_id = 0
        self.stable_of_cam = {}   # cam_id -> stable_id
        self.cam_last_seen = {}   # cam_id -> t
        self.last_pos = {}        # stable_id -> (x, y, t)   any source
        self.last_cam_t = {}      # stable_id -> t           camera only
        self.pending = {}         # cam_id -> (sid, holder_cam, t_start)
        self.coexist = set()      # frozenset({cam_a, cam_b}) seen in one frame
        self.bbox_of_cam = {}     # cam_id -> (x1, y1, x2, y2, img_stamp)

        self.create_subscription(String, self.input_topic, self.cam_cb, 10)
        self.create_subscription(String, self.bbox_topic, self.bbox_cb, 10)
        self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, 10)
        self.pub = self.create_publisher(String, self.output_topic, 10)
        self.create_timer(COAST_PERIOD_S, self.coast)
        self.create_timer(1.0, self.prune)

        self.get_logger().info(
            f"camera_ray_identity_node: {self.input_topic} -> "
            f"{self.output_topic} (coast={'on' if self.coast_enable else 'off'})")
        self.get_logger().info(f"bbox passthrough from {self.bbox_topic}")

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def scan_cb(self, msg):
        self.latest_scan = msg

    def bbox_cb(self, msg):
        """Cache YOLO boxes by track_id so publish() can fill fields 8-11."""
        p = msg.data.split(",")
        if len(p) < 6:
            return
        try:
            cam_id = int(float(p[0]))
            x1, y1 = int(float(p[2])), int(float(p[3]))
            x2, y2 = int(float(p[4])), int(float(p[5]))
        except ValueError:
            return
        # Last field is the IMAGE capture time, which is the right clock
        # for staleness - it measures when the frame was taken, not when
        # this node got round to parsing it. The keypoints field uses
        # ';' and ':' and contains no commas, so p[-1] is reliably the
        # stamp.
        try:
            stamp = float(p[-1])
        except (ValueError, IndexError):
            stamp = self.now()
        self.bbox_of_cam[cam_id] = (x1, y1, x2, y2, stamp)

    # ------------------------------------------------------------------
    def live_holder(self, sid, now, exclude_cam):
        """cam_id currently holding sid and seen recently, else None."""
        for cid, s in self.stable_of_cam.items():
            if s == sid and cid != exclude_cam and \
                    now - self.cam_last_seen.get(cid, 0.0) < CAM_ID_TIMEOUT_S:
                return cid
        return None

    def allocate(self, cam_id, x, y):
        sid = self.next_id
        self.next_id += 1
        self.get_logger().info(
            f"cam {cam_id} -> NEW stable {sid} at ({x:.2f}, {y:.2f})")
        self.stable_of_cam[cam_id] = sid
        return sid

    def assign(self, cam_id, x, y, now):
        """Stable id for cam_id, or None while the decision is pending."""
        if cam_id in self.stable_of_cam:
            return self.stable_of_cam[cam_id]

        if cam_id in self.pending:
            sid, holder, t0 = self.pending[cam_id]
            if frozenset((cam_id, holder)) in self.coexist:
                del self.pending[cam_id]
                self.get_logger().info(
                    f"cam {cam_id}: seen in same frame as cam {holder} "
                    f"-> second object")
                return self.allocate(cam_id, x, y)
            if now - t0 < PENDING_S:
                return None
            del self.pending[cam_id]
            self.get_logger().info(
                f"cam {cam_id} -> reattach stable {sid} (switch/flip "
                f"from cam {holder})")
            self.stable_of_cam[cam_id] = sid
            return sid

        best, best_d = None, None
        for sid, (px, py, pt) in self.last_pos.items():
            gap = now - pt
            if gap > REATTACH_WINDOW_S:
                continue
            d = math.hypot(x - px, y - py)
            if d <= min(MAX_REATTACH_M, JITTER_M + MAX_SPEED * gap) and \
                    (best_d is None or d < best_d):
                best, best_d = sid, d

        if best is None:
            return self.allocate(cam_id, x, y)

        holder = self.live_holder(best, now, cam_id)
        if holder is not None:
            self.pending[cam_id] = (best, holder, now)
            return None

        self.get_logger().info(
            f"cam {cam_id} -> reattach stable {best} ({best_d:.2f} m)")
        self.stable_of_cam[cam_id] = best
        return best

    def publish(self, sid, conf, x, y, source, cam_id, now):
        # A STALE box is worse than none - the classifier would crop
        # where the person no longer is, and a confident wrong verdict
        # is harder to spot than a missing one. Coasted (lidar_only)
        # samples carry cam_id -1 and never have a box.
        bb = self.bbox_of_cam.get(cam_id)
        if bb is not None and now - bb[4] < CAM_ID_TIMEOUT_S:
            x1, y1, x2, y2 = str(bb[0]), str(bb[1]), str(bb[2]), str(bb[3])
        else:
            x1, y1, x2, y2 = "0", "0", "0", "0"
        fields = [str(sid), conf, f"{x:.3f}", f"{y:.3f}",
                  "0.0", "0", "0", x1, y1, x2, y2, source, str(cam_id)]
        self.pub.publish(String(data=",".join(fields)))
        self.last_pos[sid] = (x, y, now)

    # ------------------------------------------------------------------
    def cam_cb(self, msg):
        p = msg.data.split(",")
        if len(p) < 4:
            return
        try:
            cam_id = int(float(p[0]))
            x, y = float(p[2]), float(p[3])
        except ValueError:
            return
        now = self.now()
        for other, t in self.cam_last_seen.items():
            if other != cam_id and now - t <= SAME_FRAME_S:
                self.coexist.add(frozenset((cam_id, other)))
        self.cam_last_seen[cam_id] = now
        sid = self.assign(cam_id, x, y, now)
        if sid is None:
            return
        self.last_cam_t[sid] = now
        self.publish(sid, "1.00", x, y, "camera_confirmed", cam_id, now)

    # ------------------------------------------------------------------
    def nearest_scan_point(self, x, y, gate):
        """Nearest /scan return to map point (x, y) within gate, in map frame."""
        scan = self.latest_scan
        if scan is None:
            return None
        try:
            to_scan = self.tf_buffer.lookup_transform(
                SCAN_FRAME, MAP_FRAME, Time(), timeout=Duration(seconds=0.05))
            to_map = self.tf_buffer.lookup_transform(
                MAP_FRAME, SCAN_FRAME, Time(), timeout=Duration(seconds=0.05))
        except Exception:
            return None

        q = PointStamped()
        q.header.frame_id = MAP_FRAME
        q.point.x, q.point.y = x, y
        qs = do_transform_point(q, to_scan).point

        best, best_d = None, None
        a = scan.angle_min
        for r in scan.ranges:
            if scan.range_min < r < scan.range_max:
                sx, sy = r * math.cos(a), r * math.sin(a)
                d = math.hypot(sx - qs.x, sy - qs.y)
                if d <= gate and (best_d is None or d < best_d):
                    best, best_d = (sx, sy), d
            a += scan.angle_increment
        if best is None:
            return None

        p = PointStamped()
        p.header.frame_id = SCAN_FRAME
        p.point.x, p.point.y = best
        m = do_transform_point(p, to_map).point
        return m.x, m.y

    def coast(self):
        if not self.coast_enable:
            return
        now = self.now()
        for sid, tc in list(self.last_cam_t.items()):
            silent = now - tc
            if silent < COAST_START_S or silent > COAST_S:
                continue
            px, py, pt = self.last_pos[sid]
            gate = COAST_GATE_M + MAX_SPEED * (now - pt)
            hit = self.nearest_scan_point(px, py, gate)
            if hit is None:
                continue
            self.publish(sid, "0.00", hit[0], hit[1], "lidar_only", -1, now)

    def prune(self):
        now = self.now()
        for cid in [c for c, t in self.cam_last_seen.items()
                    if now - t > REATTACH_WINDOW_S]:
            self.cam_last_seen.pop(cid, None)
            self.stable_of_cam.pop(cid, None)
            self.pending.pop(cid, None)
            self.bbox_of_cam.pop(cid, None)
            self.coexist = {p for p in self.coexist if cid not in p}
        # The bbox cache is keyed by ByteTrack id, which churns faster
        # than cam_last_seen does - drop entries whose image stamp is
        # older than the reattach window so it cannot grow unbounded on
        # a long run.
        for cid in [c for c, b in self.bbox_of_cam.items()
                    if now - b[4] > REATTACH_WINDOW_S]:
            self.bbox_of_cam.pop(cid, None)
        for sid in [s for s, (_, _, t) in self.last_pos.items()
                    if now - t > REATTACH_WINDOW_S]:
            self.last_pos.pop(sid, None)
            self.last_cam_t.pop(sid, None)


def main(args=None):
    rclpy.init(args=args)
    node = CameraRayIdentityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
