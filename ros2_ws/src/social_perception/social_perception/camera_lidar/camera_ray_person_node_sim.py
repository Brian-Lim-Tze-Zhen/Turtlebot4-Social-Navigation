#!/usr/bin/env python3
"""
camera_ray_person_node.py

THESIS ADDITION - camera-primary person detection with LiDAR ranging.

-----------------------------------------------------------------------
WHY THIS EXISTS (architecture inversion)
-----------------------------------------------------------------------
lidar_person_detector.py made the 2D LiDAR the gatekeeper: it decided
what counted as a person using width, motion and static-map filters,
and the camera only confirmed identity afterwards. That ordering is
wrong for these sensors. A 2D LiDAR cannot classify - measured on this
robot, a wall fragment of 0.061 m sits BETWEEN two real leg clusters at
0.048 m and 0.100 m, and a 2-point cluster is collinear by definition
so no shape test can exist. The motion gate (static_move_threshold)
was the only discriminator available, which is why a person standing
still from first sight was invisible to the whole pipeline.

This node inverts it. The CAMERA decides what is a person - that is
what a camera is for. The LiDAR is then asked a much easier question:
"what is the range along this bearing?" No clustering, no width filter,
no motion gate, no static-map subtraction, no confirmed-latch.

MECHANISM: yolo_leg_detector_lidar.py publishes COCO leg keypoints
13-16 (knees, ankles). Ankles sit at the RPLIDAR's own scan height, so
an ankle bearing refers to the SAME physical feature the LiDAR sees.
Cast that bearing into /scan and take the nearest return in a narrow
window around it. That is a MEASURED range at a camera-classified
bearing - not a guess from bbox height, which is unusable here because
the camera is pitched at the floor and y1 is 0-2 px on essentially
every detection.

-----------------------------------------------------------------------
WHAT THIS DOES NOT SOLVE
-----------------------------------------------------------------------
The camera's FOV is the operating envelope. A person who leaves the
frame (which happens at close range in a head-on approach, exactly
when the robot must decide) produces no detection here at all. This
node publishes nothing for them. Continuation across that gap is
identity_fusion_node's job via its lidar_only coast path - this node
deliberately does NOT coast, so it can never invent a phantom.

-----------------------------------------------------------------------
INPUT / OUTPUT
-----------------------------------------------------------------------
Subscribes:
  /person_positions_map   (String) from yolo_leg_detector_lidar.py
      track_id,conf,x1,y1,x2,y2,leg_kpts
      leg_kpts is "px:py;..." for 0-4 visible leg keypoints (COCO
      13-16), possibly empty.
  /scan                   (LaserScan) raw, unclustered.

Publishes:
  /camera_ray_clusters    (String) - "id,conf,map_x,map_y", the SAME
      format lidar_person_detector.py publishes on
      /lidar_person_clusters, so identity_fusion_node can consume this
      instead with only a topic remap. id is the camera track_id (see
      note below), conf is 1.00 (camera-classified = vouched for).

  NOTE ON id: this republishes the CAMERA's ByteTrack track_id, which
  still churns on occlusion. That is deliberate - identity is
  identity_fusion_node's job and it already handles the churn. This
  node is a detector, not a tracker.
"""

import math
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data
from geometry_msgs.msg import PointStamped
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point


# =======================================================================
# Tunables
# =======================================================================

# Half-width of the angular window searched in /scan around the camera
# bearing. Sized from the 11 Sep scan-bearing calibration: residual
# -0.1 +/- 0.9 deg, max 1.5 deg. 3.0 covers that with margin while
# staying far narrower than a person's angular width at 2 m (~14 deg),
# so the window cannot span the person AND the wall behind them.
# Too narrow: misses on a bad keypoint. Too wide: grabs background.
RAY_HALF_WIDTH_DEG = 6.0

# A return further than this is treated as "no person there" - the ray
# passed beside them and hit the far wall. Set above the range the
# camera can classify a person at.
MAX_PERSON_RANGE_M = 8.0

# Returns closer than this are the robot's own shell / noise.
MIN_PERSON_RANGE_M = 0.20

# Occlusion guard. If this camera id's range jumps more than this from
# its previous ray-derived range, the nearest return is probably an
# object that moved in front of them, not the person. Skip rather than
# publish a confidently wrong position. Scales with elapsed time at
# walking pace plus a fixed allowance for ray jitter across the body.
MAX_RANGE_JUMP_M = 0.8       # fixed allowance
MAX_RANGE_SPEED = 0.5        # m/s, additional per-second allowance
RANGE_HISTORY_TIMEOUT = 1.5  # s; forget an id's last range after this

# Occlusion fix A (per-id range tracking)
RANGE_TRACK_GATE_M = 0.4       # fixed allowance around last range
RANGE_TRACK_TIMEOUT_S = 1.0    # CAMERA id unseen longer -> treated as new
RANGE_TRACK_GATE_MAX_M = 0.6   # cap on gate growth while legs are hidden
# THESIS FIX (16 Sep, proto_C): the capped gate locked out a person walking
# away (tracking 2.17 m, refused 5.29->6.22->6.78->7.38 m). Release when a
# return matches the range implied by bbox width (width ~ 1/range) for
# RELEASE_FRAMES frames. Background behind a standing person does not
# match, because the box width does not change.
RELEASE_TOL_M = 0.5
RELEASE_FRAMES = 3
IMAGE_WIDTH_PX = 640           # bbox touching an edge is truncated -> skip
# THESIS FIX (16 Sep, proto_K2): a 2 px edge test was too tight. cam 5
# entered from the left with bbox x1 = 1, 4, 5, 6, 9 and width 28 px for a
# person later measured at ~3.9 m (expected ~47 px): still truncated at
# x1 = 9. Boxes within EDGE_MARGIN_PX of either edge are not trusted as a
# width reference.
EDGE_MARGIN_PX = 38   # SIM: 15 px * (640/250)

# THESIS FIX (16 Sep, proto_K): an object placed in front of a tracked
# person captured the range by two routes, both taking the NEAREST return:
#   (a) the nearer-only reset after OCCLUSION_MAX_MISSES refusals, and
#   (b) every new ByteTrack id (54 ids in a few minutes) starting fresh.
# (a) is removed: a genuinely approaching person widens the bbox, so the
# release rule above accepts the nearer range; a chair does not change
# the box. (b): a new cam id inherits the range track of a track seen
# within INHERIT_S at a bearing within INHERIT_DEG.
INHERIT_S = 0.3             # proto_L2: 0.87 s gap inherited a stale 3.11 m while person was at 6.5 m
INHERIT_DEG = 8.0

# THESIS FIX (16 Sep, proto_K2): a wrong FIRST lock never recovered. cam 5
# locked 2.89 m at first sighting, then refused the person's legs at
# 3.90-3.92 m for 138 frames (~10 s) while they stood still. Release could
# not help: its reference width was stored at first sighting even when the
# box was edge-truncated, and a standing person's box width does not change.
# Now (1) no reference width is stored from an edge-touching box, and
# (2) with no valid reference, the expected range comes from
# RANGE_TIMES_WIDTH_PX / width. Measured on 5 live logs (non-edge boxes):
# range*width median 175-215 px*m, p10 66-142, p90 212-265 - too noisy to
# pick a range on its own, so it only gates the release (RELEASE_TOL_M).
RANGE_TIMES_WIDTH_PX = 407.6   # SIM: 185 * (443.53/201.308) - fx-scaled estimate, re-measure in sim

# THESIS FIX (16 Sep, proto_G replay with absolute width): when an object
# hides the legs, the bbox shrinks instantly (implied range 2.4 -> 5.44 m,
# exactly the door), and the absolute release followed it (far 0 -> 28).
# A person walking away shrinks the box gradually; an occluder does it in
# one frame. Release is allowed only while the implied range has changed
# continuously (<= RHAT_JUMP_M + RHAT_SPEED * dt per frame) since the last
# accepted frame. One discontinuity blocks release until a range is
# accepted normally again.
RHAT_JUMP_M = 0.3   # (retired, see LOCK_* below)
RHAT_SPEED = 1.5

# THESIS FIX (16 Sep, width-release retired): no width rule separated the
# cases. Absolute width fixed proto_K2 but a gradually hidden leg shrinks
# the box like a person walking away (proto_G: release 2.44 -> 5.44 m,
# bbox implied 5.44 = the door; far 0 -> 28, with continuity 18). Back to
# the RELATIVE release (G far 0, C released), and remove the cause of K2
# instead: cam 5 locked 2.89 m on its first frame with bbox x1 = 1 while
# the person was entering; the box kept growing (31 -> 52 px, 89 -> 143 px
# tall) for ~1 s. A new track is not range-locked until its box is off the
# edge and its width is stable, or LOCK_MAX_WAIT_S has passed.
LOCK_STABLE_FRAMES = 3
LOCK_WIDTH_CHANGE = 0.15      # max relative width change across those frames
LOCK_MAX_WAIT_S = 3.0
# THESIS FIX (16 Sep, proto_L Nav2 walk-past replay): cam 12's first box
# touched the right edge (206-247 px); the fresh lock took the nearest
# return in the window, background at 7.59 m, and then refused the person's
# legs as they closed 6.50 -> 5.82 -> 4.91 -> 4.05 -> 3.21 -> 2.34 m
# (38 refusals). No release is possible without a width reference, and an
# edge box gives none. A new track is not range-locked (and not published)
# while its box is within EDGE_MARGIN_PX of an image edge, up to
# LOCK_MAX_WAIT_S. The width-stability condition of the earlier attempt is
# dropped: only the edge condition is used.
OCCLUSION_MAX_MISSES = 3       # refusals before falling back to min-range

# Occlusion fix B (static-map filter)
STATIC_OCCUPIED_THRESHOLD = 50
STATIC_INFLATION_CELLS = 2

# Camera intrinsics for the pixel -> bearing conversion. These MUST
# match whichever branch identity_fusion_node is running, or the two
# nodes disagree about where the person is.
#   uncalibrated branch: CAMERA_FX 201.308 / cx 124.95 / yaw 0.0
#   calibrated (11 Sep): CAMERA_FX 287.31  / cx 124.95 / yaw 2.25
CAMERA_FX = 443.53    # SIM: /oakd/rgb/preview/camera_info k[0]
CAMERA_CX = 320.0     # SIM: camera_info k[2]
CAMERA_YAW_OFFSET_DEG = 0.0

# THESIS FIX (16 Sep, ray timing): scans kept for this long so a
# detection can be paired with the scan taken closest to its image
# stamp. Measured on proto_H: a standing person's map position jumped
# 0.7-1.5 m per second while the robot moved, because the (older)
# detection was paired with the newest scan and pose.
SCAN_BUFFER_S = 1.5
ODOM_EXTRAP_S = 0.30          # odom yaw may be extrapolated this far
MAX_STAMP_MISMATCH_S = 0.30   # no scan this close to the image -> skip

# Frame the RPLIDAR registers as in the TF tree. NOTE: the scan message's
# own header.frame_id is "turtlebot4/rplidar_link/rplidar", which does
# NOT match the tree - see lidar_person_detector.py for the same fix.
SCAN_FRAME = "rplidar_link"
TARGET_FRAME = "map"
BASE_FRAME = "base_link"


class CameraRayPersonNode(Node):
    def __init__(self):
        super().__init__("camera_ray_person_node")

        self.declare_parameter("camera_topic", "/person_positions_map")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("output_topic", "/camera_ray_clusters")
        self.declare_parameter("ray_half_width_deg", RAY_HALF_WIDTH_DEG)
        self.declare_parameter("camera_fx", CAMERA_FX)
        self.declare_parameter("camera_cx", CAMERA_CX)
        self.declare_parameter("camera_yaw_offset_deg", CAMERA_YAW_OFFSET_DEG)
        self.declare_parameter("target_frame", TARGET_FRAME)
        self.declare_parameter("odom_topic", "/odom")

        self.camera_topic = self.get_parameter("camera_topic").value
        self.scan_topic = self.get_parameter("scan_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.ray_half_width = math.radians(
            float(self.get_parameter("ray_half_width_deg").value))
        self.fx = float(self.get_parameter("camera_fx").value)
        self.cx = float(self.get_parameter("camera_cx").value)
        self.cam_yaw = math.radians(
            float(self.get_parameter("camera_yaw_offset_deg").value))
        self.target_frame = self.get_parameter("target_frame").value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.latest_scan = None
        self.scan_buf = deque()   # (stamp_s, LaserScan)
        self.odom_buf = deque()   # (stamp_s, yaw)
        # Yaw of rplidar_link relative to base_link, read from TF once.
        # THESIS FIX (15 Sep): bearings were computed in base_link but
        # searched in the raw scan array, which is in rplidar_link. On
        # this robot the RPLIDAR is mounted at +90 deg yaw, so every ray
        # was cast 90 deg to the side (onto the wall). Measured:
        # tf2_echo base_link rplidar_link -> RPY yaw 90.000.
        self.lidar_yaw = None
        self.map_msg = None
        self.range_track = {}   # cam_id -> (range, t_accepted, misses)
        self.range_seen = {}    # cam_id -> t of last camera detection
        self.range_width = {}   # cam_id -> bbox width (px) at last accepted range
        self.release_count = {}
        self.range_bearing = {}  # cam_id -> (bearing_rad, t)
        self.rhat_prev = {}      # cam_id -> (implied range, t)
        self.rhat_broken = {}    # cam_id -> True after a discontinuity
        self.lock_wait = {}      # cam_id -> (t_first, [recent widths])
        # cam_id -> (range_m, t) for the occlusion guard
        self.last_range = {}

        self.create_subscription(
            LaserScan, self.scan_topic, self.scan_callback, 10)
        self.create_subscription(
            String, self.camera_topic, self.camera_callback, 10)
        self.pub = self.create_publisher(String, self.output_topic, 10)
        self.create_subscription(
            Odometry, self.get_parameter("odom_topic").value,
            self.odom_callback, qos_profile_sensor_data)
        # BEST_EFFORT QoS: the default RELIABLE subscription received nothing
        # (buffer 0 in the proto_L replay) from a best-effort odom publisher.
        map_qos = QoSProfile(depth=1,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(OccupancyGrid, "/map",
                                 self.map_callback, map_qos)

        self.get_logger().info("Camera-ray person node started")
        self.get_logger().info(f"Camera in : {self.camera_topic}")
        self.get_logger().info(f"Scan   in : {self.scan_topic}")
        self.get_logger().info(f"Out       : {self.output_topic}")
        self.get_logger().info(
            f"Ray window: +/-{math.degrees(self.ray_half_width):.1f} deg, "
            f"fx={self.fx:.2f} cx={self.cx:.2f} "
            f"yaw_offset={math.degrees(self.cam_yaw):.2f} deg")

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def scan_callback(self, msg):
        self.latest_scan = msg
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.scan_buf.append((ts, msg))
        while self.scan_buf and ts - self.scan_buf[0][0] > SCAN_BUFFER_S:
            self.scan_buf.popleft()

    def odom_callback(self, msg):
        ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.odom_buf.append((ts, yaw))
        while self.odom_buf and ts - self.odom_buf[0][0] > SCAN_BUFFER_S:
            self.odom_buf.popleft()

    def odom_yaw_at(self, stamp):
        """Odometry yaw at stamp: interpolated inside the buffer, or
        extrapolated from the last two samples up to ODOM_EXTRAP_S past
        either end (odom can arrive later than the image). None otherwise."""
        buf = self.odom_buf
        if len(buf) < 2:
            return None
        for i in range(1, len(buf)):
            t0, y0 = buf[i - 1]
            t1, y1 = buf[i]
            if t0 <= stamp <= t1 and t1 > t0:
                return y0 + self.wrap_angle(y1 - y0) * (stamp - t0) / (t1 - t0)
        if stamp > buf[-1][0] and stamp - buf[-1][0] <= ODOM_EXTRAP_S:
            (t0, y0), (t1, y1) = buf[-2], buf[-1]
        elif stamp < buf[0][0] and buf[0][0] - stamp <= ODOM_EXTRAP_S:
            (t0, y0), (t1, y1) = buf[0], buf[1]
        else:
            return None
        if t1 <= t0:
            return None
        return y1 + self.wrap_angle(y1 - y0) / (t1 - t0) * (stamp - t1)

    def scan_at(self, stamp):
        """Scan closest in time to stamp, or None if none within limit."""
        if not self.scan_buf:
            return None, None
        ts, msg = min(self.scan_buf, key=lambda e: abs(e[0] - stamp))
        dt = ts - stamp
        if abs(dt) > MAX_STAMP_MISMATCH_S:
            return None, dt
        return msg, dt

    def get_lidar_yaw(self):
        if self.lidar_yaw is not None:
            return self.lidar_yaw
        try:
            tf = self.tf_buffer.lookup_transform(
                BASE_FRAME, SCAN_FRAME, Time(),
                timeout=Duration(seconds=0.1))
        except Exception as e:
            self.get_logger().warn(
                f"base_link->rplidar_link TF not ready: {e}",
                throttle_duration_sec=2.0)
            return None
        q = tf.transform.rotation
        self.lidar_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                    1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.get_logger().info(
            f"LiDAR mount yaw (base_link->rplidar_link): "
            f"{math.degrees(self.lidar_yaw):+.2f} deg")
        return self.lidar_yaw

    def pixel_to_bearing(self, u):
        """Image column -> bearing relative to robot heading (CCW +).

        Pinhole, not the linear-FOV approximation: atan((u-cx)/fx) is
        correct at the frame edges where the linear form is worst, and
        the frame edge is exactly where a close person sits.
        """
        return -math.atan((u - self.cx) / self.fx) + self.cam_yaw

    def range_along_bearing(self, bearing):
        """Nearest /scan return within +/-ray_half_width of bearing.

        Returns (range_m, actual_bearing) or (None, None). Takes the
        MINIMUM range in the window: a person standing in front of a
        wall must win over the wall behind them.
        """
        msg = self.latest_scan
        if msg is None:
            return None, None

        best_r, best_a = None, None
        a = msg.angle_min
        for r in msg.ranges:
            if abs(self.wrap_angle(a - bearing)) <= self.ray_half_width:
                if (MIN_PERSON_RANGE_M <= r <= MAX_PERSON_RANGE_M
                        and not math.isinf(r) and not math.isnan(r)):
                    if best_r is None or r < best_r:
                        best_r, best_a = r, a
            a += msg.angle_increment
        return best_r, best_a


    # -----------------------------------------------------------------
    # THESIS FIX (15 Sep, occlusion): min-range alone publishes whatever
    # is nearest in the window, so a bag/chair between robot and person
    # captured the position. Two fixes:
    #   B  drop returns on occupied /map cells (static clutter, walls)
    #   A  per-camera-id range tracking: a KNOWN id takes the return
    #      closest to its last range; a sudden nearer object is refused.
    #      After OCCLUSION_MAX_MISSES consecutive refusals it falls back
    #      to min-range, so a stale reference cannot latch (the Result 6
    #      failure: last-acc guard refused 7 frames; hysteresis design
    #      capped that at 1-3).
    # -----------------------------------------------------------------
    def map_callback(self, msg):
        self.map_msg = msg

    def on_static_obstacle(self, mx, my):
        m = self.map_msg
        if m is None:
            return False
        res = m.info.resolution
        cx = int((mx - m.info.origin.position.x) / res)
        cy = int((my - m.info.origin.position.y) / res)
        w, h = m.info.width, m.info.height
        k = STATIC_INFLATION_CELLS
        for dy in range(-k, k + 1):
            for dx in range(-k, k + 1):
                x, y = cx + dx, cy + dy
                if 0 <= x < w and 0 <= y < h and \
                        m.data[y * w + x] >= STATIC_OCCUPIED_THRESHOLD:
                    return True
        return False

    def window_candidates(self, msg, bearing, tf):
        """All valid returns in the window as (r, a, map_x, map_y),
        excluding returns on static map obstacles."""
        out = []
        a = msg.angle_min
        for r in msg.ranges:
            if abs(self.wrap_angle(a - bearing)) <= self.ray_half_width and \
                    MIN_PERSON_RANGE_M <= r <= MAX_PERSON_RANGE_M and \
                    not math.isinf(r) and not math.isnan(r):
                pt = PointStamped()
                pt.header.frame_id = SCAN_FRAME
                pt.point.x = r * math.cos(a)
                pt.point.y = r * math.sin(a)
                p = do_transform_point(pt, tf).point
                if not self.on_static_obstacle(p.x, p.y):
                    out.append((r, a, p.x, p.y))
            a += msg.angle_increment
        return out

    def select_return(self, cam_id, cands, now, x1=0.0, x2=0.0, bearing=None):
        width = x2 - x1
        edge = x1 <= EDGE_MARGIN_PX or x2 >= IMAGE_WIDTH_PX - EDGE_MARGIN_PX
        closest = min(cands, key=lambda c: c[0])
        # THESIS FIX (16 Sep, proto_G): expire on CAMERA silence, not on
        # refused ranges. Refusals did not refresh the old timer, so hidden
        # legs for >1 s silently reset the track to the door (27 far hits).
        seen = self.range_seen.get(cam_id)
        self.range_seen[cam_id] = now
        if bearing is not None:
            self.range_bearing[cam_id] = (bearing, now)
        st = self.range_track.get(cam_id)
        if (st is None or seen is None or now - seen > RANGE_TRACK_TIMEOUT_S) \
                and bearing is not None:
            donor = None
            for cid, (b, tb) in self.range_bearing.items():
                if cid == cam_id or now - tb > INHERIT_S or cid not in self.range_track:
                    continue
                if abs(math.degrees(self.wrap_angle(b - bearing))) <= INHERIT_DEG:
                    if donor is None or tb > self.range_bearing[donor][1]:
                        donor = cid
            if donor is not None:
                dr, _, _ = self.range_track[donor]
                self.range_track[cam_id] = (dr, now, 0)
                if donor in self.range_width:
                    self.range_width[cam_id] = self.range_width[donor]
                else:
                    self.range_width.pop(cam_id, None)
                self.release_count[cam_id] = 0
                self.get_logger().info(
                    f"cam:{cam_id} inherits range {dr:.2f} m from cam:{donor}")
                st = self.range_track[cam_id]
                seen = now
        if st is None or seen is None or now - seen > RANGE_TRACK_TIMEOUT_S:
            t0, ws = self.lock_wait.get(cam_id, (now, []))
            if now - t0 > LOCK_MAX_WAIT_S + RANGE_TRACK_TIMEOUT_S:
                t0, ws = now, []          # stale wait from an earlier sighting
            ws = (ws + [width])[-LOCK_STABLE_FRAMES:]
            self.lock_wait[cam_id] = (t0, ws)
            stable = (len(ws) >= LOCK_STABLE_FRAMES and min(ws) > 1 and
                      (max(ws) - min(ws)) / max(ws) <= LOCK_WIDTH_CHANGE)
            if edge and now - t0 < LOCK_MAX_WAIT_S:
                return None
            self.lock_wait.pop(cam_id, None)
            self.range_track[cam_id] = (closest[0], now, 0)
            self.rhat_broken[cam_id] = False
            if edge:
                self.range_width.pop(cam_id, None)
            else:
                self.range_width[cam_id] = width
            self.release_count[cam_id] = 0
            return closest
        last_r, last_t, misses = st
        gate = min(RANGE_TRACK_GATE_MAX_M,
                   RANGE_TRACK_GATE_M + MAX_RANGE_SPEED * (now - last_t))
        near = min(cands, key=lambda c: abs(c[0] - last_r))
        if abs(near[0] - last_r) <= gate:
            self.range_track[cam_id] = (near[0], now, 0)
            self.rhat_broken[cam_id] = False
            if not edge:
                self.range_width[cam_id] = width
            self.release_count[cam_id] = 0
            return near
        w0 = self.range_width.get(cam_id)
        if w0 and width > 1 and not edge:
            r_hat = last_r * w0 / width
            # THESIS FIX (16 Sep, proto_K2 replay): the relative estimate
            # last_r * w0 / width inherits the error of a WRONG lock: cam 5
            # stored w0 = 36 px while locked at 2.89 m (person still walking
            # in), so at 53 px it implied 1.96 m and never matched the legs
            # at 3.92 m. The absolute estimate gave 3.49-3.78 m on the same
            # boxes. Use the absolute one only.
            match = min(cands, key=lambda c: abs(c[0] - r_hat))
            if abs(match[0] - r_hat) <= RELEASE_TOL_M and abs(match[0] - last_r) > gate:
                n = self.release_count.get(cam_id, 0) + 1
                self.release_count[cam_id] = n
                if n >= RELEASE_FRAMES:
                    self.get_logger().info(
                        f"cam:{cam_id} release {last_r:.2f} -> {match[0]:.2f} m "
                        f"(bbox implies {r_hat:.2f} m)")
                    self.range_track[cam_id] = (match[0], now, 0)
                    self.range_width[cam_id] = width
                    self.release_count[cam_id] = 0
                    return match
            else:
                self.release_count[cam_id] = 0
        misses += 1
        # THESIS FIX (16 Sep, proto_G): only reset to a NEARER return.
        # Legs hidden behind an unmapped object left only the door
        # (5.5 m) in the window; resetting to it put the person 3 m
        # away and broke identity. A farther return is accepted only
        # via the time-growing gate above, i.e. when physically possible.
        self.range_track[cam_id] = (last_r, last_t, misses)
        self.get_logger().info(
            f"cam:{cam_id} refused {closest[0]:.2f} m (tracking "
            f"{last_r:.2f} m, miss {misses})",
            throttle_duration_sec=0.5)
        return None

    @staticmethod
    def tf_yaw(tf):
        q = tf.transform.rotation
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                          1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    @staticmethod
    def wrap_angle(x):
        while x > math.pi:
            x -= 2 * math.pi
        while x < -math.pi:
            x += 2 * math.pi
        return x

    def camera_callback(self, msg):
        parts = msg.data.split(",")
        if len(parts) < 6:
            return
        try:
            cam_id = int(float(parts[0]))
            x1 = float(parts[2])
            y1 = float(parts[3])
            x2 = float(parts[4])
            y2 = float(parts[5])
        except ValueError:
            return

        now = self.now()

        # Leg keypoints preferred: ankles sit at the scan plane, so
        # their bearing refers to the same physical feature the LiDAR
        # returns. Lowest in frame (largest py) = ankles, not knees.
        kpt = []
        if len(parts) >= 7 and parts[6].strip():
            for pair in parts[6].split(";"):
                if ":" not in pair:
                    continue
                try:
                    px, py = pair.split(":")
                    kpt.append((float(px), float(py)))
                except ValueError:
                    continue

        if kpt:
            kpt.sort(key=lambda q: -q[1])
            use = kpt[:2]
            u = sum(q[0] for q in use) / len(use)
            src = f"kpt{len(use)}"
        else:
            u = (x1 + x2) / 2.0
            src = "bbox"

        bearing = self.pixel_to_bearing(u)          # base_link frame
        lyaw = self.get_lidar_yaw()
        if lyaw is None:
            return
        scan_bearing = self.wrap_angle(bearing - lyaw)  # rplidar_link frame

        # TF first: map-frame coordinates are needed for the static-map
        # filter (fix B) before a return is selected.
        img_stamp = None
        if len(parts) >= 8 and parts[7].strip():
            try:
                img_stamp = float(parts[7])
            except ValueError:
                img_stamp = None
        if img_stamp is not None:
            scan_msg, dt = self.scan_at(img_stamp)
            if scan_msg is None:
                self.get_logger().warn(
                    f"cam:{cam_id} no scan within "
                    f"{MAX_STAMP_MISMATCH_S * 1000:.0f} ms of image "
                    f"(closest {'n/a' if dt is None else f'{dt * 1000:+.0f} ms'})",
                    throttle_duration_sec=2.0)
                return
        else:
            scan_msg, dt = self.latest_scan, None   # old detector: no stamp
        if scan_msg is None:
            return
        try:
            tf = self.tf_buffer.lookup_transform(
                self.target_frame, SCAN_FRAME,
                scan_msg.header.stamp, timeout=Duration(seconds=0.05))
        except Exception:
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.target_frame, SCAN_FRAME,
                    Time(), timeout=Duration(seconds=0.3))
            except Exception as e:
                self.get_logger().warn(f"TF lookup failed: {e}",
                                       throttle_duration_sec=2.0)
                return

        # THESIS FIX (16 Sep, proto_L Nav2 walk-past): with the robot turning,
        # the bearing belongs to the IMAGE pose but the scan was taken up to
        # ~195 ms earlier. Measured: the person's box swept ~110 px in 0.5 s
        # (~45 deg/s), i.e. ~8 deg of rotation inside that lag, more than the
        # +/-6 deg window; hits sat exactly at bearing +/-6 deg on background
        # (6.5-7.3 m). Rotate the ray by the robot's yaw change between the
        # scan time and the image time.
        # First version used TF at the image stamp; in the proto_L replay it
        # returned exactly 0.0 deg on 6 of 7 rays while the robot turned
        # (lookup failed silently). Odometry yaw is buffered locally and
        # interpolated at both stamps instead; relative yaw over <300 ms is
        # what matters, and odom drift over that window is negligible.
        yaw_corr = 0.0
        if img_stamp is not None:
            ts_scan = scan_msg.header.stamp.sec + scan_msg.header.stamp.nanosec * 1e-9
            y_img = self.odom_yaw_at(img_stamp)
            y_scan = self.odom_yaw_at(ts_scan)
            if y_img is not None and y_scan is not None:
                yaw_corr = self.wrap_angle(y_img - y_scan)
                scan_bearing = self.wrap_angle(scan_bearing + yaw_corr)
            else:
                ob = self.odom_buf
                span = (f"odom {ob[0][0] - img_stamp:+.3f}..{ob[-1][0] - img_stamp:+.3f} s"
                        if ob else "odom empty")
                self.get_logger().warn(
                    f"cam:{cam_id} no odom around image/scan stamps "
                    f"(buffer {len(ob)}, {span} rel. image, "
                    f"scan {ts_scan - img_stamp:+.3f} s)",
                    throttle_duration_sec=2.0)
        cands = self.window_candidates(scan_msg, scan_bearing, tf)
        if not cands:
            self.get_logger().info(
                f"no scan return for cam:{cam_id} at "
                f"{math.degrees(bearing):+.1f} deg ({src})",
                throttle_duration_sec=2.0)
            return

        pick = self.select_return(cam_id, cands, now, x1, x2, bearing)
        if pick is None:
            return
        rng, hit_bearing, mx, my = pick
        self.last_range[cam_id] = (rng, now)


        out = String()
        out.data = (f"{cam_id},1.00,"
                    f"{mx:.3f},{my:.3f}")
        self.pub.publish(out)

        self.get_logger().info(
            f"ray cam:{cam_id} bearing {math.degrees(bearing):+.1f} deg "
            f"({src}) -> hit "
            f"{math.degrees(self.wrap_angle(hit_bearing + lyaw)):+.1f} deg "
            f"@ {rng:.2f} m -> map ({mx:.2f}, {my:.2f})" +
            ("" if dt is None else f" scan_dt {dt * 1000:+.0f} ms") +
            f" yaw_corr {math.degrees(yaw_corr):+.1f} deg",
            throttle_duration_sec=0.5)


def main(args=None):
    rclpy.init(args=args)
    node = CameraRayPersonNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()