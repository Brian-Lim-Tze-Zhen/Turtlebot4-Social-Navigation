#!/usr/bin/env python3
"""
identity_fusion_node.py

THESIS ADDITION - camera/lidar identity fusion (re-ID across occlusion).

-----------------------------------------------------------------------
THE PROBLEM
-----------------------------------------------------------------------
Every downstream layer keys its state by ByteTrack's track_id. A ~2s+
camera occlusion makes ByteTrack assign a NEW track_id to the same
physical person, which resets:

  - human_kf_predictor.py : new HumanTrackKF, update_count=0,
                            vx_filt=None, covariance reset, velocity
                            suppressed for 5 updates
  - group_formation_detector.py : close_since dict emptied, so a
                            conversation pair established for 30s must
                            re-accumulate CONV_MIN_DURATION from zero
  - social_group_cloud_node.py (planned) : group_id string changes,
                            costmap zone ages out and rebuilds -> flicker

-----------------------------------------------------------------------
THE FIX
-----------------------------------------------------------------------
The 2D lidar does not share the camera's occlusion geometry. Measured
this session: through a full walk-out / 12s hold / walk-back cycle, the
camera assigned a new track_id on return while ONE lidar track
(leg_detector_node id:2, 659 msgs, y spanning -3.95 to -0.42) stayed
continuous and was never pruned.

So: bind camera track_ids to lidar tracks while both are visible. When
a NEW camera track_id appears near a lidar track that still carries a
binding from a previously-lost camera track, adopt that old identity
instead of allocating a fresh one.

This node is a TRANSPARENT SHIM. It republishes /person_positions_map
verbatim except field [0] (track_id), which is rewritten to a stable
identity. Point human_kf_predictor's existing `input_topic` parameter
at the output topic and nothing else in the pipeline needs to change -
the KF, the close_since timers, and any future group_id strings simply
never observe the switch.

-----------------------------------------------------------------------
WHY SPURIOUS LIDAR TRACKS ARE HARMLESS HERE (design note)
-----------------------------------------------------------------------
leg_detector_node produces spurious tracks from wall fragments: a
person standing in front of a wall splits its continuous return into
short segments, and at ~4.9m range a real leg yields only 2-3 points,
making legs and fragments genuinely indistinguishable by width (measured:
fragment 0.061m sits BETWEEN person clusters at 0.048m and 0.100m) and
by shape (a 2-point cluster is exactly collinear by definition, so
curvature is undefined - no shape test can exist).

Rather than trying to purify the detector, this node makes purity
unnecessary: a lidar track can only ever be adopted as an anchor if it
was CAMERA-CONFIRMED first, i.e. it appeared in stable_of_lidar because
a real camera detection was matched to it. A wall fragment is never
camera-confirmed, so it can never donate an identity no matter how long
it persists. The detector does not need to be clean - only continuous,
which was verified independently.

-----------------------------------------------------------------------
INPUT / OUTPUT
-----------------------------------------------------------------------
Subscribes:
  /person_positions_map    (String) from yolo_detector.py
      track_id,conf,map_x,map_y,depth,u,v,x1,y1,x2,y2
  /lidar_person_clusters   (String) from leg_detector_node.py
      lidar_id,map_x,map_y,age

Publishes:
  /person_positions_fused  (String) - identical to the camera message
      except field [0] replaced by the stable id. All trailing fields
      (including the bbox corners group_formation_detector depends on)
      are passed through untouched.

-----------------------------------------------------------------------
CALIBRATION STATUS - READ BEFORE TRUSTING CAMERA_LIDAR_GATE
-----------------------------------------------------------------------
The camera-to-lidar position offset has NOT been measured yet. The two
sensors estimate the same person differently by construction:

  - lidar centroid sits on the near-facing surface only, biased toward
    the robot (measured 0.09-0.25m offset from ground truth)
  - camera position comes from a torso depth patch reprojected through
    TF, with its own error characteristics

CAMERA_LIDAR_GATE must exceed their systematic disagreement while
staying well under the smallest person-to-person spacing (1.0m in
conversation_test.sdf) or two people will cross-associate.

The default below is a deliberate over-estimate pending measurement.
This node LOGS every camera-lidar pairing distance it evaluates
(look for "pair-dist" lines) precisely so the gate can be set from the
observed distribution after one run instead of guessed. Do that before
relying on any result from this node.
"""

import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


# =======================================================================
# Tunables
# =======================================================================

# Max camera<->lidar separation to consider them the same person.
# PROVISIONAL - see CALIBRATION STATUS above.
CAMERA_LIDAR_GATE = 0.60      # m

# How long a camera track_id -> stable_id mapping is retained after the
# camera stops reporting it. Only needs to outlive brief detection
# dropouts; the lidar binding is what survives real occlusions.
CAMERA_TRACK_TIMEOUT = 2.0    # s

# A stable id is considered "in use" if some camera track reported it
# this recently. Prevents two SIMULTANEOUSLY visible camera tracks from
# collapsing onto one stable id (which would merge two people).
CAMERA_ACTIVE_WINDOW = 1.0    # s

# How long a lidar->stable binding survives without camera confirmation.
# This is the real occlusion budget: it must exceed the longest camera
# gap to be bridged.
#
# MEASURED, not guessed: an earlier value of 30s was set by (wrongly)
# equating the blind window with the mover's 12s hold phase. The actual
# camera-blind window is far longer - the person exits the FOV partway
# through the 35s walk-out and only re-enters late in the 35s walk-back,
# giving ~60s. The binding expired 0.85s before the returning camera
# track appeared, so the re-ID had nothing left to adopt and allocated a
# fresh identity instead. Confirmed the lidar anchor itself was never
# pruned during that window.
#
# Note this timeout is NOT the safety property. The real guard against
# adopting a stale identity is the continuity check in prune(): a
# binding is dropped the moment its lidar track disappears, which is
# exactly what happens if the person genuinely leaves lidar range. That
# is the "long absence" case the design scope explicitly excludes. So
# this can be generous - it only bounds how long a CONTINUOUSLY TRACKED
# but camera-invisible person keeps their identity.
LIDAR_BINDING_TIMEOUT = 300.0  # s

# Drop a cached lidar track this long after its last message.
LIDAR_TRACK_TIMEOUT = 2.0     # s


class IdentityFusionNode(Node):
    def __init__(self):
        super().__init__("identity_fusion_node")

        self.declare_parameter("camera_topic", "/person_positions_map")
        self.declare_parameter("lidar_topic", "/lidar_person_clusters")
        self.declare_parameter("output_topic", "/person_positions_fused")
        self.declare_parameter("camera_lidar_gate", CAMERA_LIDAR_GATE)

        self.camera_topic = self.get_parameter("camera_topic").value
        self.lidar_topic = self.get_parameter("lidar_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.gate = float(self.get_parameter("camera_lidar_gate").value)

        # camera track_id -> [stable_id, last_seen]
        self.stable_of_camera = {}
        # lidar_id -> [stable_id, last_confirmed]   (camera-confirmed only)
        self.stable_of_lidar = {}
        # lidar_id -> (x, y, last_seen)
        self.lidar_tracks = {}

        self.next_stable_id = 0
        self.reid_count = 0

        self.create_subscription(
            String, self.lidar_topic, self.lidar_callback, 10)
        self.create_subscription(
            String, self.camera_topic, self.camera_callback, 10)

        self.pub = self.create_publisher(String, self.output_topic, 10)

        self.create_timer(1.0, self.prune)

        self.get_logger().info("Identity fusion node started")
        self.get_logger().info(f"Camera in : {self.camera_topic}")
        self.get_logger().info(f"Lidar  in : {self.lidar_topic}")
        self.get_logger().info(f"Fused out: {self.output_topic}")
        self.get_logger().info(
            f"Camera-lidar gate: {self.gate:.2f} m (PROVISIONAL - "
            f"calibrate from 'pair-dist' log lines)")
        self.get_logger().info(
            f"Occlusion budget (lidar binding timeout): "
            f"{LIDAR_BINDING_TIMEOUT:.0f} s")

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # -----------------------------------------------------------------
    def lidar_callback(self, msg):
        parts = msg.data.split(",")
        if len(parts) < 3:
            return
        try:
            lid = int(float(parts[0]))
            x = float(parts[1])
            y = float(parts[2])
        except ValueError:
            return
        self.lidar_tracks[lid] = (x, y, self.now())

    # -----------------------------------------------------------------
    def nearest_lidar(self, x, y):
        """Nearest cached lidar track within the gate -> (lidar_id, dist)."""
        best_id, best_d = None, self.gate
        for lid, (lx, ly, _) in self.lidar_tracks.items():
            d = math.hypot(x - lx, y - ly)
            if d < best_d:
                best_id, best_d = lid, d
        return best_id, best_d

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

    # -----------------------------------------------------------------
    def camera_callback(self, msg):
        parts = msg.data.split(",")
        if len(parts) < 4:
            self.get_logger().warn(f"Invalid camera msg: {msg.data}")
            return

        try:
            cam_id = int(float(parts[0]))
            x = float(parts[2])
            y = float(parts[3])
        except ValueError:
            self.get_logger().warn(f"Parse failed: {msg.data}")
            return

        now = self.now()
        lid, dist = self.nearest_lidar(x, y)

        if lid is not None:
            # Logged so CAMERA_LIDAR_GATE can be calibrated from the
            # observed distribution rather than left at its guess.
            self.get_logger().info(
                f"pair-dist cam:{cam_id} <-> lidar:{lid} = {dist:.3f} m")

        known = self.stable_of_camera.get(cam_id)

        if known is not None:
            stable_id = known[0]
        else:
            stable_id = None

            # --- the re-ID step ---------------------------------------
            # A new camera id landed near a lidar track that still holds
            # a binding from an earlier, now-lost camera id. Adopt it.
            if lid is not None and lid in self.stable_of_lidar:
                candidate = self.stable_of_lidar[lid][0]
                if not self.stable_id_in_use(candidate, cam_id, now):
                    stable_id = candidate
                    self.reid_count += 1
                    self.get_logger().info(
                        f"RE-ID #{self.reid_count}: new camera track "
                        f"{cam_id} adopted stable id {stable_id} via "
                        f"lidar track {lid} (dist {dist:.3f} m)")
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

        # Confirm/refresh the lidar binding. THIS is what makes a lidar
        # track eligible to donate an identity later - and what keeps
        # never-confirmed wall fragments permanently ineligible.
        if lid is not None:
            self.stable_of_lidar[lid] = [stable_id, now]

        out = String()
        out.data = ",".join([str(stable_id)] + parts[1:])
        self.pub.publish(out)

    # -----------------------------------------------------------------
    def prune(self):
        now = self.now()

        for cam_id in [c for c, (_, t) in self.stable_of_camera.items()
                       if now - t > CAMERA_TRACK_TIMEOUT]:
            del self.stable_of_camera[cam_id]

        for lid in [l for l, v in self.lidar_tracks.items()
                    if now - v[2] > LIDAR_TRACK_TIMEOUT]:
            del self.lidar_tracks[lid]

        for lid in [l for l, (_, t) in self.stable_of_lidar.items()
                    if now - t > LIDAR_BINDING_TIMEOUT
                    or l not in self.lidar_tracks]:
            sid = self.stable_of_lidar[lid][0]
            age = now - self.stable_of_lidar[lid][1]
            reason = ("lidar track vanished"
                      if lid not in self.lidar_tracks
                      else f"binding expired after {age:.1f}s")
            del self.stable_of_lidar[lid]
            self.get_logger().info(
                f"Dropped lidar binding {lid} -> stable {sid} ({reason})")


def main(args=None):
    rclpy.init(args=args)
    node = IdentityFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
