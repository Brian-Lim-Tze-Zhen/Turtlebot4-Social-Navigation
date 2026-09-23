#!/usr/bin/env python3
"""
predicted_person_cloud_node.py

Turns tracked people into PointCloud2 obstacle regions for the Nav2
costmaps: a disk at where each person is now, plus a directional
ellipse over the lane they are about to walk through.

Consumed by:
  local_costmap.nonpersistent_voxel_layer   (reactive avoidance)
  global_costmap.nonpersistent_voxel_layer  (route planning)

NonPersistentVoxelLayer is required rather than VoxelLayer: VoxelLayer
clears by raytracing, which needs a real sensor origin. These are
synthetic points with none, so marks would persist forever.
"""

import math
import struct

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from nav2_msgs.srv import ClearEntireCostmap
from nav2_msgs.srv import ClearCostmapAroundPose
import tf2_ros

from std_msgs.msg import String, Header
from sensor_msgs.msg import PointCloud2, PointField


# ---------------------------------------------------------------------
# Geometry. All lane dimensions live here so the derived trail-clearing
# radius below cannot drift out of sync with them.
#
# Every value in this block is the DEPLOYED value. Earlier drafts of
# this file carried comments citing figures the code no longer used
# (b=0.20/0.30 against an actual ELLIPSE_B of 1.20, lateral_bias=0.20
# against an actual 0.40), which caused at least one wrong diagnosis
# during tuning. If you change a number here, change its comment.
# ---------------------------------------------------------------------

# Below this speed the heading estimate is dominated by camera jitter,
# so the person is marked with a symmetric disk instead of a directional
# ellipse. Measured noise floor with people bolted in place in
# conversation_test.sdf: 0.028 m/s stationary, 0.040 m/s with the robot
# rotating. 0.05 sits an order of magnitude above that and well below a
# genuine ~0.3 m/s walking pace.
STATIONARY_SPEED_DEADBAND = 0.05   # m/s

# Ellipse half-width ACROSS the heading. Kept at parity with the person
# disk radius below: a lane narrower than the person it represents left
# the predicted region half the width of the body it stood for.
ELLIPSE_B = 0.60                   # m (shrunk from 1.20 - was wider than long, causing round/omnidirectional footprint)

# Ellipse half-length ALONG the heading, as a function of walking speed.
# At the 1.2 m/s test speed this gives a = 0.60 + 0.60 = 1.20 m, so the
# marked lane runs ~2.4 m end to end.
ELLIPSE_A_BASE = 0.60              # m
ELLIPSE_A_SLOPE = 0.50             # m per (m/s)
ELLIPSE_A_MAX = 3.00               # m

# Perpendicular offset of the lane, breaking head-on left/right symmetry
# deterministically (social "keep right"). Ratio to ELLIPSE_B is
# 0.40/1.20 = 0.33.
LATERAL_BIAS = 0.20  # m

# THESIS MODIFICATION (dynamic pass-side, replaces fixed keep-right)
#
# The original fixed bias (always toward the person's right) only
# resolves head-on symmetry when the robot happens to already be on
# that side. Spawn the robot on the opposite side, or in a corridor
# narrow enough that the fixed bias eats the robot's own margin (e.g.
# a 0.5 m lateral offset against a 0.4 m bias), and the ellipse can
# push into space the robot has no room to vacate.
#
# Instead, bias toward whichever side the robot is CURRENTLY on: this
# nudges the ellipse into the robot's own lane, forcing it toward the
# empty opposite lane, regardless of which side that happens to be.
# This generalises the old fixed case (robot always spawned on the
# person's right) rather than replacing its mechanism.
#
# Below LATERAL_SIDE_DEADBAND perpendicular distance from the person's
# heading line, the side estimate is dominated by noise (robot near
# the centreline / near-head-on approach), so the last committed side
# is held rather than recomputed — recomputing every cycle here is
# exactly the kind of frame-to-frame flip that produces hesitation.
LATERAL_SIDE_DEADBAND = 0.15      # m

# ---------------------------------------------------------------------
# THESIS NOTE (dynamic pass side disabled pending a hysteresis fix)
#
# False = the fixed "person's right" bias of conditions A-E. The dynamic
# version is left in place rather than deleted: its premise is sound - a
# fixed keep-right only resolves the symmetry when the robot is already
# on that side, and pushes it INTO the person otherwise.
#
# What failed is the hysteresis, not the idea. callback() rebuilds each
# track dict on every incoming message and carries forward heading_sin,
# heading_cos and history from the previous entry - but NOT
# lateral_side. The committed side was therefore erased every cycle, and
# the deadband that exists to stop frame-to-frame flips never held
# anything.
#
# Observed: the ellipse appearing centred in RViz - it alternates
# +-0.40 m fast enough to average out - and the controller hesitating
# left/right, which is exactly the failure the fixed bias was introduced
# to remove.
#
# Re-enable after carrying lateral_side forward in callback() and
# confirming on a head-on run that the side changes at most once or
# twice per encounter.
# ---------------------------------------------------------------------
ENABLE_DYNAMIC_PASS_SIDE = False

# Person footprint marked at the current position, and the fallback disk
# used when the ellipse is suppressed. Same radius, different sampling:
# the current-position disk is denser because it is what the local
# planner collides against.
PERSON_DISK_RADIUS = 0.40    # m
PERSON_DISK_SPACING = 0.10         # m
FALLBACK_DISK_SPACING = 0.15       # m
ELLIPSE_SPACING = 0.15             # m

# Keep-out radius around the robot. Deliberately just larger than the
# TurtleBot4 footprint (0.189 m): see _apply_robot_keepout for why this
# must not be widened.
ROBOT_KEEPOUT_RADIUS = 0.20       # m

TRACK_TIMEOUT = 0.30               # s before a silent track is dropped
PUBLISH_RATE_HZ = 10.0             # cloud rate, decoupled from detection
HEADING_SMOOTH_ALPHA = 0.40        # EMA on the heading sin/cos
HEADING_MIN_DISPLACEMENT = 0.15    # m; below this, hold the last heading
HISTORY_LENGTH = 60                # positions retained per track

# Trail clearing. Derived from the lane geometry so it stays correct if
# the lane changes: the outermost ellipse mark sits at
# LATERAL_BIAS + ELLIPSE_B from the person's path.
TRAIL_CLEAR_MARGIN = 0.20          # m
TRAIL_LAG_MARGIN = 0.20            # m

PERSON_DISK_FORWARD = 0   # m，圆盘沿行进方向的前移量

# ---------------------------------------------------------------------
# THESIS FIX (duplicate tracks for one person)
#
# This node keys active_tracks by track_id and marks every one of them.
# That is correct only while ids are stable. Measured with four static
# pedestrians in queue_test: ByteTrack reassigned ids continuously (1 ->
# 75 over one run) and TRACK_TIMEOUT (0.30 s) is longer than the churn
# interval, so a person's old and new ids coexisted. Peak: 10 active
# tracks for 4 real people, cloud size swinging 142 <-> 1420 points every
# cycle. The cost field was both ~2.5x too large and reshaping
# constantly - enough to make the local planner spin in place with this
# layer alone.
#
# Two ids at the same coordinates are the same person, whatever
# ByteTrack believes. Tracks closer than this are merged, keeping the
# most recently seen. Well under the 1.2 m queue spacing, so genuinely
# distinct people are never collapsed.
# ---------------------------------------------------------------------
DUPLICATE_TRACK_RADIUS = 0.45      # m

# ---------------------------------------------------------------------
# THESIS ADDITION (layer responsibility split)
#
# A person recognised as part of a social group is marked by
# social_group_cloud_node, not here. Marking them in both layers means
# the cost field around them is the sum of two independently designed
# shapes - which is nobody's design.
#
# Measured in the queue scenario (4 static pedestrians, n=2 vs n=3):
#   group layer only : min_dist 1.129, mean_spd 0.26, commit_dist 4.4 m
#   both layers      : min_dist 1.126, mean_spd 0.21, commit_dist 4.0 m
# i.e. 17% slower, 25% longer, reacting 0.4 m later, for no change in
# clearance. Consistent with the cost-gradient flattening already
# observed across several geometric interventions.
#
# There is also a mechanism reason, not just a tuning one. This layer
# marks TRACKED individuals, so its coverage follows camera visibility:
# when the robot turns away mid-manoeuvre the track times out and the
# cost region vanishes, reappearing under a new id. For a queue member
# who is intermittently visible but never actually moves, that produces
# a region flickering with the robot's own heading. The group layer's
# position-anchored hold does not have this property - it marks occupied
# SPACE rather than a tracked individual. Running this layer alone on
# the queue left the local planner spinning in place with no stable
# solution.
#
# So the split is not a per-scenario switch. Both layers stay enabled
# everywhere; whichever has the better-founded claim on a given person
# takes them. A pedestrian in no group keeps their directional ellipse
# here; a bystander next to a group is unaffected.
#
# Matched by POSITION rather than by the member_ids in the message:
# those are detector track_ids, and id churn is exactly what this
# pipeline cannot rely on. Radius is generous relative to the
# camera-lidar offsets seen (0.04-0.11 m) and well under person spacing.
# ---------------------------------------------------------------------
GROUP_MEMBER_TOPIC = "/social_groups"
GROUP_MEMBER_RADIUS = 0.50         # m; track within this of a member = that member
GROUP_MEMBER_TIMEOUT = 2.0         # s; stop suppressing if group detection dies

class PredictedPersonCloudNode(Node):
    def __init__(self):
        super().__init__("predicted_person_cloud_node")

        self.frame_id = "map"
        self.robot_frame = "base_link"

        self.sub = self.create_subscription(
            String, "/predicted_person_positions", self.callback, 10)
        self.pub = self.create_publisher(
            PointCloud2, "/predicted_person_cloud", 10)

        # ==========================================================
        # THESIS MODIFICATION (stale mark clearing)
        #
        # NonPersistentVoxelLayer resets its own grid each cycle, but
        # updateCosts() only writes into the master costmap within the
        # bounds it reports. With no observations there are no bounds,
        # so stale master cells are never overwritten -- confirmed by
        # publishing empty clouds (width=0) while marks persisted with
        # the robot parked. clearing:true does not help: raytrace
        # clearing needs points to trace rays TO, and an empty cloud
        # has none.
        #
        # Fix: clear the local costmap once when the last track
        # expires, on the non-empty -> empty transition only. This
        # resets every layer, but obstacle_layer repopulates from the
        # next scan and static_layer from the latched map.
        #
        # LOCAL ONLY. The global equivalent used to be called here too;
        # at 10 Hz with a 1.8 m reset_distance it saturated
        # planner_server and made compute_path_to_pose time out with
        # "Goal failed" mid-run.
        # ==========================================================
        self._had_tracks = False
        self._clear_local = self.create_client(
            ClearEntireCostmap, "/local_costmap/clear_entirely_local_costmap")

        # ==========================================================
        # THESIS FIX (stale trail)
        #
        # Clear a small disk at the person's position from at least
        # _trail_min_lag metres back, so cells the ellipse previously
        # occupied get overwritten while the live zone (back edge at
        # the person's current position) is never touched. Selecting
        # the point by DISTANCE rather than time keeps this
        # independent of walking speed.
        # ==========================================================
        self._clear_pose_local = self.create_client(
            ClearCostmapAroundPose,
            "/local_costmap/clear_around_pose_local_costmap")
        self._trail_clear_radius = LATERAL_BIAS + ELLIPSE_B + TRAIL_CLEAR_MARGIN
        self._trail_min_lag = self._trail_clear_radius + TRAIL_LAG_MARGIN

        # ==========================================================
        # THESIS MODIFICATION (multi-track fix)
        #
        # This node used to rebuild and publish a fresh cloud on every
        # incoming message, containing only that message's track_id.
        # With several people in the scene each update overwrote the
        # previous person's cloud, so Nav2 only ever saw one of them.
        #
        # Fix: keep the latest state per track_id and publish one cloud
        # built from ALL active tracks on a timer. Tracks silent for
        # longer than TRACK_TIMEOUT are pruned, so a person who leaves
        # the camera FOV does not leave a phantom obstacle behind.
        # ==========================================================
        # Positions of people currently claimed by the group layer.
        # (x, y, last_seen). Cleared by age, so a group detector failure
        # returns those people to this layer rather than leaving them
        # unmarked by anything.
        self.group_member_xy = []
        self.create_subscription(
            String, GROUP_MEMBER_TOPIC, self.group_callback, 10)

        self.active_tracks = {}
        self.last_robot_xy = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.publish_timer = self.create_timer(
            1.0 / PUBLISH_RATE_HZ, self.publish_cloud)

        self.get_logger().info("Predicted person cloud node started")
        self.get_logger().info("Subscribing: /predicted_person_positions")
        self.get_logger().info("Publishing: /predicted_person_cloud")
        self.get_logger().info(f"Cloud frame: {self.frame_id}")
        self.get_logger().info(
            f"Track timeout: {TRACK_TIMEOUT:.2f}s | "
            f"Publish rate: {PUBLISH_RATE_HZ:.1f} Hz")
        self.get_logger().info(
            f"Lane: a = {ELLIPSE_A_BASE:.2f} + {ELLIPSE_A_SLOPE:.2f}*speed "
            f"(max {ELLIPSE_A_MAX:.2f}), b = {ELLIPSE_B:.2f}, "
            f"bias = {LATERAL_BIAS:.2f}")
        self.get_logger().info(
            f"Robot keep-out: {ROBOT_KEEPOUT_RADIUS:.2f} m "
            f"around frame '{self.robot_frame}'")

    # -----------------------------------------------------------------
    # Time
    # -----------------------------------------------------------------

    def get_ros_time_seconds(self):
        # Node clock, so this respects use_sim_time and falls back
        # correctly to wall clock when it is false.
        return self.get_clock().now().nanoseconds / 1e9

    # -----------------------------------------------------------------
    # Input
    # -----------------------------------------------------------------

    def callback(self, msg):
        """Record one track's latest state.

        Cloud construction happens in publish_cloud() so that every
        active track is represented together, not just whichever one
        published most recently.

        Wire format (comma separated), matched to human_kf_predictor:
            [0] track_id  [2] cur_x  [3] cur_y  [4] vx  [5] vy
            [6] pred_x    [7] pred_y [9] rotation_gated
        """
        parts = msg.data.split(",")
        if len(parts) < 9:
            self.get_logger().warn(f"Invalid msg: {msg.data}")
            return

        try:
            track_id = int(float(parts[0]))
            current_x = float(parts[2])
            current_y = float(parts[3])
            vx = float(parts[4])
            vy = float(parts[5])
            predicted_x = float(parts[6])
            predicted_y = float(parts[7])
        except ValueError:
            self.get_logger().warn(f"Parse failed: {msg.data}")
            return

        # Field [9] is human_kf_predictor's rotation gate: 1 when
        # |odom angular.z| exceeded its threshold, meaning the velocity
        # EMA was frozen because the robot's own rotation was
        # contaminating the camera-derived position estimate.
        rotation_gated = len(parts) > 9 and parts[9].strip() == "1"

        existing = self.active_tracks.get(track_id)
        s, c = self._smooth_heading(
            existing, predicted_x - current_x, predicted_y - current_y)

        prev_hist = existing.get("history", []) if existing is not None else []
        new_hist = (prev_hist + [(current_x, current_y)])[-HISTORY_LENGTH:]

        self.active_tracks[track_id] = {
            "current": (current_x, current_y),
            "predicted": (predicted_x, predicted_y),
            "speed": math.hypot(vx, vy),
            "rotation_gated": rotation_gated,
            "last_seen": self.get_ros_time_seconds(),
            "heading_sin": s,
            "heading_cos": c,
            "history": new_hist,
            # Carried forward like the heading: this dict is REBUILT on
            # every incoming message, so a committed pass side written
            # by _robot_lateral_side was erased on the next one. The
            # hysteresis that exists to stop frame-to-frame side flips
            # therefore never held anything, and the side reverted to
            # the +1 default every cycle.
            "lateral_side": (existing.get("lateral_side", 1)
                             if existing is not None else 1),
        }

    def group_callback(self, msg):
        """Cache member coordinates from /social_groups field 9.

        Field 9 is "x;y|x;y|..." in member_ids order. Absent on older
        publishers, in which case nothing is suppressed - failing open is
        correct here, since the alternative is a person marked by neither
        layer.
        """
        parts = msg.data.split(",")
        if len(parts) < 10 or not parts[9].strip():
            return

        # ==========================================================
        # THESIS FIX (deferral is type-dependent)
        #
        # Only QUEUE members are deferred. The two group types make
        # different claims on the space they occupy:
        #
        #   queue - social_group_cloud_node fills the gap between EVERY
        #     consecutive pair, so the chain of gaps plus inflation
        #     covers the whole line. Marking the bodies as well is the
        #     double-marking that made the outcome non-reproducible
        #     (min_dist 0.920 +/- 0.356 over three runs).
        #
        #   conversation - it fills ONLY the o-space between the pair,
        #     by explicit design: walking through the middle of a
        #     conversation is the socially disruptive act, while passing
        #     behind either person is not, and filling the bodies too
        #     would wall off a corridor. That design assumes the bodies
        #     are marked HERE. Deferring them leaves a conversing pair
        #     represented by a ~0.5 x 0.7 m patch of gap and nothing
        #     else - observed as a global plan that did not reroute at
        #     all, because there was almost nothing to route around.
        #
        # So the rule is not "the group layer owns its members". It is
        # "whoever covers that person's body owns them", which is the
        # group layer for a queue and this node for a conversation.
        # ==========================================================
        if parts[1].strip() != "queue":
            return

        now = self.get_ros_time_seconds()
        try:
            for pair in parts[9].strip().split("|"):
                x, y = (float(v) for v in pair.split(";"))
                self.group_member_xy.append((x, y, now))
        except ValueError:
            return

    def _is_group_member(self, x, y, now):
        for mx, my, seen in self.group_member_xy:
            if now - seen > GROUP_MEMBER_TIMEOUT:
                continue
            if math.hypot(x - mx, y - my) <= GROUP_MEMBER_RADIUS:
                return True
        return False

    def _smooth_heading(self, existing, dx, dy):
        """EMA the heading as sin/cos, returning the smoothed pair.

        Interpolating the components rather than the angle keeps the
        result on the unit circle and avoids the discontinuity at +/-pi
        that would otherwise make the heading spin the long way round
        when a person reverses.

        Without smoothing the ellipse snapped whenever the person turned,
        a noisy depth reading moved the predicted point, or ego-motion
        from robot rotation leaked into the KF velocity.
        """
        # ==========================================================
        # THESIS FIX (heading gate sized against noise, not zero)
        #
        # (dx, dy) is predicted - current. At the 1.2 m/s test walking
        # speed that displacement is on the order of a metre; for a
        # person standing still it is KF noise of a few centimetres. The
        # old 0.01 m gate let the noise through, so the heading of a
        # stationary person rotated continuously - and since the
        # fallback disk is offset by LATERAL_BIAS along that heading,
        # the disk swung around a 0.4 m circle every cycle. Four static
        # pedestrians produced four cost regions in constant motion.
        #
        # 0.15 m is an order of magnitude above the observed noise and
        # an order of magnitude below a walking person's displacement,
        # so this changes nothing for a moving pedestrian - including on
        # the rotation-gated path, which is where head-on avoidance
        # depends on the heading being right.
        #
        # Below the gate the previous heading is held (see below), so a
        # person who stops keeps the direction they were last walking,
        # which is the correct guess rather than an arbitrary one.
        # ==========================================================
        if math.hypot(dx, dy) > HEADING_MIN_DISPLACEMENT:
            raw = math.atan2(dy, dx)
            raw_sin, raw_cos = math.sin(raw), math.cos(raw)
            if existing is not None and "heading_sin" in existing:
                a = HEADING_SMOOTH_ALPHA
                return (a * raw_sin + (1.0 - a) * existing["heading_sin"],
                        a * raw_cos + (1.0 - a) * existing["heading_cos"])
            return raw_sin, raw_cos

        # No measurable displacement: hold the last heading if there is
        # one, otherwise default to east.
        if existing is not None and "heading_sin" in existing:
            return existing["heading_sin"], existing["heading_cos"]
        return 0.0, 1.0

    # -----------------------------------------------------------------
    # Per-cycle steps
    # -----------------------------------------------------------------

    def _prune_stale_tracks(self, now):
        """Drop tracks gone silent, and clear the costmap once if empty."""
        stale = [
            tid for tid, t in self.active_tracks.items()
            if now - t["last_seen"] > TRACK_TIMEOUT
        ]
        for tid in stale:
            del self.active_tracks[tid]
            self.get_logger().info(
                f"Pruned stale track id:{tid} from obstacle cloud")

        if not self.active_tracks and self._had_tracks:
            if self._clear_local.service_is_ready():
                self._clear_local.call_async(ClearEntireCostmap.Request())
            self.get_logger().info("All tracks expired - cleared local costmap")

        self._had_tracks = bool(self.active_tracks)

    def _clear_trail(self, track):
        """Clear a disk at the oldest history point still far enough back."""
        current_x, current_y = track["current"]
        target = None
        for hx, hy in reversed(track.get("history", [])):
            if math.hypot(current_x - hx, current_y - hy) > self._trail_min_lag:
                target = (hx, hy)
                break
        if target is None:
            return
        if not self._clear_pose_local.service_is_ready():
            return

        req = ClearCostmapAroundPose.Request()
        req.pose.header.frame_id = self.frame_id
        req.pose.header.stamp = self.get_clock().now().to_msg()
        req.pose.pose.position.x = float(target[0])
        req.pose.pose.position.y = float(target[1])
        req.pose.pose.orientation.w = 1.0
        req.reset_distance = self._trail_clear_radius
        self._clear_pose_local.call_async(req)

    def _track_points(self, track):
        current_x, current_y = track["current"]
        predicted_x, predicted_y = track["predicted"]

        rotation_gated = track.get("rotation_gated", False)
        speed = track.get("speed", 0.0)
        use_ellipse = not (rotation_gated or speed < STATIONARY_SPEED_DEADBAND)

        if "heading_sin" in track:
            heading = math.atan2(track["heading_sin"], track["heading_cos"])
        elif use_ellipse:
            heading = math.atan2(predicted_y - current_y, predicted_x - current_x)
        else:
            heading = None

        # 圆盘沿行进方向前移，让它落在人即将占据的位置而不是当前位置
        disk_x, disk_y = current_x, current_y
        if heading is not None:
            disk_x += PERSON_DISK_FORWARD * math.cos(heading)
            disk_y += PERSON_DISK_FORWARD * math.sin(heading)

        points = self.make_disk_points(
            disk_x, disk_y,
            radius=PERSON_DISK_RADIUS,
            spacing=PERSON_DISK_SPACING,
            z=0.3)

        if use_ellipse:
            points.extend(self._ellipse_points(current_x, current_y, heading, speed, track))
        else:
            points.extend(self._fallback_disk_points(predicted_x, predicted_y, heading, track))
        return points

    def _robot_lateral_side(self, current_x, current_y, heading, track):
        """Return +1 or -1: which side of the person's heading line the
        robot is currently on, with hysteresis.

        THESIS MODIFICATION — see LATERAL_SIDE_DEADBAND comment above
        LATERAL_BIAS for why this holds the last decision near the
        centreline instead of recomputing every cycle.

        Falls back to +1 (the old fixed "person's right" side) if no
        robot pose has been observed yet, so behaviour degrades to the
        original fixed-bias scheme rather than failing outright.
        """
        if not ENABLE_DYNAMIC_PASS_SIDE:
            return 1
        if self.last_robot_xy is None:
            return track.get("lateral_side", 1)

        rx, ry = self.last_robot_xy
        # (perp_x, perp_y) is heading rotated -90 deg — the person's
        # right-hand side, same convention as the bias itself below.
        perp_x = math.sin(heading)
        perp_y = -math.cos(heading)

        rvx = rx - current_x
        rvy = ry - current_y
        perp_dist = rvx * perp_x + rvy * perp_y

        if abs(perp_dist) < LATERAL_SIDE_DEADBAND:
            # Too close to the centreline to trust — hold the last
            # committed side (default to +1 if none decided yet).
            return track.get("lateral_side", 1)

        side = 1 if perp_dist > 0 else -1
        track["lateral_side"] = side
        return side

    def _ellipse_points(self, current_x, current_y, heading, speed, track):
        a = min(ELLIPSE_A_MAX, ELLIPSE_A_BASE + speed * ELLIPSE_A_SLOPE)

        # Shift forward by a along the heading so the ellipse's BACK edge
        # sits at the current position. Centring it on the predicted
        # point instead would extend the lane behind the person, and by
        # a distance that grew with their speed.
        cx = current_x + a * math.cos(heading)
        cy = current_y + a * math.sin(heading)

        # ==========================================================
        # THESIS MODIFICATION (head-on symmetry break / pass-right)
        #
        # A perfectly head-on approach leaves the ellipse symmetric
        # left-right about the heading axis, so MPPI's cost gradient
        # carries no directional preference and the optimiser has to
        # wait on sampling noise to break the tie. Observed as visible
        # hesitation and a late, close-range (<0.3 m) decision.
        #
        # Shifting the lane centre perpendicular to the heading, always
        # to the same side (keep-right, matching pedestrian convention),
        # resolves the pass side deterministically and early.
        #
        # (perp_x, perp_y) is the heading rotated -90 deg: the person's
        # right-hand side. Keep this in step with the fallback disk's
        # bias below, so the choice of side does not flip depending on
        # which branch runs this cycle.
        # ==========================================================
        # ==========================================================
        # THESIS MODIFICATION (dynamic pass-side — see LATERAL_BIAS /
        # LATERAL_SIDE_DEADBAND comments above for the full rationale
        # and the deadband/hysteresis this depends on)
        #
        # Was: always shift toward the person's right (fixed +1).
        # Now: shift toward whichever side the robot currently
        # occupies, so the ellipse pushes into the robot's own lane
        # and forces it toward the empty opposite lane — this works
        # regardless of which side the robot starts on, instead of
        # only working when it happens to start on the person's right.
        # ==========================================================
        side = self._robot_lateral_side(current_x, current_y, heading, track)
        cx += side * LATERAL_BIAS * math.sin(heading)
        cy += side * LATERAL_BIAS * -math.cos(heading)

        return self.make_ellipse_points(
            cx, cy, heading=heading, a=a, b=ELLIPSE_B,
            spacing=ELLIPSE_SPACING, z=0.3)

    def _fallback_disk_points(self, predicted_x, predicted_y, heading, track):
        cx, cy = predicted_x, predicted_y
        if heading is not None:
            # Same side as the ellipse. Zeroing the heading while
            # rotation-gated made the pass-side bias vanish at exactly
            # the moment it mattered most - swerve onset.
            side = self._robot_lateral_side(cx, cy, heading, track)
            cx += side * LATERAL_BIAS * math.sin(heading)
            cy += side * LATERAL_BIAS * -math.cos(heading)
        return self.make_disk_points(
            cx, cy,
            radius=PERSON_DISK_RADIUS,
            spacing=FALLBACK_DISK_SPACING,
            z=0.3)

    def _apply_robot_keepout(self, points):
        """Drop points landing on the robot itself.

        ==========================================================
        THESIS MODIFICATION (robot keep-out filter)

        In a head-on encounter the ellipse points straight at the
        approaching robot. Once the gap closes below the lane's forward
        extent plus costmap inflation, these synthetic points land on
        the robot's own footprint. The robot is then standing inside
        lethal cost generated by its own prediction layer, and the
        controller can no longer score a valid forward trajectory: it
        oscillates between "blocked" and "clear", or sinks into the
        inflation and stalls.

        The radius is footprint-sized on purpose. Widening it carves a
        moving hole through the risk zone and lets the robot push
        straight down the person's lane - which is the failure this
        filter exists to prevent, arrived at from the other direction.

        On TF failure the last known robot pose is reused; with none
        known yet the cloud passes through unfiltered.
        ==========================================================
        """
        try:
            tfm = self.tf_buffer.lookup_transform(
                self.frame_id, self.robot_frame,
                rclpy.time.Time(), timeout=Duration(seconds=0.05))
            self.last_robot_xy = (tfm.transform.translation.x,
                                  tfm.transform.translation.y)
        except Exception:
            pass

        if self.last_robot_xy is None or not points:
            return points

        rx, ry = self.last_robot_xy
        r2 = ROBOT_KEEPOUT_RADIUS ** 2
        kept = [
            p for p in points
            if (p[0] - rx) ** 2 + (p[1] - ry) ** 2 > r2
        ]
        removed = len(points) - len(kept)
        if removed:
            self.get_logger().info(
                f"Keep-out: removed {removed} point(s) within "
                f"{ROBOT_KEEPOUT_RADIUS:.2f} m of robot")
        return kept

    def publish_cloud(self):
        now = self.get_ros_time_seconds()
        self._prune_stale_tracks(now)

        self.group_member_xy = [m for m in self.group_member_xy
                                if now - m[2] <= GROUP_MEMBER_TIMEOUT]

        points = []
        suppressed = 0
        for track in self._deduplicated_tracks():
            cx, cy = track["current"]
            if self._is_group_member(cx, cy, now):
                suppressed += 1
                continue
            self._clear_trail(track)
            points.extend(self._track_points(track))
        if suppressed:
            self.get_logger().info(
                f"Deferred {suppressed} track(s) to the group layer")

        points = self._apply_robot_keepout(points)
        self.pub.publish(self.create_cloud(points, self.frame_id))

        if self.active_tracks:
            ids = ",".join(str(tid) for tid in self.active_tracks)
            gated = [
                str(tid) for tid, t in self.active_tracks.items()
                if t.get("rotation_gated", False)
            ]
            gated_str = f" [ROT GATED: {','.join(gated)}]" if gated else ""
            self.get_logger().info(
                f"Published cloud for {len(self.active_tracks)} track(s) "
                f"[{ids}] points={len(points)}{gated_str}")

    def _deduplicated_tracks(self):
        """Collapse tracks that sit on top of each other into one.

        See DUPLICATE_TRACK_RADIUS. Freshest track wins, so the surviving
        entry carries the most recent heading and speed estimate rather
        than a stale one from an id that is about to be pruned.
        """
        ordered = sorted(self.active_tracks.values(),
                         key=lambda t: t["last_seen"], reverse=True)
        kept = []
        for t in ordered:
            x, y = t["current"]
            if any(math.hypot(x - k["current"][0], y - k["current"][1])
                   < DUPLICATE_TRACK_RADIUS for k in kept):
                continue
            kept.append(t)
        if len(kept) < len(ordered):
            self.get_logger().info(
                f"Merged {len(ordered) - len(kept)} duplicate track(s) "
                f"-> {len(kept)} distinct person(s)")
        return kept

    def destroy_node(self):
        self.pub.publish(self.create_cloud([], self.frame_id))
        self.get_logger().info(
            "Published empty cloud to clear costmap on shutdown")
        super().destroy_node()

    # -----------------------------------------------------------------
    # Geometry primitives
    # -----------------------------------------------------------------

    def make_disk_points(self, cx, cy, radius=0.4, spacing=0.1, z=0.3):
        """Filled circle, sampled on a square grid."""
        points = []
        steps = int(radius / spacing)
        r2 = radius ** 2
        for ix in range(-steps, steps + 1):
            for iy in range(-steps, steps + 1):
                x = cx + ix * spacing
                y = cy + iy * spacing
                if (x - cx) ** 2 + (y - cy) ** 2 <= r2:
                    points.append((x, y, z))
        return points

    def make_ellipse_points(self, cx, cy, heading, a=1.5, b=0.5,
                            spacing=0.15, z=0.3):
        """Filled ellipse, long axis a along heading, short axis b across.

        Unlike make_disk_points this region is not rotationally
        symmetric, which is the whole point: it encodes direction of
        travel as an occupied lane rather than an undirected blob.
        """
        points = []
        cos_h, sin_h = math.cos(heading), math.sin(heading)
        steps_u = int(a / spacing)
        steps_v = int(b / spacing)
        for iu in range(-steps_u, steps_u + 1):
            u = iu * spacing
            for iv in range(-steps_v, steps_v + 1):
                v = iv * spacing
                if (u / a) ** 2 + (v / b) ** 2 <= 1.0:
                    points.append((cx + u * cos_h - v * sin_h,
                                   cy + u * sin_h + v * cos_h,
                                   z))
        return points

    def create_cloud(self, points, frame_id):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id

        cloud = PointCloud2()
        cloud.header = header
        cloud.height = 1
        cloud.width = len(points)
        cloud.fields = [
            PointField(name="x", offset=0,
                       datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4,
                       datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8,
                       datatype=PointField.FLOAT32, count=1),
        ]
        cloud.is_bigendian = False
        cloud.point_step = 12
        cloud.row_step = cloud.point_step * len(points)
        cloud.data = b"".join(struct.pack("fff", x, y, z) for x, y, z in points)
        cloud.is_dense = True
        return cloud


def main(args=None):
    rclpy.init(args=args)
    node = PredictedPersonCloudNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()