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
ELLIPSE_B = 1.2               # m

# Ellipse half-length ALONG the heading, as a function of walking speed.
# At the 1.2 m/s test speed this gives a = 0.60 + 0.60 = 1.20 m, so the
# marked lane runs ~2.4 m end to end.
ELLIPSE_A_BASE = 0.60              # m
ELLIPSE_A_SLOPE = 0.50             # m per (m/s)
ELLIPSE_A_MAX = 3.00               # m

# Perpendicular offset of the lane, breaking head-on left/right symmetry
# deterministically (social "keep right"). Ratio to ELLIPSE_B is
# 0.40/1.20 = 0.33.
LATERAL_BIAS = 0.4  # m

# Person footprint marked at the current position, and the fallback disk
# used when the ellipse is suppressed. Same radius, different sampling:
# the current-position disk is denser because it is what the local
# planner collides against.
PERSON_DISK_RADIUS = 0.55    # m
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
HISTORY_LENGTH = 60                # positions retained per track

# Trail clearing. Derived from the lane geometry so it stays correct if
# the lane changes: the outermost ellipse mark sits at
# LATERAL_BIAS + ELLIPSE_B from the person's path.
TRAIL_CLEAR_MARGIN = 0.20          # m
TRAIL_LAG_MARGIN = 0.20            # m

PERSON_DISK_FORWARD = 0.0   # m，圆盘沿行进方向的前移量

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
        }

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
        if math.hypot(dx, dy) > 0.01:
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
            points.extend(self._ellipse_points(current_x, current_y, heading, speed))
        else:
            points.extend(self._fallback_disk_points(predicted_x, predicted_y, heading))
        return points

    def _ellipse_points(self, current_x, current_y, heading, speed):
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
        cx += LATERAL_BIAS * math.sin(heading)
        cy += LATERAL_BIAS * -math.cos(heading)

        return self.make_ellipse_points(
            cx, cy, heading=heading, a=a, b=ELLIPSE_B,
            spacing=ELLIPSE_SPACING, z=0.3)

    def _fallback_disk_points(self, predicted_x, predicted_y, heading):
        cx, cy = predicted_x, predicted_y
        if heading is not None:
            # Same side as the ellipse. Zeroing the heading while
            # rotation-gated made the pass-side bias vanish at exactly
            # the moment it mattered most - swerve onset.
            cx += LATERAL_BIAS * math.sin(heading)
            cy += LATERAL_BIAS * -math.cos(heading)
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

        points = []
        for track in self.active_tracks.values():
            self._clear_trail(track)
            points.extend(self._track_points(track))

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