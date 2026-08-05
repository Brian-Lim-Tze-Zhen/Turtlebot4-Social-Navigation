#!/usr/bin/env python3

import math
import struct

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

import tf2_ros

from std_msgs.msg import String, Header
from sensor_msgs.msg import PointCloud2, PointField


# Below this speed a person is treated as stationary and marked with a
# symmetric disk rather than a directional ellipse. See the deadband
# comment in publish_cloud() for the measurements behind this value.
STATIONARY_SPEED_DEADBAND = 0.05   # m/s


class PredictedPersonCloudNode(Node):
    def __init__(self):
        super().__init__("predicted_person_cloud_node")

        self.sub = self.create_subscription(
            String,
            "/predicted_person_positions",
            self.callback,
            10
        )

        self.pub = self.create_publisher(
            PointCloud2,
            "/predicted_person_cloud",
            10
        )

        self.frame_id = "map"

        # ==========================================================
        # THESIS MODIFICATION (multi-track fix)
        #
        # Previously this node rebuilt and published a brand new
        # point cloud on every incoming message, containing only the
        # single track_id from that message. With multiple people in
        # the scene, each person's update would overwrite the
        # previous person's obstacle cloud in the costmap, so only
        # one person was ever visible to Nav2 at a time.
        #
        # Fix: store the latest state per track_id in a dict, and
        # publish a single point cloud built from ALL currently
        # active tracks on a fixed timer. Stale tracks (no update
        # for > track_timeout seconds, e.g. person left camera FOV
        # or ID was lost) are pruned automatically, so the costmap
        # doesn't keep a phantom obstacle forever.
        # ==========================================================
        self.active_tracks = {}       # track_id -> dict(current, predicted, last_seen, heading_sin, heading_cos)
        self.track_timeout = 0.5        # seconds before a silent track is dropped
        self.publish_rate_hz = 10.0   # cloud publish rate, decoupled from detection rate

        # ==========================================================
        # THESIS MODIFICATION (heading smoothing)
        #
        # The ellipse heading was previously recomputed fresh every
        # tick from atan2(predicted - current). This caused the
        # ellipse to snap instantly when the person reverses, a noisy
        # depth reading shifts the predicted point, or ego-motion from
        # robot rotation contaminates the KF velocity.
        #
        # Fix: apply EMA smoothing to the heading using circular mean
        # interpolation (sin/cos components separately) so the ellipse
        # rotates gradually rather than snapping. Lower alpha = slower
        # smoother rotation; higher alpha = faster response.
        # ==========================================================
        self.heading_smooth_alpha = 0.40

        # ==========================================================
        # THESIS MODIFICATION (robot keep-out filter)
        #
        # In a head-on encounter, the ellipse points directly at the
        # approaching robot. Once the robot-person gap closes below
        # the ellipse's forward extent (plus costmap inflation), the
        # synthetic obstacle points land ON the robot's own footprint.
        # The robot is then standing inside lethal/high cost created
        # by its own prediction layer, and the controller can no
        # longer score a valid forward trajectory -> it oscillates
        # jerkily between "path blocked" and "path clear" states, or
        # sinks into the inflation and stalls.
        #
        # Fix: before publishing, drop every cloud point that falls
        # within robot_keepout_radius of the robot's current position
        # (looked up from TF map->base_link). The radius is chosen to
        # be just larger than the TurtleBot4 footprint, so lethal
        # marks can never appear under the robot, while the remaining
        # ellipse/disk points (and their inflation gradient) still
        # repel the robot sideways as intended. Making this radius
        # much larger would carve a moving "hole" through the risk
        # zone and let the robot push straight through the person's
        # lane, so keep it footprint-sized.
        #
        # If the TF lookup fails on a given cycle (startup, TF lag),
        # the last known robot position is reused; if none is known
        # yet, the cloud is published unfiltered (original behavior).
        # ==========================================================
        self.robot_frame = "base_link"
        self.robot_keepout_radius = 0.25
        self.last_robot_xy = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.publish_timer = self.create_timer(
            1.0 / self.publish_rate_hz,
            self.publish_cloud
        )

        self.get_logger().info("Predicted person cloud node started")
        self.get_logger().info("Subscribing: /predicted_person_positions")
        self.get_logger().info("Publishing: /predicted_person_cloud")
        self.get_logger().info(f"Cloud frame: {self.frame_id}")
        self.get_logger().info(
            f"Track timeout: {self.track_timeout:.2f}s | "
            f"Publish rate: {self.publish_rate_hz:.1f} Hz"
        )
        self.get_logger().info(
            f"Robot keep-out: {self.robot_keepout_radius:.2f} m "
            f"around frame '{self.robot_frame}'"
        )

    def get_ros_time_seconds(self):
        # Uses the node's ROS clock, which respects use_sim_time.
        # Falls back correctly to wall clock if use_sim_time is false.
        return self.get_clock().now().nanoseconds / 1e9

    def callback(self, msg):
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

        # Field [9] is human_kf_predictor's rotation-gate flag: it is 1
        # when |odom angular.z| exceeded rot_gate_threshold, meaning the
        # KF velocity EMA was frozen because the robot's own rotation was
        # contaminating the camera-derived position estimate.
        rotation_gated = len(parts) > 9 and parts[9].strip() == "1"

        # Compute raw heading from current -> predicted displacement.
        # If displacement is zero (stationary / warm-up), heading is undefined
        # so we skip the heading update and keep the last known value.
        dx = predicted_x - current_x
        dy = predicted_y - current_y
        dist = math.hypot(dx, dy)

        existing = self.active_tracks.get(track_id)

        if dist > 0.01:
            raw_heading = math.atan2(dy, dx)
            raw_sin = math.sin(raw_heading)
            raw_cos = math.cos(raw_heading)

            # ==========================================================
            # THESIS MODIFICATION (heading smoothing)
            #
            # EMA on sin/cos components keeps the interpolation on the
            # unit circle (avoids the discontinuity at ±pi that would
            # cause the heading to spin the long way around on reversal).
            # The smoothed heading is recovered via atan2(sin, cos).
            # ==========================================================
            if existing is not None and "heading_sin" in existing:
                a = self.heading_smooth_alpha
                s = a * raw_sin + (1.0 - a) * existing["heading_sin"]
                c = a * raw_cos + (1.0 - a) * existing["heading_cos"]
            else:
                s, c = raw_sin, raw_cos
        else:
            # No displacement — keep last known heading if available
            if existing is not None and "heading_sin" in existing:
                s = existing["heading_sin"]
                c = existing["heading_cos"]
            else:
                s, c = 0.0, 1.0  # default: east

        # Just record the latest state for this track. Actual cloud
        # construction/publishing happens in publish_cloud() so that
        # all active tracks are represented together, not just the
        # one that happened to publish most recently.
        self.active_tracks[track_id] = {
            "current": (current_x, current_y),
            "predicted": (predicted_x, predicted_y),
            "speed": math.hypot(vx, vy),
            "rotation_gated": rotation_gated,
            "last_seen": self.get_ros_time_seconds(),
            "heading_sin": s,
            "heading_cos": c,
        }

    def publish_cloud(self):
        now = self.get_ros_time_seconds()

        # Prune tracks that have gone silent (person out of FOV,
        # occluded, or ID lost). Without this, an obstacle would
        # freeze in the costmap forever at the last known position.
        stale_ids = [
            tid for tid, t in self.active_tracks.items()
            if now - t["last_seen"] > self.track_timeout
        ]
        for tid in stale_ids:
            del self.active_tracks[tid]
            self.get_logger().info(f"Pruned stale track id:{tid} from obstacle cloud")

        points = []

        for tid, t in self.active_tracks.items():
            current_x, current_y = t["current"]
            predicted_x, predicted_y = t["predicted"]

            # ==========================================================
            # THESIS MODIFICATION
            #
            # Convert current human position into a small obstacle region
            # instead of a single point.
            #
            # This improves costmap visibility and makes the current
            # pedestrian position more robust inside the Nav2 costmap.
            # ==========================================================
            points.extend(
                self.make_disk_points(
                    current_x,
                    current_y,
                    radius=0.40,
                    spacing=0.10,
                    z=0.3
                )
            )

            # ==========================================================
            # THESIS MODIFICATION
            #
            # Convert predicted human position into a future risk zone,
            # shaped as an ELLIPSE oriented along the direction of travel
            # (current -> predicted) instead of a symmetric disk.
            #
            # Rationale: a symmetric disk inflates the costmap equally in
            # every direction around the predicted point, which pushes the
            # robot away from that point but gives no preference for going
            # *behind* the person. An ellipse elongated along the heading
            # represents the "occupied lane" the person is walking through:
            #
            #   - long axis (a)  -> extends ahead/behind along the heading,
            #                       so the robot anticipates further into
            #                       the person's path of travel.
            #   - short axis (b) -> kept narrow across the heading, so the
            #                       costmap does not over-inflate sideways
            #                       and the robot can pass close behind the
            #                       person instead of stopping or detouring
            #                       far around them.
            #
            # Keep both axes conservative. Too large a long axis may block
            # the local planner and prevent the robot from reaching a goal;
            # too large a short axis removes the "pass behind" gap entirely.
            #
            # If the person has no measurable displacement this tick
            # (current == predicted, e.g. stationary or track just
            # started), heading is undefined, so we fall back to a small
            # symmetric disk for that one cycle.
            # ==========================================================
            # ==========================================================
            # THESIS MODIFICATION (rotation gate + stationary deadband)
            #
            # Two independent reasons to fall back to the symmetric disk
            # instead of the directional ellipse:
            #
            # 1. Rotation gate (main): human_kf_predictor freezes its
            #    velocity EMA while the robot rotates fast enough that
            #    ego-motion contaminates the camera-derived position
            #    estimate (field [9] on /predicted_person_positions).
            #    The heading carried in this track is stale/frozen and
            #    should not drive a directional obstacle right now.
            #
            # 2. Stationary speed deadband (feature): the old trigger was
            #    exact float equality (dx == 0.0 and dy == 0.0), which
            #    effectively never fired - camera jitter always produces
            #    some non-zero displacement. Measured noise floor with
            #    people bolted in place in conversation_test.sdf: max
            #    0.028 m/s gentle motion, 0.040 m/s with robot rotating.
            #    STATIONARY_SPEED_DEADBAND=0.05 sits an order of
            #    magnitude above that noise and well below a genuine
            #    ~0.3 m/s walking pace.
            #
            # Either condition forces the disk. When rotation-gated,
            # still bias the disk toward the last-smoothed heading
            # rather than zero - gating heading to 0.0 made the lateral
            # pass-side bias below disappear at exactly the moment it's
            # needed most (swerve onset), confirmed in testing.
            # ==========================================================
            rotation_gated = t.get("rotation_gated", False)
            stationary = t.get("speed", 0.0) < STATIONARY_SPEED_DEADBAND
            use_ellipse = not (rotation_gated or stationary)

            if "heading_sin" in t:
                heading = math.atan2(t["heading_sin"], t["heading_cos"])
            elif use_ellipse:
                heading = math.atan2(predicted_y - current_y, predicted_x - current_x)
            else:
                heading = None  # no prior heading yet; skip bias

            if not use_ellipse:
                disk_cx = predicted_x
                disk_cy = predicted_y

                if heading is not None:
                    lateral_bias = 0.20  # matches ellipse b=0.20 below
                    disk_cx += lateral_bias * math.sin(heading)
                    disk_cy += lateral_bias * -math.cos(heading)

                points.extend(
                    self.make_disk_points(
                        disk_cx,
                        disk_cy,
                        radius=0.40,
                        spacing=0.15,
                        z=0.3
                    )
                )
            else:
                # Ellipse is shifted forward by `a` along the heading so
                # its back edge starts at the current position, rather
                # than being centered on the predicted point - prevents
                # the ellipse from extending behind the person regardless
                # of walking speed. NOTE: thesis prose cites a=1.10/b=0.40; tested worse
                # for head-on - deployed value is 0.60/0.20.
                speed = t.get("speed", 0.0)
                a = min(1.6, 0.60 + speed * 0.5)  # TEST: speed-scaled reach, 1.2 m/s fast-person scenario
                b = 0.20
                ellipse_cx = current_x + a * math.cos(heading)
                ellipse_cy = current_y + a * math.sin(heading)

                # ==========================================================
                # THESIS MODIFICATION (head-on symmetry break / pass-right bias)
                #
                # A perfectly head-on approach (person's heading points
                # straight at the robot) makes the ellipse symmetric
                # left-right around the heading axis, giving MPPI's cost
                # gradient no directional preference. Observed in testing:
                # visible hesitation/wobbling and a late, close-range
                # (<0.3 m) avoidance decision, since the optimizer had to
                # wait on sampling noise to break the tie.
                #
                # Fix: shift the ellipse center slightly perpendicular to
                # heading, consistently to one side (social "keep right"
                # convention, matching pedestrian/traffic norms). This
                # breaks left/right symmetry deterministically and early,
                # so MPPI resolves the pass-side decision well before the
                # robot is close.
                #
                # perp_x, perp_y = heading rotated -90 deg (clockwise) =
                # the right-hand side relative to the person's direction
                # of travel. lateral_bias matched to the current b=0.20
                # (roughly one full short-axis width) — large enough to
                # clearly favor one side, but re-tune if the pass-behind
                # gap feels too tight or the bias feels too weak to
                # resolve hesitation. Must be updated together with the
                # disk-fallback lateral_bias above whenever b changes, so
                # the bias stays consistent regardless of whether the
                # ellipse or the rotation-gated disk fallback is active
                # on a given cycle.
                # ==========================================================
                lateral_bias = 0.20
                perp_x = math.sin(heading)
                perp_y = -math.cos(heading)
                ellipse_cx += lateral_bias * perp_x
                ellipse_cy += lateral_bias * perp_y

                points.extend(
                    self.make_ellipse_points(
                        ellipse_cx,
                        ellipse_cy,
                        heading=heading,
                        a=a,
                        b=b,
                        spacing=0.15,
                        z=0.3
                    )
                )

        # ==========================================================
        # THESIS MODIFICATION (robot keep-out filter) - see __init__
        # ==========================================================
        try:
            tfm = self.tf_buffer.lookup_transform(
                self.frame_id,
                self.robot_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.05)
            )
            self.last_robot_xy = (
                tfm.transform.translation.x,
                tfm.transform.translation.y,
            )
        except Exception:
            # TF momentarily unavailable: reuse last known robot pose.
            pass

        if self.last_robot_xy is not None and points:
            rx, ry = self.last_robot_xy
            r2 = self.robot_keepout_radius ** 2
            n_before = len(points)
            points = [
                p for p in points
                if (p[0] - rx) ** 2 + (p[1] - ry) ** 2 > r2
            ]
            n_removed = n_before - len(points)
            if n_removed > 0:
                self.get_logger().info(
                    f"Keep-out: removed {n_removed} point(s) within "
                    f"{self.robot_keepout_radius:.2f} m of robot"
                )

        cloud = self.create_cloud(points, self.frame_id)
        self.pub.publish(cloud)

        if self.active_tracks:
            ids_str = ",".join(str(tid) for tid in self.active_tracks.keys())
            gated_ids = [
                str(tid) for tid, t in self.active_tracks.items()
                if t.get("rotation_gated", False)
            ]
            gated_str = f" [ROT GATED: {','.join(gated_ids)}]" if gated_ids else ""
            self.get_logger().info(
                f"Published cloud for {len(self.active_tracks)} track(s) "
                f"[{ids_str}] points={len(points)}{gated_str}"
            )

    def destroy_node(self):
        empty_cloud = self.create_cloud([], self.frame_id)
        self.pub.publish(empty_cloud)
        self.get_logger().info("Published empty cloud to clear costmap on shutdown")
        super().destroy_node()

    # ==============================================================
    # THESIS MODIFICATION
    #
    # Generate a circular obstacle region around a given position.
    #
    # This converts a single human position into an executable
    # PointCloud2 obstacle area for Nav2 costmap integration.
    #
    # radius:
    #   Obstacle/risk radius around the human position.
    #
    # spacing:
    #   Distance between generated points inside the disk.
    # ==============================================================
    def make_disk_points(self, cx, cy, radius=0.4, spacing=0.1, z=0.3):
        points = []

        steps = int(radius / spacing)

        for ix in range(-steps, steps + 1):
            for iy in range(-steps, steps + 1):
                x = cx + ix * spacing
                y = cy + iy * spacing

                if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                    points.append((x, y, z))

        return points

    # ==============================================================
    # THESIS MODIFICATION
    #
    # Generate an elliptical obstacle region around a given position,
    # oriented along a heading direction (radians).
    #
    # Unlike make_disk_points, this region is NOT rotationally
    # symmetric: it is elongated along 'heading' and narrow across it.
    # This is used for the predicted/future-risk zone so that the
    # costmap encodes the person's direction of travel as an occupied
    # "lane", instead of an undirected blob. This lets the local
    # planner route the robot behind the person rather than just
    # nudging it sideways away from a point.
    #
    # cx, cy   : ellipse center (the predicted position)
    # heading  : direction of travel, radians, from atan2(dy, dx)
    #            where (dx, dy) = predicted - current
    # a        : semi-axis length ALONG heading (ahead/behind)
    # b        : semi-axis length ACROSS heading (left/right)
    # spacing  : approximate distance between sampled grid points
    # z        : height to publish points at
    # ==============================================================
    def make_ellipse_points(self, cx, cy, heading, a=1.5, b=0.5, spacing=0.15, z=0.3):
        points = []

        cos_h = math.cos(heading)
        sin_h = math.sin(heading)

        steps_u = int(a / spacing)
        steps_v = int(b / spacing)

        for iu in range(-steps_u, steps_u + 1):
            u = iu * spacing
            for iv in range(-steps_v, steps_v + 1):
                v = iv * spacing

                if (u / a) ** 2 + (v / b) ** 2 <= 1.0:
                    # rotate local (u, v) into world frame using heading
                    x = cx + u * cos_h - v * sin_h
                    y = cy + u * sin_h + v * cos_h
                    points.append((x, y, z))

        return points

    def create_cloud(self, points, frame_id):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        data = b"".join([struct.pack("fff", x, y, z) for x, y, z in points])

        cloud = PointCloud2()
        cloud.header = header
        cloud.height = 1
        cloud.width = len(points)
        cloud.fields = fields
        cloud.is_bigendian = False
        cloud.point_step = 12
        cloud.row_step = cloud.point_step * len(points)
        cloud.data = data
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