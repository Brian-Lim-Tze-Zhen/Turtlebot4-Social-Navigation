#!/usr/bin/env python3

import math
import struct

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Header
from sensor_msgs.msg import PointCloud2, PointField


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
        self.active_tracks = {}       # track_id -> dict(current, predicted, last_seen)
        self.track_timeout = 0.3        # seconds before a silent track is dropped
        self.publish_rate_hz = 10.0   # cloud publish rate, decoupled from detection rate

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

            predicted_x = float(parts[6])
            predicted_y = float(parts[7])

        except ValueError:
            self.get_logger().warn(f"Parse failed: {msg.data}")
            return

        # Just record the latest state for this track. Actual cloud
        # construction/publishing happens in publish_cloud() so that
        # all active tracks are represented together, not just the
        # one that happened to publish most recently.
        self.active_tracks[track_id] = {
            "current": (current_x, current_y),
            "predicted": (predicted_x, predicted_y),
            "last_seen": self.get_ros_time_seconds(),
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
                    radius=0.20,
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
            dx = predicted_x - current_x
            dy = predicted_y - current_y

            if dx == 0.0 and dy == 0.0:
                points.extend(
                    self.make_disk_points(
                        predicted_x,
                        predicted_y,
                        radius=0.30,
                        spacing=0.15,
                        z=0.3
                    )
                )
            else:
                heading = math.atan2(dy, dx)
                points.extend(
                    self.make_ellipse_points(
                        predicted_x,
                        predicted_y,
                        heading=heading,
                        a=0.80,   # along direction of travel
                        b=0.40,   # across direction of travel (kept narrow)
                        spacing=0.15,
                        z=0.3
                    )
                )

        cloud = self.create_cloud(points, self.frame_id)
        self.pub.publish(cloud)

        if self.active_tracks:
            ids_str = ",".join(str(tid) for tid in self.active_tracks.keys())
            self.get_logger().info(
                f"Published cloud for {len(self.active_tracks)} track(s) "
                f"[{ids_str}] points={len(points)}"
            )

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