#!/usr/bin/env python3
"""
person_marker_publisher.py

Visualization-only node: subscribes to the raw LIDAR clusters and the
fused stable-id output, republishes both as RViz MarkerArray so you can
SEE detections overlaid on the map instead of reading log lines.

Add a "MarkerArray" display in RViz, topic /person_markers, fixed frame
"map" - you should see:
  - small blue dots  = raw /lidar_person_clusters (every candidate that
                        passed the width filter and is moving)
  - green spheres     = /person_positions_fused, source=camera_confirmed
  - orange spheres    = /person_positions_fused, source=lidar_only
                        (camera-occluded, coasting on LIDAR alone)
  - text label above each fused sphere showing its stable_id

What to look for:
  - Number of green/orange spheres should match the number of real
    people in the room, not the number of raw blue dots (which will
    include some clutter/leg-pair splits).
  - Each real person's ID label should stay the SAME as they walk
    around and briefly leave camera view (orange), not jump to a new
    number.
  - Spheres should track smoothly with actual foot position, not jump
    erratically between scans.
"""

import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


MARKER_LIFETIME_SEC = 2.0  # markers vanish this long after last update
                             # so stale/dropped tracks disappear from view


class PersonMarkerPublisher(Node):
    def __init__(self):
        super().__init__("person_marker_publisher")

        self.declare_parameter("lidar_topic", "/lidar_person_clusters")
        self.declare_parameter("fused_topic", "/person_positions_fused")
        self.declare_parameter("output_topic", "/person_markers")
        self.declare_parameter("frame_id", "map")

        self.lidar_topic = self.get_parameter("lidar_topic").value
        self.fused_topic = self.get_parameter("fused_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.frame_id = self.get_parameter("frame_id").value

        self.create_subscription(String, self.lidar_topic, self.lidar_cb, 10)
        self.create_subscription(String, self.fused_topic, self.fused_cb, 10)
        self.pub = self.create_publisher(MarkerArray, self.output_topic, 10)

        self.get_logger().info("Person marker publisher started")
        self.get_logger().info(f"Lidar in : {self.lidar_topic}")
        self.get_logger().info(f"Fused in : {self.fused_topic}")
        self.get_logger().info(f"Markers out: {self.output_topic}")
        self.get_logger().info(
            "Add a MarkerArray display in RViz on this topic, "
            f"fixed frame '{self.frame_id}'")

    def lifetime(self):
        d = rclpy.duration.Duration(seconds=MARKER_LIFETIME_SEC)
        return d.to_msg()

    # -----------------------------------------------------------------
    def lidar_cb(self, msg):
        parts = msg.data.split(",")
        if len(parts) < 4:
            return
        try:
            lid = int(float(parts[0]))
            x = float(parts[2])
            y = float(parts[3])
        except ValueError:
            return

        m = Marker()
        m.header.frame_id = self.frame_id
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "lidar_raw"
        m.id = lid
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = 0.05
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.12
        m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 0.4, 1.0, 0.6
        m.lifetime = self.lifetime()

        arr = MarkerArray()
        arr.markers.append(m)
        self.pub.publish(arr)

    # -----------------------------------------------------------------
    def fused_cb(self, msg):
        parts = msg.data.split(",")
        if len(parts) < 12:
            return
        try:
            stable_id = int(float(parts[0]))
            x = float(parts[2])
            y = float(parts[3])
        except ValueError:
            return
        source = parts[11]

        # THESIS MODIFICATION: colour by stable id, not by source.
        # Two overlapping identities rendered in one colour were
        # indistinguishable in RViz (10 Sep: a wall artefact and the
        # real person both drew the same colour). Source is still
        # readable in the text label. Golden-ratio hue hash keeps
        # adjacent ids far apart in colour; value is dimmed for
        # lidar_only so the source distinction survives at a glance.
        import colorsys
        hue = (stable_id * 0.61803398875) % 1.0
        val = 1.0 if source == "camera_confirmed" else 0.65
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, val)

        sphere = Marker()
        sphere.header.frame_id = self.frame_id
        sphere.header.stamp = self.get_clock().now().to_msg()
        sphere.ns = "fused_person"
        sphere.id = stable_id
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position.x = x
        sphere.pose.position.y = y
        sphere.pose.position.z = 0.15
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.25
        sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = r, g, b, 0.9
        sphere.lifetime = self.lifetime()

        label = Marker()
        label.header.frame_id = self.frame_id
        label.header.stamp = self.get_clock().now().to_msg()
        label.ns = "fused_person_label"
        label.id = stable_id
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.x = x
        label.pose.position.y = y
        label.pose.position.z = 0.6
        label.pose.orientation.w = 1.0
        label.scale.z = 0.25
        label.color.r, label.color.g, label.color.b, label.color.a = r, g, b, 1.0
        cam_txt = ""
        if len(parts) >= 13 and parts[12].strip() not in ("", "-1"):
            cam_txt = f" cam:{parts[12].strip()}"
        label.text = f"id:{stable_id}{cam_txt} ({source})"
        label.lifetime = self.lifetime()

        arr = MarkerArray()
        arr.markers.append(sphere)
        arr.markers.append(label)
        self.pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = PersonMarkerPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
