#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


class PredictionMarkerNode(Node):
    def __init__(self):
        super().__init__("prediction_marker_node")

        # =====================================
        # User configurable parameters
        # =====================================
        self.declare_parameter("input_topic", "/predicted_person_positions")
        self.declare_parameter("output_topic", "/predicted_person_markers")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("marker_lifetime", 0.5)
        self.declare_parameter("risk_radius", 0.8)
        self.declare_parameter("marker_timeout", 1.0)

        # =====================================
        # Load parameters
        # =====================================
        self.input_topic = (
            self.get_parameter("input_topic")
            .get_parameter_value()
            .string_value
        )

        self.output_topic = (
            self.get_parameter("output_topic")
            .get_parameter_value()
            .string_value
        )

        self.frame_id = (
            self.get_parameter("frame_id")
            .get_parameter_value()
            .string_value
        )

        self.marker_lifetime = (
            self.get_parameter("marker_lifetime")
            .get_parameter_value()
            .double_value
        )

        self.risk_radius = (
            self.get_parameter("risk_radius")
            .get_parameter_value()
            .double_value
        )

        self.marker_timeout = (
            self.get_parameter("marker_timeout")
            .get_parameter_value()
            .double_value
        )

        self.last_msg_time = None

        self.sub = self.create_subscription(
            String,
            self.input_topic,
            self.callback,
            10
        )

        self.pub = self.create_publisher(
            MarkerArray,
            self.output_topic,
            10
        )

        self.clear_timer = self.create_timer(0.2, self.clear_stale_markers)

        self.get_logger().info("Prediction marker node started")
        self.get_logger().info(f"Subscribing: {self.input_topic}")
        self.get_logger().info(f"Publishing : {self.output_topic}")
        self.get_logger().info(f"Frame      : {self.frame_id}")
        self.get_logger().info(f"Lifetime   : {self.marker_lifetime:.2f} s")
        self.get_logger().info(f"Timeout    : {self.marker_timeout:.2f} s")

    def get_ros_time_seconds(self):
        return self.get_clock().now().nanoseconds / 1e9

    def set_marker_lifetime(self, marker):
        lifetime = max(0.0, self.marker_lifetime)
        marker.lifetime.sec = int(lifetime)
        marker.lifetime.nanosec = int((lifetime - int(lifetime)) * 1e9)

    def clear_stale_markers(self):
        if self.last_msg_time is None:
            return

        now = self.get_ros_time_seconds()

        if now - self.last_msg_time <= self.marker_timeout:
            return

        clear_array = MarkerArray()

        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL

        clear_array.markers.append(clear_marker)
        self.pub.publish(clear_array)

        self.last_msg_time = None
        self.get_logger().info("Cleared stale prediction markers")

    def callback(self, msg):
        parts = [part.strip() for part in msg.data.split(",")]

        if len(parts) < 9:
            self.get_logger().warn(
                f"Invalid prediction msg, expected 9 values: {msg.data}"
            )
            return

        try:
            track_id = int(float(parts[0]))
            confidence = float(parts[1])

            current_x = float(parts[2])
            current_y = float(parts[3])

            vx = float(parts[4])
            vy = float(parts[5])

            predicted_x = float(parts[6])
            predicted_y = float(parts[7])

            horizon = float(parts[8])

        except ValueError:
            self.get_logger().warn(f"Parse failed: {msg.data}")
            return

        self.last_msg_time = self.get_ros_time_seconds()
        now = self.get_clock().now().to_msg() 
        marker_array = MarkerArray()

        base_id = track_id * 10 if track_id >= 0 else 0

        # -----------------------------
        # Current human position: blue sphere
        # -----------------------------
        current_marker = Marker()
        current_marker.header.frame_id = self.frame_id
        current_marker.header.stamp = now
        current_marker.ns = "current_human"
        current_marker.id = base_id + 1
        current_marker.type = Marker.SPHERE
        current_marker.action = Marker.ADD

        current_marker.pose.position.x = current_x
        current_marker.pose.position.y = current_y
        current_marker.pose.position.z = 0.3
        current_marker.pose.orientation.w = 1.0

        current_marker.scale.x = 0.35
        current_marker.scale.y = 0.35
        current_marker.scale.z = 0.35

        current_marker.color.r = 0.0
        current_marker.color.g = 0.2
        current_marker.color.b = 1.0
        current_marker.color.a = 1.0

        self.set_marker_lifetime(current_marker)
        marker_array.markers.append(current_marker)

        # -----------------------------
        # Predicted human position: red sphere
        # -----------------------------
        predicted_marker = Marker()
        predicted_marker.header.frame_id = self.frame_id
        predicted_marker.header.stamp = now
        predicted_marker.ns = "predicted_human"
        predicted_marker.id = base_id + 2
        predicted_marker.type = Marker.SPHERE
        predicted_marker.action = Marker.ADD

        predicted_marker.pose.position.x = predicted_x
        predicted_marker.pose.position.y = predicted_y
        predicted_marker.pose.position.z = 0.3
        predicted_marker.pose.orientation.w = 1.0

        predicted_marker.scale.x = 0.35
        predicted_marker.scale.y = 0.35
        predicted_marker.scale.z = 0.35

        predicted_marker.color.r = 1.0
        predicted_marker.color.g = 0.0
        predicted_marker.color.b = 0.0
        predicted_marker.color.a = 1.0

        self.set_marker_lifetime(predicted_marker)
        marker_array.markers.append(predicted_marker)

        # -----------------------------
        # Velocity / prediction arrow: green
        # -----------------------------
        arrow_marker = Marker()
        arrow_marker.header.frame_id = self.frame_id
        arrow_marker.header.stamp = now
        arrow_marker.ns = "human_prediction_arrow"
        arrow_marker.id = base_id + 3
        arrow_marker.type = Marker.ARROW
        arrow_marker.action = Marker.ADD

        start = Point()
        start.x = current_x
        start.y = current_y
        start.z = 0.55

        end = Point()
        end.x = predicted_x
        end.y = predicted_y
        end.z = 0.55

        arrow_marker.points.append(start)
        arrow_marker.points.append(end)

        arrow_marker.scale.x = 0.05
        arrow_marker.scale.y = 0.15
        arrow_marker.scale.z = 0.15

        arrow_marker.color.r = 0.0
        arrow_marker.color.g = 1.0
        arrow_marker.color.b = 0.0
        arrow_marker.color.a = 1.0

        self.set_marker_lifetime(arrow_marker)
        marker_array.markers.append(arrow_marker)

        # -----------------------------
        # Risk zone: transparent red cylinder
        # -----------------------------
        risk_marker = Marker()
        risk_marker.header.frame_id = self.frame_id
        risk_marker.header.stamp = now
        risk_marker.ns = "human_risk_zone"
        risk_marker.id = base_id + 4
        risk_marker.type = Marker.CYLINDER
        risk_marker.action = Marker.ADD

        risk_marker.pose.position.x = predicted_x
        risk_marker.pose.position.y = predicted_y
        risk_marker.pose.position.z = 0.05
        risk_marker.pose.orientation.w = 1.0

        risk_marker.scale.x = self.risk_radius * 2.0
        risk_marker.scale.y = self.risk_radius * 2.0
        risk_marker.scale.z = 0.1

        risk_marker.color.r = 1.0
        risk_marker.color.g = 0.0
        risk_marker.color.b = 0.0
        risk_marker.color.a = 0.25

        self.set_marker_lifetime(risk_marker)
        marker_array.markers.append(risk_marker)

        # -----------------------------
        # Text label
        # -----------------------------
        text_marker = Marker()
        text_marker.header.frame_id = self.frame_id
        text_marker.header.stamp = now
        text_marker.ns = "human_text"
        text_marker.id = base_id + 5
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD

        text_marker.pose.position.x = current_x
        text_marker.pose.position.y = current_y
        text_marker.pose.position.z = 1.0
        text_marker.pose.orientation.w = 1.0

        speed = math.sqrt(vx * vx + vy * vy)

        text_marker.text = (
            f"ID:{track_id}\n"
            f"conf:{confidence:.2f}\n"
            f"v:{speed:.2f} m/s\n"
            f"T:{horizon:.1f}s"
        )

        text_marker.scale.z = 0.25

        text_marker.color.r = 1.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0
        text_marker.color.a = 1.0

        self.set_marker_lifetime(text_marker)
        marker_array.markers.append(text_marker)

        self.pub.publish(marker_array)

        self.get_logger().info(
            f"ID:{track_id} current=({current_x:.2f},{current_y:.2f}) "
            f"pred=({predicted_x:.2f},{predicted_y:.2f}) "
            f"v=({vx:.2f},{vy:.2f})"
        )


def main(args=None):
    rclpy.init(args=args)
    node = PredictionMarkerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()