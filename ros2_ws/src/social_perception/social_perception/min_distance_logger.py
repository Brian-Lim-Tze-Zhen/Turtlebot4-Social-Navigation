#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
import tf2_ros

class MinDistanceLogger(Node):
    def __init__(self):
        super().__init__("min_distance_logger")
        self.person_xy = None
        self.min_dist = float("inf")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.create_subscription(PoseArray, "/person_ground_truth", self.cb, 10)
        self.create_timer(0.1, self.tick)
        self.get_logger().info("Min-distance logger started (robot base_link vs /person_ground_truth)")

    def cb(self, msg):
        if msg.poses:
            self.person_xy = (msg.poses[0].position.x, msg.poses[0].position.y)

    def tick(self):
        if self.person_xy is None:
            return
        try:
            t = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
        except Exception:
            return
        rx = t.transform.translation.x
        ry = t.transform.translation.y
        d = math.hypot(rx - self.person_xy[0], ry - self.person_xy[1])
        if d < self.min_dist:
            self.min_dist = d
        self.get_logger().info(f"dist={d:.3f} m | min so far={self.min_dist:.3f} m")

def main(args=None):
    rclpy.init(args=args)
    node = MinDistanceLogger()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        print(f"=== TRIAL MIN DISTANCE: {node.min_dist:.3f} m ===")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
