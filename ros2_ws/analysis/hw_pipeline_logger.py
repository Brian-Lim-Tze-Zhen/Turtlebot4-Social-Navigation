#!/usr/bin/env python3
"""hw_pipeline_logger.py <out.csv> - log the perception pipeline's outputs during a
hardware-bag replay (see hw_pipeline_replay.sh).

Rows: "<topic>,<sim t>,<robot x>,<robot y>,<robot yaw>,<message data>"
The robot pose is map -> base_link at the time the message arrives ("nan" if
TF has no map frame, as in bags recorded without localisation).
"""
import math
import sys

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
import tf2_ros

TOPICS = ["/person_positions_map", "/camera_ray_clusters", "/person_positions_fused"]


class Logger(Node):
    def __init__(self, path):
        super().__init__("hw_pipeline_logger")
        self.f = open(path, "w")
        self.buf = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buf, self)
        for t in TOPICS:
            self.create_subscription(String, t, lambda m, t=t: self.on_msg(t, m), 50)

    def on_msg(self, topic, msg):
        now = self.get_clock().now().nanoseconds * 1e-9
        pose = "nan,nan,nan"
        for frame in ("map", "odom"):
            try:
                tr = self.buf.lookup_transform(frame, "base_link", Time()).transform
                q = tr.rotation
                yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
                pose = f"{tr.translation.x:.3f},{tr.translation.y:.3f},{yaw:.4f}"
                break
            except Exception:
                continue
        self.f.write(f"{topic},{now:.3f},{pose},{msg.data}\n")
        self.f.flush()


def main():
    rclpy.init()
    node = Logger(sys.argv[1])
    try:
        rclpy.spin(node)
    except Exception:
        pass


if __name__ == "__main__":
    main()
