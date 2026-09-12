#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class ScanIndexDump(Node):
    def __init__(self):
        super().__init__("scan_index_dump")
        self.done = False
        self.sub = self.create_subscription(LaserScan, "/scan", self.cb, 10)

    def cb(self, msg):
        if self.done:
            return
        self.done = True
        n = len(msg.ranges)
        print(f"angle_min={msg.angle_min:.6f} angle_increment={msg.angle_increment:.6f} n_ranges={n}")
        for i in range(120, 200):
            print(i, msg.ranges[i])


def main():
    rclpy.init()
    node = ScanIndexDump()
    while rclpy.ok() and not node.done:
        rclpy.spin_once(node, timeout_sec=1.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
