#!/usr/bin/env python3

import math
import subprocess
import threading

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose


class MovePersonCombined(Node):
    def __init__(self):
        super().__init__("move_person_combined")

        self.world_name = "combined_scenario"
        self.model_name = "person_mover"

        # Movement endpoints in Gazebo world frame
        self.point_a = (2.0, 1.0, 0.0)
        self.point_b = (2.0, 7.0, 0.0)

        self.speed = 0.2          # m/s
        self.update_dt = 0.5      # seconds

        # Same endpoint-pause pattern as move_person_gazebo.py — gives
        # the KF time to register near-zero velocity before the
        # direction flip.
        self.pause_duration = 1.5  # seconds
        self.pause_timer = 0.0     # 0 = not currently pausing

        self.current_x = self.point_a[0]
        self.current_y = self.point_a[1]
        self.current_z = self.point_a[2]

        self.target = self.point_b

        self.timer = self.create_timer(self.update_dt, self.timer_callback)
        self.last_time = self.get_clock().now()

        self.ground_truth_pub = self.create_publisher(
            PoseArray,
            "/person_ground_truth",
            10
        )
        self.frame_id = "map"

        self.get_logger().info("Moving person_mover using Gazebo set_pose service")
        self.get_logger().info(f"World: {self.world_name}")
        self.get_logger().info(f"Endpoints: {self.point_a} <-> {self.point_b}")
        self.get_logger().info(f"Speed: {self.speed} m/s, pause: {self.pause_duration}s at endpoints")

    def timer_callback(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now

        dt = max(0.01, min(dt, self.update_dt * 3.0))

        if self.pause_timer > 0.0:
            self.pause_timer = max(0.0, self.pause_timer - dt)
            self.get_logger().info(
                f"Pausing at ({self.current_x:.2f},{self.current_y:.2f}) "
                f"— {self.pause_timer:.2f}s remaining"
            )
            self.publish_ground_truth()
            return

        target_x, target_y, target_z = self.target

        dx = target_x - self.current_x
        dy = target_y - self.current_y

        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 0.05:
            self.pause_timer = self.pause_duration
            if self.target == self.point_b:
                self.target = self.point_a
            else:
                self.target = self.point_b
            self.get_logger().info(
                f"Reached endpoint, pausing {self.pause_duration}s "
                f"then switching to {self.target}"
            )
            self.publish_ground_truth()
            return

        step = min(self.speed * dt, dist)

        ux = dx / dist
        uy = dy / dist

        self.current_x += ux * step
        self.current_y += uy * step

        self.get_logger().info(
            f"dt={dt:.3f}s step={step:.3f}m "
            f"pos=({self.current_x:.2f},{self.current_y:.2f}) "
            f"target={self.target}"
        )

        # +pi/2 offset to correct for person_standing model's default
        # mesh orientation — same calibration as move_person_gazebo.py.
        yaw = math.atan2(uy, ux) + math.pi / 2.0
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))

        threading.Thread(
            target=self.set_model_pose,
            args=(self.current_x, self.current_y, self.current_z, yaw),
            daemon=True,
        ).start()

        self.publish_ground_truth()

    def publish_ground_truth(self):
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        pose = Pose()
        pose.position.x = self.current_x
        pose.position.y = self.current_y
        pose.position.z = self.current_z
        msg.poses.append(pose)

        self.ground_truth_pub.publish(msg)

    def set_model_pose(self, x, y, z, yaw):
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)

        service_name = f"/world/{self.world_name}/set_pose"

        req = (
            f"name: '{self.model_name}', "
            f"position: {{x: {x}, y: {y}, z: {z}}}, "
            f"orientation: {{x: 0, y: 0, z: {qz}, w: {qw}}}"
        )

        cmd = [
            "gz", "service",
            "-s", service_name,
            "--reqtype", "gz.msgs.Pose",
            "--reptype", "gz.msgs.Boolean",
            "--timeout", "3000",
            "--req", req
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=6.0
            )
            if result.returncode != 0:
                self.get_logger().warn(f"set_pose failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            self.get_logger().warn("set_pose service timeout")

    def destroy_node(self):
        self.get_logger().info("Stopping move_person_combined node")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MovePersonCombined()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()