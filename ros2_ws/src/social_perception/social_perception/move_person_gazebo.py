#!/usr/bin/env python3

import math
import subprocess

import rclpy
from rclpy.node import Node


class MovePersonGazebo(Node):
    def __init__(self):
        super().__init__("move_person_gazebo")

        self.world_name = "empty_human"
        self.model_name = "person_1"

        # Movement endpoints in Gazebo world frame
        self.point_a = (3.0, 0.0, 0.0)
        self.point_b = (5.0, 0.0, 0.0)

        self.speed = 0.2          # m/s
        self.update_dt = 1.0     # seconds

        self.current_x = self.point_a[0]
        self.current_y = self.point_a[1]
        self.current_z = self.point_a[2]

        self.target = self.point_b

        self.timer = self.create_timer(self.update_dt, self.timer_callback)
        self.last_time = self.get_clock().now()

        self.get_logger().debug("Moving person_1 using Gazebo set_pose service")
        self.get_logger().debug(f"World: {self.world_name}")
        self.get_logger().debug(f"Model: {self.model_name}")
  

    def timer_callback(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now

        # Clamp dt to avoid huge jumps after lag or pause
        dt = max(0.01, min(dt, 0.5))

        target_x, target_y, target_z = self.target

        dx = target_x - self.current_x
        dy = target_y - self.current_y

        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 0.05:
            if self.target == self.point_b:
                self.target = self.point_a
            else:
                self.target = self.point_b

            self.get_logger().info(f"Switching target to {self.target}")
            return

        step = self.speed * dt

        # Avoid overshooting the target
        step = min(step, dist)

        ux = dx / dist
        uy = dy / dist

        self.current_x += ux * step
        self.current_y += uy * step

        self.get_logger().info(
            f"dt={dt:.3f}s step={step:.3f}m "
            f"pos=({self.current_x:.2f},{self.current_y:.2f}) "
            f"target={self.target}"
        )
        # Face movement direction
        yaw = math.atan2(uy, ux)

        self.set_model_pose(
            self.current_x,
            self.current_y,
            self.current_z,
            yaw
        )
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
        self.get_logger().info("Stopping move_person_gazebo node")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = MovePersonGazebo()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()