#!/usr/bin/env python3
#
# move_person_crossing.py
#
# THESIS ADDITION (crossing scenario)
#
# Sibling of move_person_gazebo.py. Identical mechanism (Gazebo
# set_pose service, endpoint pause, ground-truth publishing), but the
# person walks ACROSS the robot's path instead of head-on toward it.
#
#   head-on   (move_person_gazebo.py):  (3,0)  <-> (6,0)   along +x
#   crossing  (this file):              (5,-2) <-> (5,2)   along +y
#
# The robot still navigates (0,0) -> (8,0), so the person's track
# intersects the robot's straight-line path at x=5.
#
# Kept as a separate file rather than parameterising the original so
# that the head-on scenario used for the recorded evaluation runs is
# frozen and cannot be accidentally altered while setting this up.
#
# NOTE ON YAW: the +pi/2 offset below corrects for the person_standing
# model's default mesh orientation and was verified for x-axis motion
# (x 3->5 => yaw=+pi/2). It is applied here unchanged, so the model
# should face its direction of travel along y as well - but VERIFY
# VISUALLY in RViz or a GUI session before recording, since the
# original verification was only done for x-axis motion.

import math
import subprocess
import threading

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose


class MovePersonCrossing(Node):
    def __init__(self):
        super().__init__("move_person_crossing")

        self.world_name = "empty_human"
        self.model_name = "person_1"

        # ==================================================
        # CROSSING GEOMETRY
        #
        # Person walks along the y-axis at fixed x=5.0, crossing the
        # robot's (0,0) -> (8,0) path perpendicularly.
        #
        # x=5.0 is chosen so the person is well beyond the robot's
        # 2.5 m lidar obstacle_max_range at the start of the run,
        # giving the prediction pipeline room to act before the
        # reactive baseline can see anything.
        # ==================================================
        self.point_a = (5.0, -2.0, 0.0)
        self.point_b = (5.0, 2.0, 0.0)

        self.speed = 0.2          # m/s - matches head-on scenario
        self.update_dt = 0.5      # seconds - matches head-on scenario

        # ==================================================
        # Endpoint pause (carried over from move_person_gazebo.py)
        #
        # Person pauses at each endpoint before reversing, giving the
        # KF time to register near-zero velocity before the direction
        # flip and reducing the reversal error spike. Kept identical
        # to the head-on scenario so the two are comparable.
        # ==================================================
        self.pause_duration = 1.5
        self.pause_timer = 0.0

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

        self.get_logger().info("Moving person_1 in CROSSING scenario")
        self.get_logger().info(f"World: {self.world_name}")
        self.get_logger().info(f"Endpoints: {self.point_a} <-> {self.point_b}")
        self.get_logger().info(
            f"Speed: {self.speed} m/s, pause: {self.pause_duration}s at endpoints"
        )

    def timer_callback(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now

        dt = max(0.01, min(dt, self.update_dt * 3.0))

        if self.pause_timer > 0.0:
            self.pause_timer = max(0.0, self.pause_timer - dt)
            self.get_logger().info(
                f"Pausing at ({self.current_x:.2f},{self.current_y:.2f}) "
                f"- {self.pause_timer:.2f}s remaining"
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

        # +pi/2 yaw offset for person_standing mesh orientation.
        # Verified for x-axis motion in move_person_gazebo.py; verify
        # visually for y-axis motion before recording evaluation runs.
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
        self.get_logger().info("Stopping move_person_crossing node")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MovePersonCrossing()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
