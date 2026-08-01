#!/usr/bin/env python3

import math
import subprocess
import threading

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseArray, Pose


class PersonMover:
    """
    Holds the moving state for a single Gazebo person model and
    knows how to step itself toward its current target waypoint.
    """

    def __init__(self, model_name, point_a, point_b, speed=0.2):
        self.model_name = model_name
        self.point_a = point_a
        self.point_b = point_b
        self.speed = speed

        self.current_x = point_a[0]
        self.current_y = point_a[1]
        self.current_z = point_a[2]

        self.target = point_b

    def step(self, dt):
        target_x, target_y, target_z = self.target

        dx = target_x - self.current_x
        dy = target_y - self.current_y

        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 0.05:
            self.target = self.point_a if self.target == self.point_b else self.point_b
            return None  # signal: just switched target, no pose to send this tick

        step = min(self.speed * dt, dist)

        ux = dx / dist
        uy = dy / dist

        self.current_x += ux * step
        self.current_y += uy * step

        # THESIS FIX (restored from move_person_gazebo2.py): continuous
        # heading instead of the binary 0.0/3.1 this file had reverted
        # to, with the same +pi/2 mesh-orientation offset verified
        # there. The correction is a constant for the model's default
        # facing, not axis-specific, so it applies the same way to this
        # file's perpendicular (y-axis) movement as it did to the
        # original's x-axis movement.
        yaw = math.atan2(uy, ux) + math.pi / 2.0
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))  # normalize to [-pi, pi]

        return self.current_x, self.current_y, self.current_z, yaw


class MovePeopleGazeboCrossing(Node):
    def __init__(self):
        super().__init__("move_people_gazebo_crossing")

        self.world_name = "two_human"
        self.update_dt = 1.0  # seconds; see dt clamp note in timer_callback

        # ==================================================
        # person_1: starts (3, -2). Moves along the y-axis at
        # fixed x=3, bouncing between y=-2 and y=2.
        # ==================================================
        # person_2: starts (6, 2). Moves along the y-axis at
        # fixed x=6, bouncing between y=2 and y=-2 (opposite
        # phase to person_1). Perpendicular-crossing variant of
        # move_person_gazebo2.py's parallel-lane scenario.
        # ==================================================
        self.people = [
            PersonMover("person_1", point_a=(3.0, -2.0, 0.0), point_b=(3.0, 2.0, 0.0)),
            PersonMover("person_2", point_a=(6.0, 2.0, 0.0), point_b=(6.0, -2.0, 0.0)),
        ]

        self.last_time = self.get_clock().now()
        self.timer = self.create_timer(self.update_dt, self.timer_callback)

        # ==================================================
        # THESIS EVALUATION ADDITION (ground-truth logging)
        #
        # Publishes the exact simulated x,y of every person on a
        # PoseArray, in the order self.people is defined, so that
        # ros2 bag can capture ground-truth human position synced
        # against /odom and /cmd_vel for A/B comparison runs
        # (stock Nav2 vs. predicted_person_cloud pipeline).
        #
        # This does NOT feed into navigation in any way - it's a
        # read-only "ground truth" feed for offline evaluation,
        # entirely separate from the camera/lidar perception path
        # the robot actually uses to sense people.
        # ==================================================
        self.ground_truth_pub = self.create_publisher(
            PoseArray,
            "/person_ground_truth",
            10
        )
        self.frame_id = "map"

        self.get_logger().info("Moving multiple people using Gazebo set_pose service")
        self.get_logger().info(f"World: {self.world_name}")
        for p in self.people:
            self.get_logger().info(f"  - {p.model_name}: {p.point_a} <-> {p.point_b}")

    def timer_callback(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now

        # THESIS FIX (restored from move_person_gazebo2.py, effective-
        # speed-halving bug): ceiling must stay above update_dt, not a
        # flat 0.5s. This file had reverted to min(dt, 0.5) with
        # update_dt=1.0 - every normal ~1.0s tick gets clamped to 0.5s,
        # silently halving speed=0.2 m/s to an effective 0.1 m/s on
        # every tick, not just during genuine stalls.
        dt = max(0.01, min(dt, self.update_dt * 3.0))

        for person in self.people:
            result = person.step(dt)

            if result is None:
                self.get_logger().info(
                    f"{person.model_name} switching target to {person.target}"
                )
                continue

            x, y, z, yaw = result

            self.get_logger().info(
                f"{person.model_name} pos=({x:.2f},{y:.2f}) target={person.target}"
            )

            # Run the blocking gz service call in a background thread.
            # This is different from the previous Popen-based
            # fire-and-forget approach: here, subprocess.run() still
            # blocks and waits for Gazebo's reply (so the request/
            # response cycle completes cleanly and Gazebo doesn't log
            # "Host unreachable" errors from an abandoned client), but
            # it does so on a separate thread, so the ROS timer
            # callback itself returns immediately and isn't held up
            # waiting for the gz service round-trip.
            threading.Thread(
                target=self.set_model_pose,
                args=(person.model_name, x, y, z, yaw),
                daemon=True,
            ).start()

        # Publish ground truth for ALL people every tick, regardless
        # of whether any individual person just switched waypoint
        # target this tick (those return None above and are skipped
        # for set_pose, but their position is still valid/current).
        self.publish_ground_truth()

    def publish_ground_truth(self):
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        for person in self.people:
            pose = Pose()
            pose.position.x = person.current_x
            pose.position.y = person.current_y
            pose.position.z = person.current_z
            msg.poses.append(pose)

        self.ground_truth_pub.publish(msg)

    def set_model_pose(self, model_name, x, y, z, yaw):
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)

        service_name = f"/world/{self.world_name}/set_pose"

        req = (
            f"name: '{model_name}', "
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
                timeout=3.0
            )

            if result.returncode != 0:
                self.get_logger().warn(f"set_pose failed for {model_name}: {result.stderr}")

        except subprocess.TimeoutExpired:
            self.get_logger().warn(f"set_pose service timeout for {model_name}")

    def destroy_node(self):
        self.get_logger().info("Stopping move_people_gazebo_crossing node")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = MovePeopleGazeboCrossing()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()