#!/usr/bin/env python3

import math
import subprocess
import threading

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose


class MovePersonGazebo(Node):
    def __init__(self):
        super().__init__("move_person_gazebo")

        self.world_name = "empty_human"
        self.model_name = "person_1"

        # Movement endpoints in Gazebo world frame
        self.point_a = (3.0, 0.0, 0.0)
        self.point_b = (5.0, 0.0, 0.0)

        self.speed = 0.2          # m/s
        self.update_dt = 0.5     # seconds

        self.current_x = self.point_a[0]
        self.current_y = self.point_a[1]
        self.current_z = self.point_a[2]

        self.target = self.point_b

        self.timer = self.create_timer(self.update_dt, self.timer_callback)
        self.last_time = self.get_clock().now()

        # ==================================================
        # THESIS ADDITION (ground-truth logging, ported from
        # move_person_gazebo2.py)
        #
        # Publishes the exact simulated x,y of person_1 on a
        # PoseArray (single-element, to keep the message format
        # consistent with the two-person version so existing
        # logging/analysis scripts - kf_prediction_logger.py,
        # plot_kf_log.py, compare_true_vs_filtered_speed.py - work
        # unchanged against a single-person run).
        #
        # This does NOT feed into navigation in any way - it's a
        # read-only "ground truth" feed for offline evaluation,
        # entirely separate from the camera/lidar perception path
        # the robot actually uses to sense the person.
        # ==================================================
        self.ground_truth_pub = self.create_publisher(
            PoseArray,
            "/person_ground_truth",
            10
        )
        self.frame_id = "map"

        self.get_logger().debug("Moving person_1 using Gazebo set_pose service")
        self.get_logger().debug(f"World: {self.world_name}")
        self.get_logger().debug(f"Model: {self.model_name}")
  

    def timer_callback(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now

        # ==================================================
        # THESIS FIX (effective-speed halving bug)
        #
        # The previous ceiling here was 0.5s. At the time this bug was
        # found, self.update_dt was 1.0s, and the timer fired once per
        # update_dt under normal operation - so a normal ~1.0s tick was
        # being clamped down to 0.5s, silently applying only HALF the
        # intended displacement on every single tick (turning the
        # configured speed=0.2 m/s into an effective 0.1 m/s under
        # completely normal conditions, not just during genuine lag/
        # pause events, which is what this clamp was actually meant to
        # guard against - see the original comment below).
        #
        # NOTE: self.update_dt has since been tuned down from 1.0s to
        # smooth out the person's motion (see its definition above) -
        # the fix below scales with whatever self.update_dt currently
        # is, so it remains correct regardless of that value.
        #
        # Fix: raise the ceiling comfortably above update_dt (3x),
        # so a genuinely stalled callback (e.g. sim pause, multi-
        # second lag) still gets its dt capped and doesn't produce a
        # huge teleport jump, but a normal on-schedule tick is no
        # longer clamped at all.
        # ==================================================
        dt = max(0.01, min(dt, self.update_dt * 3.0))

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
            # Still publish ground truth on this tick - the person's
            # position is valid/current even though no new set_pose
            # call is being sent this tick (only the target changed).
            self.publish_ground_truth()
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