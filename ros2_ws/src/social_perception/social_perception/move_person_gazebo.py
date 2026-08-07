#!/usr/bin/env python3

import math
import subprocess
import threading

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
from ros_gz_interfaces.srv import SetEntityPose  # <--- 1. ADD THIS IMPORT


class MovePersonGazebo(Node):
    def __init__(self):
        super().__init__("move_person_gazebo")

        self.world_name = "empty_human"
        self.model_name = "person_1"
        
        self.gz_client = self.create_client(
            SetEntityPose, f"/world/{self.world_name}/set_pose"
        )
        self.get_logger().info("Waiting for set_pose service...")
        self.gz_client.wait_for_service()
        self.get_logger().info("set_pose service ready")

        # Movement endpoints in Gazebo world frame
        self.point_a = (8.0, 0.0, 0.0)
        self.point_b = (1.0, 0.0, 0.0)

        self.speed = 1.2       # m/s
        self.update_dt = 0.2   # seconds. 0.05 was tried to smooth the set_pose
                               # teleporting, but each tick spawns a gz service
                               # subprocess, and 20 Hz of process spawns cost
                               # ~50 points of RTF. Reverted; the jump-gate
                               # rejections it targeted were solved by
                               # process_every_n_frames=2 instead.

        # ==================================================
        # THESIS ADDITION (endpoint pause)
        #
        # Person pauses at each endpoint before reversing.
        # This gives the KF time to register near-zero velocity
        # before the direction flip, reducing the reversal error
        # spike. Also more realistic — people briefly stop before
        # changing direction.
        #
        # pause_duration: how long to hold position at endpoint (s)
        # pause_timer:    counts down remaining pause time
        # ==================================================
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

        self.get_logger().info("Moving person_1 using Gazebo set_pose service")
        self.get_logger().info(f"World: {self.world_name}")
        self.get_logger().info(f"Endpoints: {self.point_a} <-> {self.point_b}")
        self.get_logger().info(f"Speed: {self.speed} m/s, pause: {self.pause_duration}s at endpoints")

    def timer_callback(self):
        now = self.get_clock().now()
        if not hasattr(self, "_first_tick_done"):
            self._first_tick_done = True
            self.last_time = now
            self.publish_ground_truth()
            return
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now
        dt = max(0.01, min(dt, self.update_dt * 3.0))

        # ==================================================
        # Endpoint pause logic:
        # When pause_timer > 0, the person is stationary at
        # the endpoint. Publish ground truth (position valid)
        # but send no new set_pose and do no movement. Tick
        # the timer down by dt each callback.
        # ==================================================
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
            # Reached endpoint — start pause before switching target
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

        # THESIS FIX: +pi/2 yaw offset so model faces direction of travel.
        # Verified: x 3->5 => yaw=+pi/2, x 5->3 => yaw=-pi/2
        yaw = math.atan2(uy, ux) + math.pi / 2.0
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))

        self.set_model_pose(self.current_x, self.current_y, self.current_z, yaw)

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

        if not self.gz_client.service_is_ready():
            self.get_logger().warn("set_pose service not ready, skipping this tick")
            return

        req = SetEntityPose.Request()
        req.entity.name = self.model_name
        req.entity.type = req.entity.MODEL  # matches "name+type" lookup, avoids ambiguous entity resolution
        req.pose.position.x = x
        req.pose.position.y = y
        req.pose.position.z = z
        req.pose.orientation.x = 0.0
        req.pose.orientation.y = 0.0
        req.pose.orientation.z = qz
        req.pose.orientation.w = qw

        future = self.gz_client.call_async(req)
        future.add_done_callback(self._set_pose_done)

    def _set_pose_done(self, future):
        try:
            result = future.result()
            if not result.success:
                self.get_logger().warn("set_pose service call returned failure")
        except Exception as e:
            self.get_logger().warn(f"set_pose service call raised: {e}")

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