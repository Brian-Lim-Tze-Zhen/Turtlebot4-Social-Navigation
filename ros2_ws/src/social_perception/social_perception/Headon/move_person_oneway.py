#!/usr/bin/env python3
"""
One-way person mover for the head-on avoidance scenario.

Walks person_1 from point_a to point_b at a constant speed, then stops
and holds position. Unlike move_person_gazebo.py there is no pause-and-
reverse: the return leg was contaminating min_distance in
analyse_avoidance.py, which takes a global minimum over the whole
recorded path with no encounter window. The person walking back past a
robot parked near the goal set that value instead of the head-on
encounter, invalidating several trials.

Stopping at point_b also removes a hand-timed step (Ctrl+C on the mover
partway through), which was itself a source of run-to-run variation.

Run with:
  python3 move_person_oneway.py --ros-args -p use_sim_time:=true

Requires the gz service bridge to be running:
  ros2 run ros_gz_bridge parameter_bridge \
    /world/empty_human/set_pose@ros_gz_interfaces/srv/SetEntityPose
"""

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose
from ros_gz_interfaces.srv import SetEntityPose


class MovePersonOneWay(Node):
    def __init__(self):
        super().__init__("move_person_oneway")

        # ==========================================================
        # THESIS CHANGE (parameterised for the combined world)
        #
        # World, model and endpoints were hardcoded for empty_human.
        # The combined world uses a different world name, a different
        # model name (person_mover) and a corridor running along +y
        # rather than +x, so a hardcoded mover fails SILENTLY there:
        # the set_pose service name does not exist, every call is
        # skipped, and the person simply never moves while the node
        # logs normally.
        #
        # Parameterised rather than forked so both scenarios keep using
        # the mover that conditions A-E were recorded with - including
        # the first-tick guard and the one-way traverse, both of which
        # exist because their absence invalidated earlier trials.
        #
        # combined_scenario invocation:
        #   -p world_name:=combined_scenario
        #   -p model_name:=person_mover
        #   -p point_a:="[2.0, 8.0]"  -p point_b:="[2.0, -2.0]"
        # ==========================================================
        self.declare_parameter("world_name", "empty_human")
        self.declare_parameter("model_name", "person_1")
        self.declare_parameter("point_a", [8.0, 0.0])
        self.declare_parameter("point_b", [-3.0, 0.0])
        self.declare_parameter("speed", 1.2)

        # In a world where queue_ground_truth_node is also publishing,
        # this must be off. Two publishers on /person_ground_truth send
        # PoseArrays of DIFFERENT LENGTHS - one pose here, N there - and
        # analyse_avoidance.py indexes people by position in the array,
        # so the two interleave and every person's identity shifts
        # between messages. Nothing errors; the metrics are simply wrong.
        self.declare_parameter("publish_ground_truth", True)

        # Yaw offset between the mesh's own forward axis and the
        # direction of travel. The +pi/2 baked in below was measured in
        # empty_human, where the pedestrian walks along -x. In a world
        # where the corridor runs along y the same constant is 90 deg
        # out and the mesh walks sideways, facing a wall.
        #
        # Not cosmetic: person_standing is raycast against the VISUAL
        # mesh (two legs), not the SDF collision cylinder, so
        # leg_detector's cluster count is bearing-dependent - one
        # cluster below ~25 deg, two above ~40 deg. A pedestrian
        # rotated 90 deg presents a different leg profile than the one
        # the clustering thresholds were set against, and the VLM
        # facing classification sees a different silhouette entirely.
        self.declare_parameter("mesh_yaw_offset", math.pi / 2.0)

        self.world_name = self.get_parameter("world_name").value
        self.model_name = self.get_parameter("model_name").value

        self.gz_client = self.create_client(
            SetEntityPose, f"/world/{self.world_name}/set_pose"
        )
        self.get_logger().info("Waiting for set_pose service...")
        if not self.gz_client.wait_for_service(timeout_sec=15.0):
            self.get_logger().error(
                "set_pose service unavailable - is the ros_gz_bridge running?"
            )
            raise RuntimeError("set_pose service unavailable")
        self.get_logger().info("set_pose service ready")

        # Traverse endpoints in the Gazebo world frame. point_b is well
        # clear of wall_3 at x=-4.5, and far enough past the robot's
        # start that the encounter happens mid-stride rather than during
        # the person's deceleration.
        pa = self.get_parameter("point_a").value
        pb = self.get_parameter("point_b").value
        self.point_a = (float(pa[0]), float(pa[1]), 0.0)
        self.point_b = (float(pb[0]), float(pb[1]), 0.0)

        self.speed = float(self.get_parameter("speed").value)
        self.update_dt = 0.2   # s

        self.current_x, self.current_y, self.current_z = self.point_a
        self.target = self.point_b
        self.finished = False

        self.publish_gt = self.get_parameter("publish_ground_truth").value
        self.mesh_yaw_offset = float(
            self.get_parameter("mesh_yaw_offset").value)
        self.ground_truth_pub = self.create_publisher(
            PoseArray, "/person_ground_truth", 10
        )
        if not self.publish_gt:
            self.get_logger().info(
                "publish_ground_truth=false - another node owns "
                "/person_ground_truth in this world")
        self.frame_id = "map"

        self.last_time = self.get_clock().now()
        self.timer = self.create_timer(self.update_dt, self.timer_callback)

        self.get_logger().info("One-way person mover started")
        self.get_logger().info(f"World: {self.world_name}, model: {self.model_name}")
        self.get_logger().info(f"{self.point_a} -> {self.point_b} at {self.speed} m/s")

    def timer_callback(self):
        now = self.get_clock().now()

        # First tick carries dt measured from __init__, which includes
        # node startup and service discovery. Left unguarded that becomes
        # a clamped ~0.72 m teleport, and the internal position advances
        # while the Gazebo model has not moved - offsetting
        # /person_ground_truth for the whole run.
        if not hasattr(self, "_first_tick_done"):
            self._first_tick_done = True
            self.last_time = now
            self.publish_ground_truth()
            return

        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now
        dt = max(0.01, min(dt, self.update_dt * 3.0))

        if self.finished:
            self.publish_ground_truth()
            return

        tx, ty, _ = self.target
        dx = tx - self.current_x
        dy = ty - self.current_y
        dist = math.hypot(dx, dy)

        if dist < 0.05:
            self.finished = True
            self.get_logger().info(
                f"Reached ({self.current_x:.2f},{self.current_y:.2f}) - holding position"
            )
            self.publish_ground_truth()
            return

        step = min(self.speed * dt, dist)
        ux, uy = dx / dist, dy / dist
        self.current_x += ux * step
        self.current_y += uy * step

        self.get_logger().info(
            f"dt={dt:.3f}s step={step:.3f}m "
            f"pos=({self.current_x:.2f},{self.current_y:.2f})"
        )

        # +pi/2 offset so the mesh faces its direction of travel.
        yaw = math.atan2(uy, ux) + self.mesh_yaw_offset
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))

        self.set_model_pose(self.current_x, self.current_y, self.current_z, yaw)
        self.publish_ground_truth()

    def publish_ground_truth(self):
        if not self.publish_gt:
            return
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
        if not self.gz_client.service_is_ready():
            self.get_logger().warn("set_pose service not ready, skipping this tick")
            return

        req = SetEntityPose.Request()
        req.entity.name = self.model_name
        req.entity.type = req.entity.MODEL
        req.pose.position.x = x
        req.pose.position.y = y
        req.pose.position.z = z
        req.pose.orientation.x = 0.0
        req.pose.orientation.y = 0.0
        req.pose.orientation.z = math.sin(yaw / 2.0)
        req.pose.orientation.w = math.cos(yaw / 2.0)

        future = self.gz_client.call_async(req)
        future.add_done_callback(self._set_pose_done)

    def _set_pose_done(self, future):
        try:
            if not future.result().success:
                self.get_logger().warn("set_pose returned failure")
        except Exception as e:
            self.get_logger().warn(f"set_pose raised: {e}")

    def destroy_node(self):
        self.get_logger().info("Stopping one-way person mover")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MovePersonOneWay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
