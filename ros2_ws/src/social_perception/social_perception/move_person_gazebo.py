#!/usr/bin/env python3
"""
move_person_gazebo.py — actor-mode ground-truth publisher

person_1 is now a Gazebo <actor> whose motion is entirely driven by the
SDF script in empty_human.sdf (10 s per 2 m leg → 0.2 m/s, looping).
This node no longer calls gz service or controls any motion itself.

Its sole job is to bridge the actor's Gazebo pose onto /person_ground_truth
(geometry_msgs/PoseArray) so the existing evaluation scripts
(kf_prediction_logger.py, compare_true_vs_filtered_speed.py, etc.) continue
to work without modification.

BRIDGE REQUIREMENT:
    Gazebo's SceneBroadcaster publishes the actor pose on the gz topic
    /model/person_1/pose (gz.msgs.Pose).  ros_gz_bridge must map it to
    a ROS 2 topic before this node can subscribe.  Start the bridge with:

        ros2 run ros_gz_bridge parameter_bridge \
            /model/person_1/pose@geometry_msgs/msg/Pose[gz.msgs.Pose

    or add the equivalent mapping to the turtlebot4 bridge launch config.
    Without the bridge this node will start but receive no messages and
    publish nothing.

USAGE:
    ros2 run social_perception move_person_gazebo --ros-args -p use_sim_time:=true
    ros2 run social_perception move_person_gazebo --ros-args \
        -p use_sim_time:=true -p world_name:=empty_human
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseArray


class MovePersonGazebo(Node):
    def __init__(self):
        super().__init__("move_person_gazebo")

        self.declare_parameter("world_name", "empty_human")
        self.world_name = (
            self.get_parameter("world_name").get_parameter_value().string_value
        )

        self._ground_truth_pub = self.create_publisher(
            PoseArray,
            "/person_ground_truth",
            10,
        )

        self.create_subscription(
            Pose,
            "/model/person_1/pose",
            self._actor_pose_callback,
            10,
        )

        self.get_logger().info(
            f"move_person_gazebo (actor mode): world={self.world_name}, "
            "subscribing to /model/person_1/pose, "
            "publishing /person_ground_truth. "
            "Requires ros_gz_bridge — see module docstring."
        )

    def _actor_pose_callback(self, msg: Pose) -> None:
        out = PoseArray()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "map"
        out.poses.append(msg)
        self._ground_truth_pub.publish(out)


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
