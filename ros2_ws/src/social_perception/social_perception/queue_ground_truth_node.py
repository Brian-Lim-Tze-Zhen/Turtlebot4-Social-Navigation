#!/usr/bin/env python3
"""
queue_ground_truth_node.py

THESIS ADDITION - ground-truth pose feed for the queue scenario.

-----------------------------------------------------------------------
WHY THIS EXISTS
-----------------------------------------------------------------------
The head-on scenario gets /person_ground_truth for free: move_person_*.py
commands the pedestrian's pose, so it already knows where the person is
and republishes it as a PoseArray for offline evaluation.

The queue scenario has no mover - its four pedestrians are
<static>true</static> in queue_test.sdf - so nothing was publishing
ground truth, and the first recorded trial captured
/person_ground_truth with a count of 0.

Without it, any proximity metric would have to be computed from the
perception pipeline's own position estimates. Since the perception
pipeline is the system under test, that is circular: a run where
detection drifts would report a distance error as a navigation result.

-----------------------------------------------------------------------
WHY IT QUERIES GAZEBO INSTEAD OF HARDCODING THE SDF VALUES
-----------------------------------------------------------------------
The coordinates are in queue_test.sdf and could simply be copied here.
They are not, because a copy silently goes stale the moment the world
file is edited - and this session already changed the queue layout twice
(horizontal -> in-line -> horizontal). A hardcoded ground truth that
disagrees with the world would corrupt every metric derived from it,
with no error raised anywhere.

Instead the poses are read from the running simulator at startup via
`gz model -m <name> -p`. One query per model is sufficient BECAUSE the
models are static; this node is not valid for moving pedestrians, and
refuses to pretend otherwise (see the staleness note in the docstring of
query_model_pose).

-----------------------------------------------------------------------
OUTPUT
-----------------------------------------------------------------------
Publishes /person_ground_truth (geometry_msgs/PoseArray, frame "map"),
matching the format move_person_crossing2.py already uses, so existing
analysis scripts need no change.

Poses are published in the order given by --model-names, which defaults
to the queue's spatial order from +y to -y.

This feed is READ-ONLY with respect to navigation. Nothing in the robot's
control path subscribes to it.

-----------------------------------------------------------------------
USAGE
-----------------------------------------------------------------------
    python3 queue_ground_truth_node.py --ros-args -p use_sim_time:=true

Optional parameters:
    -p world_name:=queue_test
    -p model_names:="['person_queue_a','person_queue_b']"
    -p publish_rate_hz:=10.0
"""

import math
import re
import subprocess

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseArray, Pose


DEFAULT_MODELS = [
    "person_queue_a",
    "person_queue_b",
    "person_queue_c",
    "person_queue_d",
]


class QueueGroundTruthNode(Node):
    def __init__(self):
        super().__init__("queue_ground_truth_node")

        self.declare_parameter("world_name", "queue_test")
        self.declare_parameter("model_names", DEFAULT_MODELS)
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("query_timeout_s", 5.0)

        self.world_name = self.get_parameter("world_name").value
        self.model_names = list(self.get_parameter("model_names").value)
        rate = float(self.get_parameter("publish_rate_hz").value)
        self.query_timeout = float(self.get_parameter("query_timeout_s").value)

        self.frame_id = "map"
        self.pub = self.create_publisher(PoseArray, "/person_ground_truth", 10)

        self.get_logger().info("Queue ground truth node starting")
        self.get_logger().info(f"World : {self.world_name}")
        self.get_logger().info(f"Models: {self.model_names}")

        self.poses = []
        for name in self.model_names:
            pose = self.query_model_pose(name)
            if pose is None:
                # Refuse to publish a partial array. A short PoseArray
                # would be silently misread downstream as "one of the
                # people is missing" rather than "the query failed".
                self.get_logger().error(
                    f"Could not read pose for '{name}' - is Gazebo running "
                    f"with world '{self.world_name}'? Publishing nothing.")
                self.poses = []
                return
            self.poses.append(pose)
            self.get_logger().info(
                f"  {name}: x={pose.position.x:.3f} y={pose.position.y:.3f}")

        self.create_timer(1.0 / rate, self.publish_ground_truth)
        self.get_logger().info(
            f"Publishing /person_ground_truth at {rate:.1f} Hz "
            f"({len(self.poses)} static models, frame '{self.frame_id}')")

    def query_model_pose(self, model_name):
        """Read one model's pose from the running Gazebo instance.

        Queried ONCE at startup, not per cycle. That is correct only
        because these models are static: `gz model -p` is a blocking
        service call taking tens of milliseconds, so polling it at 10 Hz
        would be both wasteful and jittery. If this node is ever reused
        for a scenario with moving pedestrians, this must be replaced
        with a subscription to the simulator's pose stream - a cached
        value would report a person standing where they no longer are,
        which is worse than no ground truth at all.
        """
        try:
            result = subprocess.run(
                ["gz", "model", "-m", model_name, "-p"],
                capture_output=True, text=True, timeout=self.query_timeout)
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            self.get_logger().error(f"gz model query failed for {model_name}: {e}")
            return None

        if result.returncode != 0:
            self.get_logger().error(
                f"gz model returned {result.returncode} for {model_name}: "
                f"{result.stderr.strip()}")
            return None

        return self.parse_pose(result.stdout, model_name)

    def parse_pose(self, text, model_name):
        """Extract position and orientation from `gz model -p` output.

        Verified against Gazebo Harmonic, which prints two bracketed
        triples under a "Pose [ XYZ (m) ] [ RPY (rad) ]" header:

            [3.000000 1.000000 0.000000]
            [0.000000 -0.000000 0.000000]

        Note the second triple is ROLL/PITCH/YAW, not a quaternion - an
        earlier draft of this parser looked for four numbers and would
        have silently fallen back to an identity orientation. Matching
        labelled structure rather than counting numbers, and erroring
        loudly on a miss, keeps a format change from turning into a
        wrong ground truth.
        """
        num = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
        triples = re.findall(rf"\[({num})\s+({num})\s+({num})\]", text)

        if len(triples) < 2:
            self.get_logger().error(
                f"Could not parse pose for {model_name} "
                f"(found {len(triples)} coordinate triples, expected 2). "
                f"Raw output:\n{text}")
            return None

        x, y, z = (float(v) for v in triples[0])
        roll, pitch, yaw = (float(v) for v in triples[1])

        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z

        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        pose.orientation.w = cr * cp * cy + sr * sp * sy
        pose.orientation.x = sr * cp * cy - cr * sp * sy
        pose.orientation.y = cr * sp * cy + sr * cp * sy
        pose.orientation.z = cr * cp * sy - sr * sp * cy

        return pose

    def publish_ground_truth(self):
        if not self.poses:
            return
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.poses = self.poses
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = QueueGroundTruthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
