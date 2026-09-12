#!/usr/bin/env python3
"""
compute_min_distance.py

Evaluation node for the "with prediction" vs "without prediction"
Nav2 ablation study.

Subscribes to:
    /odom                 (nav_msgs/Odometry)       -> robot position
    /person_ground_truth  (geometry_msgs/PoseArray) -> true person position(s),
                                                        published by
                                                        move_person_gazebo2.py

At every ground-truth tick, computes the distance from the robot to the
CLOSEST person in the PoseArray, logs it to a CSV, and tracks running
stats. On shutdown (Ctrl+C), prints and saves a summary:

    - mean / median / std / min / max distance
    - % of samples under the danger threshold
    - number of discrete "close-call" events (contiguous stretches
      under threshold, counted once per stretch rather than once per
      sample, so a 2-second close pass doesn't get counted as 40 events)

Usage:
    ros2 run <your_pkg> compute_min_distance --ros-args \
        -p label:=with_prediction \
        -p danger_threshold:=0.5

    ros2 run <your_pkg> compute_min_distance --ros-args \
        -p label:=no_prediction \
        -p danger_threshold:=0.5

Run once per config (social_nav2.yaml vs social_nav2_no_predicted_cloud.yaml),
each producing its own timestamped CSV + summary file, named by `label`,
so results from both runs can be loaded side by side afterward for the
comparison table / box plot.

NOTE: move_person_gazebo.py (single-person script) does NOT currently
publish /person_ground_truth -- only move_person_gazebo2.py does. If
you want to run the ablation with a single person, either switch to
move_person_gazebo2.py (only one PersonMover in self.people), or add
the same ground-truth PoseArray publisher to move_person_gazebo.py.
"""

import csv
import math
import os
import statistics
from datetime import datetime

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray


class MinDistanceEvaluator(Node):
    def __init__(self):
        super().__init__("compute_min_distance")

        # =====================================
        # User configurable parameters
        # =====================================
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("ground_truth_topic", "/person_ground_truth")
        self.declare_parameter("danger_threshold", 0.5)  # meters
        self.declare_parameter("label", "run")           # e.g. "with_prediction"
        self.declare_parameter("output_dir", "/root/thesis_social_navigation_ws/eval_logs")

        self.odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        self.gt_topic = self.get_parameter("ground_truth_topic").get_parameter_value().string_value
        self.danger_threshold = self.get_parameter("danger_threshold").get_parameter_value().double_value
        self.label = self.get_parameter("label").get_parameter_value().string_value
        self.output_dir = self.get_parameter("output_dir").get_parameter_value().string_value

        os.makedirs(self.output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(
            self.output_dir, f"min_distance_{self.label}_{timestamp}.csv"
        )
        self.summary_path = os.path.join(
            self.output_dir, f"min_distance_{self.label}_{timestamp}_summary.txt"
        )

        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            ["time_sec", "robot_x", "robot_y", "closest_person_idx", "distance_m"]
        )

        # Internal state
        self.robot_x = None
        self.robot_y = None
        self.distances = []          # every sample, for stats
        self.timestamps = []         # matching timestamps, for event counting
        self.below_threshold_flags = []  # bool per sample, for event counting

        self.sub_odom = self.create_subscription(
            Odometry, self.odom_topic, self.odom_callback, 20
        )
        self.sub_gt = self.create_subscription(
            PoseArray, self.gt_topic, self.gt_callback, 20
        )

        self.get_logger().info("Min-distance evaluator started")
        self.get_logger().info(f"  Odom topic         : {self.odom_topic}")
        self.get_logger().info(f"  Ground truth topic : {self.gt_topic}")
        self.get_logger().info(f"  Danger threshold    : {self.danger_threshold:.2f} m")
        self.get_logger().info(f"  Label                : {self.label}")
        self.get_logger().info(f"  Logging to           : {self.csv_path}")

    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

    def gt_callback(self, msg):
        if self.robot_x is None or self.robot_y is None:
            return  # no odom yet

        if len(msg.poses) == 0:
            return

        # Distance to every person in the array; keep the closest.
        best_dist = None
        best_idx = -1

        for idx, pose in enumerate(msg.poses):
            dx = pose.position.x - self.robot_x
            dy = pose.position.y - self.robot_y
            dist = math.hypot(dx, dy)

            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = idx

        now_sec = self.get_clock().now().nanoseconds / 1e9

        self.csv_writer.writerow(
            [f"{now_sec:.3f}", f"{self.robot_x:.3f}", f"{self.robot_y:.3f}",
             best_idx, f"{best_dist:.3f}"]
        )
        self.csv_file.flush()

        self.distances.append(best_dist)
        self.timestamps.append(now_sec)
        self.below_threshold_flags.append(best_dist < self.danger_threshold)

        if best_dist < self.danger_threshold:
            self.get_logger().warn(
                f"CLOSE CALL: dist={best_dist:.2f} m to person {best_idx} "
                f"(threshold={self.danger_threshold:.2f} m)"
            )

    def count_close_call_events(self):
        """
        Count contiguous stretches of below-threshold samples as single
        events, rather than counting every sample in a stretch. e.g. a
        2-second close pass sampled at 10Hz is one event, not ~20.
        """
        events = 0
        in_event = False

        for flag in self.below_threshold_flags:
            if flag and not in_event:
                events += 1
                in_event = True
            elif not flag:
                in_event = False

        return events

    def write_summary(self):
        if len(self.distances) == 0:
            self.get_logger().warn("No distance samples collected — nothing to summarize.")
            return

        n = len(self.distances)
        mean_d = statistics.mean(self.distances)
        median_d = statistics.median(self.distances)
        std_d = statistics.stdev(self.distances) if n > 1 else 0.0
        min_d = min(self.distances)
        max_d = max(self.distances)

        pct_below = 100.0 * sum(self.below_threshold_flags) / n
        num_events = self.count_close_call_events()

        summary_lines = [
            f"Min-distance evaluation summary",
            f"Label              : {self.label}",
            f"Samples            : {n}",
            f"Danger threshold   : {self.danger_threshold:.2f} m",
            f"",
            f"Mean distance      : {mean_d:.3f} m",
            f"Median distance    : {median_d:.3f} m",
            f"Std dev            : {std_d:.3f} m",
            f"Min distance       : {min_d:.3f} m",
            f"Max distance       : {max_d:.3f} m",
            f"",
            f"%% time < threshold : {pct_below:.1f} %%",
            f"Close-call events  : {num_events}",
        ]

        summary_text = "\n".join(summary_lines)

        with open(self.summary_path, "w") as f:
            f.write(summary_text + "\n")

        self.get_logger().info("\n" + summary_text)
        self.get_logger().info(f"Summary saved to: {self.summary_path}")

    def destroy_node(self):
        self.write_summary()
        self.csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = MinDistanceEvaluator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
