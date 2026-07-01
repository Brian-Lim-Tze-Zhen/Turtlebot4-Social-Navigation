#!/usr/bin/env python3
"""
kf_prediction_logger.py

Subscribes to:
  - /predicted_person_positions  (output of human_kf_predictor.py)
  - /person_ground_truth         (output of move_person_gazebo2.py, PoseArray)

and logs both to CSV files for later graphing/analysis.

IMPORTANT - sim time:
    Your other nodes (human_kf_predictor, move_person_gazebo2, etc.) are all
    launched with -p use_sim_time:=true, which means their internal clock
    follows Gazebo's /clock topic, not the wall clock. This node now does
    the same thing: pass use_sim_time:=true here too, or the logged
    timestamps will NOT line up with the rest of your pipeline (e.g. if
    Gazebo's real_time_factor isn't exactly 1.0).

Usage (live run, matching your launch sequence):
    ros2 run social_perception kf_prediction_logger --ros-args -p use_sim_time:=true
    # or:
    python3 kf_prediction_logger.py --ros-args -p use_sim_time:=true

Usage (bag replay instead of a live run):
    Terminal 1: ros2 bag play my_bag --clock     (the bag must contain /clock)
    Terminal 2: ros2 run <pkg> human_kf_predictor --ros-args -p use_sim_time:=true
                (only needed if the bag has raw detections, not predictions)
    Terminal 3: python3 kf_prediction_logger.py --ros-args -p use_sim_time:=true

Output:
    ~/kf_prediction_log.csv   (override with -p output_path:=...)
    ~/ground_truth_log.csv    (override with -p ground_truth_output_path:=...)

Columns (kf_prediction_log.csv):
    sim_time, track_id, conf, x, y, vx, vy, pred_x, pred_y, horizon

Columns (ground_truth_log.csv):
    sim_time, person_index, x, y, z
"""

import csv
import os

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseArray


class KFPredictionLogger(Node):
    def __init__(self):
        super().__init__("kf_prediction_logger")

        # NOTE: use_sim_time is automatically declared by rclpy.node.Node
        # itself for every node - do NOT call self.declare_parameter on it
        # here, that throws ParameterAlreadyDeclaredException. Just read it.
        self.declare_parameter("input_topic", "/predicted_person_positions")
        self.declare_parameter("output_path", os.path.expanduser("~/kf_prediction_log.csv"))

        self.declare_parameter("ground_truth_topic", "/person_ground_truth")
        self.declare_parameter(
            "ground_truth_output_path", os.path.expanduser("~/ground_truth_log.csv")
        )

        using_sim_time = self.get_parameter("use_sim_time").get_parameter_value().bool_value
        self.input_topic = (
            self.get_parameter("input_topic").get_parameter_value().string_value
        )
        self.output_path = (
            self.get_parameter("output_path").get_parameter_value().string_value
        )
        self.ground_truth_topic = (
            self.get_parameter("ground_truth_topic").get_parameter_value().string_value
        )
        self.ground_truth_output_path = (
            self.get_parameter("ground_truth_output_path").get_parameter_value().string_value
        )

        # --- Prediction log ---
        self._file = open(self.output_path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            ["sim_time", "track_id", "conf", "x", "y", "vx", "vy", "pred_x", "pred_y", "horizon"]
        )
        self._row_count = 0

        self.sub = self.create_subscription(
            String, self.input_topic, self.callback, 10
        )

        # --- Ground truth log ---
        self._gt_file = open(self.ground_truth_output_path, "w", newline="")
        self._gt_writer = csv.writer(self._gt_file)
        self._gt_writer.writerow(["sim_time", "person_index", "x", "y", "z"])
        self._gt_row_count = 0

        self.gt_sub = self.create_subscription(
            PoseArray, self.ground_truth_topic, self.ground_truth_callback, 10
        )

        self.get_logger().info(
            f"use_sim_time={using_sim_time} "
            f"({'reading Gazebo /clock' if using_sim_time else 'reading WALL clock - pass -p use_sim_time:=true to match the rest of your pipeline'})"
        )
        self.get_logger().info(f"Logging '{self.input_topic}' -> {self.output_path}")
        self.get_logger().info(f"Logging '{self.ground_truth_topic}' -> {self.ground_truth_output_path}")

    def callback(self, msg: String):
        parts = msg.data.split(",")

        if len(parts) < 9:
            self.get_logger().warn(f"Skipping malformed msg: {msg.data}")
            return

        try:
            track_id = int(float(parts[0]))
            conf = float(parts[1])
            x, y = float(parts[2]), float(parts[3])
            vx, vy = float(parts[4]), float(parts[5])
            pred_x, pred_y = float(parts[6]), float(parts[7])
            horizon = float(parts[8])
        except ValueError:
            self.get_logger().warn(f"Could not parse msg: {msg.data}")
            return

        sim_time = self.get_clock().now().nanoseconds * 1e-9

        self._writer.writerow(
            [sim_time, track_id, conf, x, y, vx, vy, pred_x, pred_y, horizon]
        )
        self._row_count += 1

        if self._row_count % 20 == 0:
            self._file.flush()

    def ground_truth_callback(self, msg: PoseArray):
        sim_time = self.get_clock().now().nanoseconds * 1e-9

        for i, pose in enumerate(msg.poses):
            self._gt_writer.writerow(
                [sim_time, i, pose.position.x, pose.position.y, pose.position.z]
            )
            self._gt_row_count += 1

        if self._gt_row_count % 20 == 0:
            self._gt_file.flush()

    def destroy_node(self):
        self._file.flush()
        self._file.close()
        self._gt_file.flush()
        self._gt_file.close()
        self.get_logger().info(f"Wrote {self._row_count} prediction rows to {self.output_path}")
        self.get_logger().info(f"Wrote {self._gt_row_count} ground-truth rows to {self.ground_truth_output_path}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = KFPredictionLogger()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()