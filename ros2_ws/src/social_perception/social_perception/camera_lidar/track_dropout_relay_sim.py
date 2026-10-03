#!/usr/bin/env python3
"""
track_dropout_relay_sim.py - SIMULATION TEST TOOL, not part of the pipeline.

Reproduces, for the SocialCritic only, what the real robot's tracker did in
7 of the 11 head-on bags of 25 Sep: the person's track is lost during the
approach and comes back as a NEW track id whose KF velocity starts from zero.

  /predicted_person_positions  ->  /predicted_person_positions_dropout

Messages pass through unchanged until the robot-person distance first drops
to `dropout_at_m`. Then nothing is published for `dropout_s` seconds, and
after that every message goes out with its id shifted by `id_shift` and its
velocity (and KF prediction) ramped from zero over `velocity_ramp_s`, as a
restarted Kalman filter would report it.

The SocialCritic is pointed at the output topic by run_headon_F_trial.sh
(TRACK_DROPOUT_AT=<m>). The person cloud node keeps reading the original
topic, so the costmaps are not affected.

Run:
  python3 track_dropout_relay_sim.py --ros-args -p use_sim_time:=true \
      -p dropout_at_m:=5.5 -p dropout_s:=1.0
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from std_msgs.msg import String


class TrackDropoutRelay(Node):
    def __init__(self):
        super().__init__("track_dropout_relay")
        self.declare_parameter("input_topic", "/predicted_person_positions")
        self.declare_parameter("output_topic", "/predicted_person_positions_dropout")
        self.declare_parameter("robot_topic", "/sim_ground_truth_pose")
        self.declare_parameter("dropout_at_m", 5.5)
        self.declare_parameter("dropout_s", 1.0)
        self.declare_parameter("id_shift", 100)
        self.declare_parameter("velocity_ramp_s", 0.8)

        self.dropout_at = float(self.get_parameter("dropout_at_m").value)
        self.dropout_s = float(self.get_parameter("dropout_s").value)
        self.id_shift = int(self.get_parameter("id_shift").value)
        self.ramp_s = float(self.get_parameter("velocity_ramp_s").value)

        self.robot = None
        self.t_drop = None       # sim time the dropout started
        self.t_back = None       # sim time the track came back

        self.pub = self.create_publisher(
            String, self.get_parameter("output_topic").value, 10)
        self.create_subscription(
            Odometry, self.get_parameter("robot_topic").value, self.robot_cb,
            qos_profile_sensor_data)   # the publisher is BEST_EFFORT
        self.create_subscription(
            String, self.get_parameter("input_topic").value, self.person_cb, 10)
        self.get_logger().info(
            f"track dropout relay: lose the track at {self.dropout_at:.1f} m for "
            f"{self.dropout_s:.1f} s, then id + {self.id_shift}, velocity ramp "
            f"{self.ramp_s:.1f} s")

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def robot_cb(self, msg):
        self.robot = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    def person_cb(self, msg):
        f = msg.data.split(",")
        if len(f) < 9:
            return
        now = self.now_s()

        if self.t_drop is None:
            if self.robot is not None:
                try:
                    d = math.hypot(float(f[2]) - self.robot[0], float(f[3]) - self.robot[1])
                except ValueError:
                    return
                if d <= self.dropout_at:
                    self.t_drop = now
                    self.get_logger().info(
                        f"track id {f[0]} LOST at {d:.2f} m (dropout {self.dropout_s:.1f} s)")
                    return
            self.pub.publish(msg)
            return

        if now - self.t_drop < self.dropout_s:
            return

        if self.t_back is None:
            self.t_back = now
            self.get_logger().info(
                f"track back as id {int(float(f[0])) + self.id_shift}")
        k = min(1.0, (now - self.t_back) / self.ramp_s) if self.ramp_s > 0.0 else 1.0
        try:
            x, y = float(f[2]), float(f[3])
            vx, vy = float(f[4]) * k, float(f[5]) * k
            horizon = float(f[8])
            f[0] = str(int(float(f[0])) + self.id_shift)
        except ValueError:
            return
        f[4] = f"{vx:.3f}"
        f[5] = f"{vy:.3f}"
        f[6] = f"{x + vx * horizon:.3f}"
        f[7] = f"{y + vy * horizon:.3f}"
        out = String()
        out.data = ",".join(f)
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = TrackDropoutRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
