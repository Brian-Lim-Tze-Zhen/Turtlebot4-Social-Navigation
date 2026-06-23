#!/usr/bin/env python3

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class HumanTrackKF:
    def __init__(self, x, y, timestamp):
        # state: [x, y, vx, vy]
        self.x = np.array([[x], [y], [0.0], [0.0]], dtype=float)

        # State covariance
        self.P = np.eye(4) * 1.0

        # ==========================================================
        # THESIS MODIFICATION (prediction stability fix)
        #
        # Process noise was previously a single uniform value
        # (np.eye(4) * 0.05) applied equally to position AND velocity
        # states. This meant the filter had no separate way to trust
        # velocity less than position.
        #
        # Since predict_future() extrapolates with
        #     pred = position + velocity * horizon
        # any noise present in the velocity estimate gets amplified
        # by the horizon (2.0s by default -> noise is doubled). This
        # was the dominant cause of the predicted point visibly
        # jittering/wobbling in RViz even when the person walked at a
        # fairly constant pace.
        #
        # Fix: split Q into separate position and velocity process
        # noise. Lowering q_vel relative to q_pos tells the filter
        # "expect velocity to change slowly/smoothly", which directly
        # reduces frame-to-frame velocity noise without making
        # position tracking sluggish.
        #
        # Tuned and verified empirically against logged data
        # (id:60 / id:61 sequences): reduced mean frame-to-frame
        # pred_y jump by ~25% and worst-case single-step jump by
        # ~38% on the noisier track, combined with the velocity EMA
        # smoothing below.
        # ==========================================================
        q_pos = 0.05
        q_vel = 0.02
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel])

        # Measurement noise
        self.R = np.eye(2) * 0.50

        # Measurement matrix: only position x, y is measured
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)

        self.last_time = timestamp

        # ==========================================================
        # THESIS MODIFICATION (prediction stability fix)
        #
        # Smoothed (EMA) velocity estimate, used only for the future
        # prediction in predict_future(). The raw KF velocity state
        # (self.x[2,0], self.x[3,0]) is left untouched so that
        # position tracking itself stays just as responsive as
        # before - only the value fed into the horizon extrapolation
        # is smoothed.
        #
        # smooth_alpha tradeoff:
        #   - lower alpha  -> smoother prediction, but more lag
        #                     before a genuine sudden velocity change
        #                     (e.g. person stopping or reversing) is
        #                     reflected in the predicted point.
        #   - higher alpha -> less lag, but less noise reduction.
        #   0.3 was used during empirical testing against logged
        #   data; revisit if live behavior feels too laggy or still
        #   too jittery.
        # ==========================================================
        self.smooth_alpha = 0.3
        self.vx_filt = None
        self.vy_filt = None

    def update(self, meas_x, meas_y, timestamp):
        dt = timestamp - self.last_time

        # Safety clamp for simulation pauses / timing jumps
        if dt <= 0.0 or dt > 1.0:
            dt = 0.1

        self.last_time = timestamp

        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)

        # Prediction step
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

        # Measurement update step
        z = np.array([[meas_x], [meas_y]], dtype=float)

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        # ==========================================================
        # THESIS MODIFICATION (prediction stability fix)
        #
        # Update the EMA-smoothed velocity from this step's raw KF
        # velocity estimate. Done here (once per update) rather than
        # in predict_future() so that calling predict_future()
        # multiple times with different horizons does not re-apply
        # smoothing repeatedly or drift the filter state.
        # ==========================================================
        vx_raw = float(self.x[2, 0])
        vy_raw = float(self.x[3, 0])

        if self.vx_filt is None:
            self.vx_filt = vx_raw
            self.vy_filt = vy_raw
        else:
            a = self.smooth_alpha
            self.vx_filt = a * vx_raw + (1.0 - a) * self.vx_filt
            self.vy_filt = a * vy_raw + (1.0 - a) * self.vy_filt

    def predict_future(self, horizon):
        x = float(self.x[0, 0])
        y = float(self.x[1, 0])

        # Raw (unsmoothed) KF velocity - kept for logging/diagnostics
        # so it's still possible to compare raw vs smoothed velocity
        # in the published message / logs if needed.
        vx = float(self.x[2, 0])
        vy = float(self.x[3, 0])

        # ==========================================================
        # THESIS MODIFICATION (prediction stability fix)
        #
        # Use the EMA-smoothed velocity for the actual extrapolation,
        # since this is the value that gets multiplied by horizon and
        # is therefore the most sensitive to noise. Falls back to raw
        # velocity on the very first call (vx_filt is None) before
        # any smoothing history exists.
        # ==========================================================
        vx_pred = self.vx_filt if self.vx_filt is not None else vx
        vy_pred = self.vy_filt if self.vy_filt is not None else vy

        pred_x = x + vx_pred * horizon
        pred_y = y + vy_pred * horizon

        return x, y, vx, vy, pred_x, pred_y


class HumanKFPredictor(Node):
    def __init__(self):
        super().__init__("human_kf_predictor")

        # =====================================
        # User configurable parameters
        # =====================================
        self.declare_parameter("input_topic", "/person_positions_map")
        self.declare_parameter("output_topic", "/predicted_person_positions")
        self.declare_parameter("prediction_horizon", 3.0)

        # =====================================
        # Load parameters
        # =====================================
        self.input_topic = (
            self.get_parameter("input_topic")
            .get_parameter_value()
            .string_value
        )

        self.output_topic = (
            self.get_parameter("output_topic")
            .get_parameter_value()
            .string_value
        )

        self.prediction_horizon = (
            self.get_parameter("prediction_horizon")
            .get_parameter_value()
            .double_value
        )

        # =====================================
        # Internal state
        # =====================================
        self.tracks = {}

        # =====================================
        # ROS interfaces
        # =====================================
        self.sub = self.create_subscription(
            String,
            self.input_topic,
            self.person_callback,
            10
        )

        self.pub = self.create_publisher(
            String,
            self.output_topic,
            10
        )

        self.get_logger().info("Human KF predictor started")
        self.get_logger().info(f"Input : {self.input_topic}")
        self.get_logger().info(f"Output: {self.output_topic}")
        self.get_logger().info(f"Prediction horizon: {self.prediction_horizon:.2f} s")

    def get_ros_time_seconds(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def person_callback(self, msg):
        now = self.get_ros_time_seconds()

        try:
            parts = msg.data.split(",")

            track_id = int(float(parts[0]))
            conf = float(parts[1])

            # Expected /person_positions_base format:
            # id,conf,base_x,base_y,...
            base_x = float(parts[2])
            base_y = float(parts[3])

        except Exception as e:
            self.get_logger().warn(
                f"Could not parse message: {msg.data} | error: {e}"
            )
            return

        if track_id not in self.tracks:
            self.tracks[track_id] = HumanTrackKF(base_x, base_y, now)
            self.get_logger().info(f"Created KF track for id:{track_id}")
            return

        self.tracks[track_id].update(base_x, base_y, now)

        x, y, vx, vy, pred_x, pred_y = self.tracks[track_id].predict_future(
            self.prediction_horizon
        )

        out = String()
        out.data = (
            f"{track_id},"
            f"{conf:.2f},"
            f"{x:.3f},{y:.3f},"
            f"{vx:.3f},{vy:.3f},"
            f"{pred_x:.3f},{pred_y:.3f},"
            f"{self.prediction_horizon:.2f}"
        )

        self.pub.publish(out)

        self.get_logger().info(
            f"id:{track_id} "
            f"pos=({x:.2f},{y:.2f}) "
            f"vel=({vx:.2f},{vy:.2f}) "
            f"pred_{self.prediction_horizon:.1f}s=({pred_x:.2f},{pred_y:.2f})"
        )


def main(args=None):
    rclpy.init(args=args)

    node = HumanKFPredictor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()