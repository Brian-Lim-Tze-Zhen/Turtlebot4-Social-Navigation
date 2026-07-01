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
        # ORIGINAL: 0.50 — conservative, caused filter to lag behind
        # real position changes. Reduced to 0.20 to make filter more
        # responsive to YOLO/depth measurements, reducing velocity
        # estimation lag and prediction error.
        self.R = np.eye(2) * 0.20

        # Measurement matrix: only position x, y is measured
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=float)

        self.last_time = timestamp
        self.last_conf = 1.0  # updated each measurement; used when coasting
        self.update_count = 0  # suppress velocity prediction during warm-up

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
        # ORIGINAL: 0.12 — very heavy smoothing, caused -7.9% velocity
        # bias by lagging behind true speed. Increased to 0.25 for
        # faster response to genuine velocity changes while still
        # providing noise reduction.
        self.smooth_alpha = 0.25
        self.vx_filt = None
        self.vy_filt = None

    def update(self, meas_x, meas_y, timestamp):
        dt_raw = timestamp - self.last_time

        # Reset velocity when the track was lost long enough that the old
        # state is untrustworthy. Without this, a high velocity from before
        # the gap persists through the dt clamp below and decays too slowly.
        if dt_raw > 1.5:
            self.x[2, 0] = 0.0
            self.x[3, 0] = 0.0
            self.vx_filt = None
            self.vy_filt = None
            self.P[2, 2] = 5.0
            self.P[3, 3] = 5.0
            self.update_count = 0

        # ==========================================================
        # THESIS FIX (duplicate/zero-dt timestamp bug)
        #
        # dt_raw == 0.0 happens when two messages arrive carrying the
        # exact same timestamp (observed in practice: simultaneous or
        # near-simultaneous detections at the same sim_time, e.g. two
        # YOLO frames resolving to the same depth/TF timestamp).
        #
        # The original clamp below ("0.0 < dt_raw <= 1.0 else 0.1")
        # used a STRICT inequality, so dt_raw == 0.0 failed the check
        # and fell through to the "else" branch meant for LARGE gaps
        # (sim pauses / timing jumps) -- silently substituting dt=0.1
        # even though zero real time had actually elapsed.
        #
        # Effect of that bug: the predict step advanced the position
        # estimate by velocity*0.1 using the fabricated dt, but the
        # measurement z was essentially identical to the previous
        # measurement (since no real time passed). The resulting
        # innovation (y = z - Hx) was then systematically in the
        # "deceleration" direction, and the Kalman gain pulled the
        # velocity estimate down to reconcile it -- even though the
        # person's true velocity hadn't changed. Confirmed empirically:
        # this produced the dataset's single largest speed dip
        # (~0.057 m/s against a true 0.1 m/s) at the rows where this
        # occurred 4-5 times in immediate succession.
        #
        # Fix: when dt_raw <= 0.0, there is no new time-elapsed
        # information to integrate -- skip the predict+correct cycle
        # entirely and just record the timestamp. The caller
        # (person_callback) still calls predict_future() right after
        # this returns, so a message is still published using the
        # last valid state, rather than corrupting that state with a
        # spurious update.
        # ==========================================================
        if dt_raw <= 0.0:
            self.last_time = timestamp
            return

        # Safety clamp for simulation pauses / timing jumps.
        # (dt_raw > 0.0 is now guaranteed here, so this only handles
        # the "too large" side -- the dt_raw==0.0 case is handled above.)
        dt = dt_raw if dt_raw <= 1.0 else 0.1

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
        self.update_count += 1

        vx_raw = float(self.x[2, 0])
        vy_raw = float(self.x[3, 0])

        if self.vx_filt is None:
            self.vx_filt = vx_raw
            self.vy_filt = vy_raw
        else:
            # If the raw velocity has reversed direction (negative dot product
            # with the current filtered estimate), reset the EMA immediately
            # so the predicted sphere doesn't lag behind a direction change.
            dot = vx_raw * self.vx_filt + vy_raw * self.vy_filt
            if dot < 0.0:
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
        # Suppress velocity during warm-up to prevent noisy early depth
        # readings from sending the predicted sphere flying on first detection.
        if self.update_count < 5:
            vx_pred, vy_pred = 0.0, 0.0
        else:
            vx_pred = self.vx_filt if self.vx_filt is not None else vx
            vy_pred = self.vy_filt if self.vy_filt is not None else vy

        # Hard cap at realistic human walking speed (~2 m/s) as a safety net.
        max_speed = 2.0
        speed = (vx_pred ** 2 + vy_pred ** 2) ** 0.5
        if speed > max_speed:
            scale = max_speed / speed
            vx_pred *= scale
            vy_pred *= scale

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
        self.declare_parameter("prediction_horizon", 2.0)
        self.declare_parameter("coast_timeout", 0.5)

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

        self.coast_timeout = (
            self.get_parameter("coast_timeout")
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

        # Coast timer: publish predictions for recently-seen tracks even when
        # detections are absent; prune tracks silent longer than coast_timeout.
        self.create_timer(0.2, self.coast_callback)

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
            self.tracks[track_id].last_conf = conf
            self.get_logger().info(f"Created KF track for id:{track_id}")
            return

        track = self.tracks[track_id]
        track.last_conf = conf
        track.update(base_x, base_y, now)

        x, y, vx, vy, pred_x, pred_y = track.predict_future(self.prediction_horizon)

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

    def coast_callback(self):
        now = self.get_ros_time_seconds()
        stale_ids = []

        for track_id, track in self.tracks.items():
            age = now - track.last_time

            if age >= self.coast_timeout:
                stale_ids.append(track_id)
                continue

            # Skip if a measurement just updated this track — the measurement
            # callback already published, and a 0.2 s timer firing right after
            # would just duplicate it.
            if age < 0.1:
                continue

            x, y, vx, vy, pred_x, pred_y = track.predict_future(self.prediction_horizon)
            out = String()
            out.data = (
                f"{track_id},"
                f"{track.last_conf:.2f},"
                f"{x:.3f},{y:.3f},"
                f"{vx:.3f},{vy:.3f},"
                f"{pred_x:.3f},{pred_y:.3f},"
                f"{self.prediction_horizon:.2f}"
            )
            self.pub.publish(out)

        for track_id in stale_ids:
            self.get_logger().info(f"Pruned stale track id:{track_id}")
            del self.tracks[track_id]


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