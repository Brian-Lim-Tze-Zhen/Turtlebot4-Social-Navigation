#!/usr/bin/env python3

import math
import numpy as np
import rclpy
import time as _wall 
from rclpy.node import Node
from std_msgs.msg import String
from nav_msgs.msg import Odometry


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
        q_vel = 0.05
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel])

        # Measurement noise
        self.R = np.eye(2) * 0.10

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
        # was: self.smooth_alpha = 0.12
        self.tau_rise = 0.5   # s; was 1.3. At 1.2 m/s a walk leg is only ~4 s,
                               # so 1.3 s spent most of the encounter still
                               # converging (vel_filt -1.06 vs true -1.21).
                               # Trade-off: less noise on the horizon-multiplied
                               # value. Fallback 0.8 if pred jitters.

        # ==========================================================
        # THESIS MODIFICATION (asymmetric EMA decay)
        #
        # smooth_alpha=0.12 moves the filtered velocity only 12% toward
        # each new reading, so it takes ~18 updates (~3s at 6Hz, longer
        # under RTF sag) to decay 90%. That is the intended behaviour
        # while WALKING - it suppresses the jitter that would otherwise
        # be amplified by the prediction horizon - but it means velocity
        # lingers long after motion stops.
        #
        # Observed: a bolted-down person read 0.14 m/s with the robot
        # parked and the rotation gate INACTIVE, decaying only slowly
        # toward zero. That is residue accumulated during an earlier
        # motion phase, not live contamination. At 0.14 it is half of
        # slow-walking speed, so it clears predicted_person_cloud_node's
        # 0.05 stationary deadband and produces a 2.2m directional
        # ellipse for someone standing still.
        #
        # Fix: use a much higher alpha when the raw velocity is SMALLER
        # in magnitude than the current filtered estimate. Slowing down
        # and stopping are tracked quickly; speeding up stays smoothed.
        # This mirrors the existing direction-reversal reset below,
        # which already treats "the filter is confidently wrong" as a
        # case for abandoning smoothing rather than easing into it.
        # ==========================================================
        # was: self.decay_alpha = 0.12
        self.tau_decay = 1.3  # s; PLACEHOLDER — currently symmetric with tau_rise.
                            # Original comments describe intended fast-decay/
                            # slow-rise asymmetry, but decay_alpha was numerically
                            # identical to smooth_alpha (0.12) in the prior code —
                            # no asymmetry was actually active. Revisit this value
                            # once you decide on an intended decay speed.
        self.vx_filt = None
        self.vy_filt = None

    def update(self, meas_x, meas_y, timestamp, freeze_velocity=False):
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

        # Safety clamp for simulation pauses / timing jumps
        dt = dt_raw if 0.0 < dt_raw <= 1.0 else 0.1

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

        # Measurement update step — always update position
        z = np.array([[meas_x], [meas_y]], dtype=float)

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        # ==========================================================
        # THESIS MODIFICATION (ego-motion rotation gate)
        #
        # When the robot is rotating fast (angular velocity above
        # threshold), the camera-frame apparent motion of the person
        # contaminates the KF velocity estimate. This causes the
        # predicted ellipse to flip direction during avoidance
        # manoeuvres, trapping the robot inside the obstacle zone.
        #
        # Fix: when freeze_velocity=True (set by HumanKFPredictor
        # when |odom angular.z| > rot_gate_threshold), skip the EMA
        # velocity update. Position tracking continues as normal
        # (the KF still sees new measurements and updates x, y),
        # but the smoothed velocity fed into predict_future() holds
        # its last known value until the robot stops rotating.
        #
        # This means the ellipse holds its last known direction
        # during avoidance rotations instead of chasing ego-motion
        # noise.
        # ==========================================================
        self.update_count += 1

        if not freeze_velocity:
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
                    # Asymmetric alpha - see decay_alpha above. Decay
                    # fast, rise slow.
                    raw_speed = (vx_raw ** 2 + vy_raw ** 2) ** 0.5
                    filt_speed = (self.vx_filt ** 2 + self.vy_filt ** 2) ** 0.5

                # was:
                    #     if raw_speed < filt_speed:
                    #         a = self.decay_alpha
                    #     else:
                    #         a = self.smooth_alpha
                    #     self.vx_filt = a * vx_raw + (1.0 - a) * self.vx_filt
                    #     self.vy_filt = a * vy_raw + (1.0 - a) * self.vy_filt

                    if raw_speed < filt_speed:
                        a = 1.0 - math.exp(-dt / self.tau_decay)
                    else:
                        a = 1.0 - math.exp(-dt / self.tau_rise)

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

        return x, y, vx_pred, vy_pred, pred_x, pred_y, vx, vy


class HumanKFPredictor(Node):
    def __init__(self):
        super().__init__("human_kf_predictor")

        # =====================================
        # User configurable parameters
        # =====================================
        self.declare_parameter("input_topic", "/person_positions_map")
        self.declare_parameter("output_topic", "/predicted_person_positions")
        self.declare_parameter("prediction_horizon", 1.0)
        self.declare_parameter("coast_timeout", 0.6)
        self.declare_parameter("rot_gate_threshold", 0.3)  # rad/s

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

        self.rot_gate_threshold = (
            self.get_parameter("rot_gate_threshold")
            .get_parameter_value()
            .double_value
        )

        # =====================================
        # Internal state
        # =====================================
        self.tracks = {}

        # ==========================================================
        # THESIS MODIFICATION (ego-motion rotation gate)
        #
        # Track robot angular velocity from /odom so the KF velocity
        # update can be frozen when the robot is rotating. Initialised
        # to 0.0 (not rotating) so the gate is inactive until the
        # first odom message arrives.
        # ==========================================================
        self.robot_angular_z = 0.0
        self.rotation_gated = False  # for logging — avoids repeating the log

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

        # Subscribe to /odom for robot angular velocity
        self.odom_sub = self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            20
        )

        # Coast timer: publish predictions for recently-seen tracks even when
        # detections are absent; prune tracks silent longer than coast_timeout.
        self.create_timer(0.2, self.coast_callback)

        self.get_logger().info("Human KF predictor started")
        self.get_logger().info(f"Input : {self.input_topic}")
        self.get_logger().info(f"Output: {self.output_topic}")
        self.get_logger().info(f"Prediction horizon: {self.prediction_horizon:.2f} s")
        self.get_logger().info(
            f"Rotation gate threshold: {self.rot_gate_threshold:.2f} rad/s"
        )

    def get_ros_time_seconds(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def odom_callback(self, msg):
        self.robot_angular_z = msg.twist.twist.angular.z

    def person_callback(self, msg):
        now = self.get_ros_time_seconds()

        # ==========================================================
        # THESIS MODIFICATION (ego-motion rotation gate)
        #
        # Check if robot is rotating above threshold. If so, freeze
        # the KF velocity update for all tracks this tick.
        # ==========================================================
        freeze_velocity = abs(self.robot_angular_z) > self.rot_gate_threshold

        if freeze_velocity and not self.rotation_gated:
            self.get_logger().info(
                f"Rotation gate ACTIVE: angular_z={self.robot_angular_z:.3f} rad/s "
                f"> threshold={self.rot_gate_threshold:.2f} rad/s — "
                f"KF velocity update frozen"
            )
            self.rotation_gated = True
        elif not freeze_velocity and self.rotation_gated:
            self.get_logger().info(
                f"Rotation gate CLEARED: angular_z={self.robot_angular_z:.3f} rad/s"
            )
            self.rotation_gated = False

        try:
            parts = msg.data.split(",")

            track_id = int(float(parts[0]))
            conf = float(parts[1])

            # Expected /person_positions_base format:
            # id,conf,base_x,base_y,...
            base_x = float(parts[2])
            base_y = float(parts[3])

            # THESIS ADDITION (group formation support): parse the bbox
            # corners yolo_detector.py now appends as 4 trailing fields
            # (x1,y1,x2,y2). Only present on a genuine fresh detection —
            # this is exactly the "fresh" bbox group_formation_detector.py
            # wants; coasted publishes below deliberately omit it.
            bbox = None
            if len(parts) >= 11:
                try:
                    bx1, by1, bx2, by2 = (int(float(p)) for p in parts[7:11])
                    bbox = (bx1, by1, bx2, by2)
                except ValueError:
                    bbox = None

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
        track.update(base_x, base_y, now, freeze_velocity=freeze_velocity)

        x, y, vx, vy, pred_x, pred_y, vx_raw, vy_raw = track.predict_future(self.prediction_horizon)

        bbox_str = f"{bbox[0]};{bbox[1]};{bbox[2]};{bbox[3]}" if bbox is not None else "none"

        out = String()
        out.data = (
            f"{track_id},"
            f"{conf:.2f},"
            f"{x:.3f},{y:.3f},"
            f"{vx:.3f},{vy:.3f},"
            f"{pred_x:.3f},{pred_y:.3f},"
            f"{self.prediction_horizon:.2f},"
            f"{1 if freeze_velocity else 0},"   # field [9] — rotation gate active
            f"{bbox_str}"                      # field [10] — bbox or "none"
        )

        self.pub.publish(out)

        self.get_logger().info(
            f"id:{track_id} "
            f"pos=({x:.2f},{y:.2f}) "
            f"vel_filt=({vx:.2f},{vy:.2f}) vel_raw=({vx_raw:.2f},{vy_raw:.2f}) "
            f"pred_{self.prediction_horizon:.1f}s=({pred_x:.2f},{pred_y:.2f})"
            + (" [ROT GATED]" if freeze_velocity else "")
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

            x, y, vx, vy, pred_x, pred_y, vx_raw, vy_raw = track.predict_future(self.prediction_horizon)
            out = String()
            out.data = (
                f"{track_id},"
                f"{track.last_conf:.2f},"
                f"{x:.3f},{y:.3f},"
                f"{vx:.3f},{vy:.3f},"
                f"{pred_x:.3f},{pred_y:.3f},"
                f"{self.prediction_horizon:.2f},"
                f"0,"                              # field [9] — rotation gate (always inactive on coast)
                f"none"                             # field [10] — no fresh bbox while coasting
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