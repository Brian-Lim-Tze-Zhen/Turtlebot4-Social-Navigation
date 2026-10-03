#!/usr/bin/env python3
import time
import math
import cv2
import numpy as np
import os
import torch #(gemini to uncomment)
import subprocess
torch.set_num_threads(2)  # or 1 — start low and measure (gemini to uncomment)

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge

from ultralytics import YOLO

import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped




class YoloByteTrackPositionNode(Node):
    def __init__(self):
        super().__init__("yolo_bytetrack_position_node")

        self.bridge = CvBridge()

        self.rgb_topic = "/oakd/rgb/preview/image_raw"
        self.depth_topic = "/oakd/rgb/preview/depth"
        self.camera_info_topic = "/oakd/rgb/preview/camera_info"

        self.camera_frame = "oakd_rgb_camera_optical_frame"
        self.target_frame = "map"

        export_path = "/root/thesis_social_navigation_ws/models/yolov8n_openvino_model/"
        if not os.path.isdir(export_path):
            self.get_logger().info("OpenVINO export not found, exporting once...")
            YOLO("yolov8n.pt").export(format="openvino", imgsz=320)
        self.model = YOLO(export_path)

        self.frame_count = 0
        self.process_every_n_frames = 2
        self.show_debug_image = False

        self.latest_depth = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # Depth filtering
        self.min_depth = 0.2
        # was: self.max_depth = 7.0
        self.max_depth = 10.0
        self.min_depth_pixels = 50
        self.max_depth_std = 0.30  # meters; rejects bimodal body+background patches
        # Jump rejection in map frame
        self.last_positions = {}   # track_id -> (x, y, timestamp)
        self.max_speed = 2.5  # m/s; margin above human_kf_predictor's 2.0 m/s hard cap
        self.jump_timeout = 2.0    # seconds; stale entry skips jump check

        self.pub = self.create_publisher(String, "/person_positions_map", 10)

        # ==========================================================
        # THESIS FIX (frame staleness)
        #
        # Default depth-10 queue let up to 10 frames back up, so the
        # callback always processed the OLDEST queued frame. Measured
        # 325 ms mean frame age at callback entry -- roughly 8 frames
        # at the ~25 Hz camera rate. depth=1 + BEST_EFFORT drops late
        # frames instead of queueing them, so the newest frame is
        # always the one processed.
        # ==========================================================
        sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )

        self.create_subscription(Image, self.rgb_topic, self.rgb_callback, sensor_qos)
        self.create_subscription(Image, self.depth_topic, self.depth_callback, sensor_qos)
        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, 10)

        self.tf_buffer = tf2_ros.Buffer(
            cache_time=rclpy.duration.Duration(seconds=10.0)
        )
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info("YOLO + ByteTrack + depth position node started")
        self.get_logger().info(f"RGB topic: {self.rgb_topic}")
        self.get_logger().info(f"Depth topic: {self.depth_topic}")
        self.get_logger().info(f"CameraInfo topic: {self.camera_info_topic}")
        self.get_logger().info(f"Publishing: /person_positions_map")
        self.get_logger().info(f"Target frame: {self.target_frame}")

    def camera_info_callback(self, msg):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def depth_callback(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            if depth.dtype == np.uint16:
                depth = depth.astype(np.float32) / 1000.0
            else:
                depth = depth.astype(np.float32)

            self.latest_depth = depth

        except Exception as e:
            self.get_logger().warn(f"Depth conversion failed: {e}")

    def get_valid_depth_from_bbox(self, x1, y1, x2, y2):
        if self.latest_depth is None:
            return None, None, None

        h, w = self.latest_depth.shape[:2]

        x1 = max(0, min(w - 1, int(x1)))
        x2 = max(0, min(w - 1, int(x2)))
        y1 = max(0, min(h - 1, int(y1)))
        y2 = max(0, min(h - 1, int(y2)))

        if x2 <= x1 or y2 <= y1:
            return None, None, None

        # ==========================================================
        # THESIS FIX (leg-gap depth contamination)
        #
        # Previously sampled the lower-body region (60-90% height).
        # During a normal walking gait, at moderate range (~1-1.5m
        # confirmed in testing) the legs are spread far enough apart in
        # -frame that this patch can straddle the gap between them,
        # mixing body-surface depth with floor/background depth behind
        # the person. np.median() on a bimodal patch like this can
        # return a value in the gap, or be dominated by the background
        # cluster if it has more valid pixels -- producing a depth
        # reading that places the person's estimated position BEHIND
        # their real body.
        #
        # Confirmed empirically: id:14 log showed an ACCEPTED position
        # jump from depth=1.65m to depth=2.35m within ~0.2s while the
        # person walks at ~0.2 m/s toward the robot (depth should have
        # decreased slightly, not jumped +0.7m) -- consistent with a
        # stray reading catching a surface behind the person, not
        # genuine motion. Several similar readings were correctly
        # rejected by the jump filter, but this one landed just under
        # max_jump and slipped through.
        #
        # Fix: sample the torso (30-55% height) instead of the lower
        # body. This region is a single contiguous body mass that a
        # walking gait cannot split apart, unlike the legs.
        # ==========================================================
        u1 = int(x1 + 0.35 * (x2 - x1))
        u2 = int(x1 + 0.65 * (x2 - x1))
        v1 = int(y1 + 0.30 * (y2 - y1))
        v2 = int(y1 + 0.55 * (y2 - y1))

        u1 = max(0, min(w - 1, u1))
        u2 = max(0, min(w - 1, u2))
        v1 = max(0, min(h - 1, v1))
        v2 = max(0, min(h - 1, v2))

        if u2 <= u1 or v2 <= v1:
            return None, None, None

        patch = self.latest_depth[v1:v2, u1:u2]

        valid = patch[np.isfinite(patch)]
        valid = valid[(valid > self.min_depth) & (valid < self.max_depth)]

        if valid.size < self.min_depth_pixels:
            return None, None, None

        # ==========================================================
        # THESIS FIX (bimodal patch rejection)
        #
        # A clean single-surface patch (torso) should have low depth
        # variance. A patch that still straddles two different surfaces
        # (e.g. an arm swung away from the body, partial occlusion, or
        # edge-of-bbox spillover onto background) shows up as an
        # unusually wide spread, since it's really two different depths
        # mixed together. Reject those rather than trusting median() to
        # silently paper over a bimodal distribution -- this is a
        # second line of defense on top of the torso-window fix above,
        # not a replacement for it.
        #
        # max_depth_std is a starting value, not a measured optimum:
        # a healthy torso patch should show only a few cm of spread
        # from body curvature/tilt; a patch mixing body + background
        # typically shows tens of cm to meters of spread. Re-tune if
        # this starts rejecting valid frames too often (raise it) or
        # letting bad frames through (lower it).
        # ==========================================================
        depth_std = float(np.std(valid))
        if depth_std > self.max_depth_std:
            return None, None, None

        depth = float(np.median(valid))

        u = int((u1 + u2) / 2)
        v = int((v1 + v2) / 2)

        return depth, u, v

    def pixel_to_camera_xyz(self, u, v, depth):
        if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
            self.get_logger().warn("Waiting for camera_info")
            return None

        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth

        return x, y, z

    def transform_camera_to_target(self, x, y, z, stamp=None):
        point_cam = PointStamped()

        # ==========================================================
        # THESIS FIX (rotation-induced jump rejection)
        #
        # Previously always used rclpy.time.Time() ("latest available TF")
        # rather than the image's actual capture stamp. This is fine while
        # the robot is stationary or moving in a straight line, but during
        # active rotation (exactly when avoidance manoeuvres happen), the
        # map<-odom<-base_link<-camera TF chain changes fast enough that
        # transforming a slightly-stale detection with a *newer* TF injects
        # a rotation-induced position offset. That offset was large enough
        # to regularly exceed max_jump, causing detections to be rejected
        # right when the pipeline needed them most (during robot turns).
        #
        # Fix: use the RGB message's own header.stamp when available, so
        # the transform reflects the TF at actual capture time.
        #
        # THESIS FIX 2 (TF lag fallback)
        #
        # In practice, this sim's TF publishing lags the image timestamp by
        # ~0.2-0.3s consistently (confirmed via repeated "extrapolation
        # into the future" warnings), which exceeds the lookup timeout.
        # Using the image stamp unconditionally then means EVERY detection
        # fails the TF lookup and gets silently dropped -- worse than the
        # original behaviour. So: try the accurate image-stamp transform
        # first: if TF hasn't caught up yet, fall back to "latest" rather
        # than losing the detection entirely. This keeps the rotation-jump
        # fix for the common case (TF caught up) while never being worse
        # than the old behaviour when TF is lagging.
        # ==========================================================
        point_cam.header.frame_id = self.camera_frame
        point_cam.point.x = float(x)
        point_cam.point.y = float(y)
        point_cam.point.z = float(z)

        stamp_to_try = stamp if stamp is not None else rclpy.time.Time().to_msg()
        point_cam.header.stamp = stamp_to_try

        try:
            point_target = self.tf_buffer.transform(
                point_cam,
                self.target_frame,
                timeout=rclpy.duration.Duration(seconds=0.05)
            )

            return (
                point_target.point.x,
                point_target.point.y,
                point_target.point.z
            )

        except Exception as e:
            # If the failure was specifically due to TF lag (extrapolation
            # into the future) and we were trying the image stamp, retry
            # once against "latest available" instead of dropping the
            # detection outright.
            if stamp is not None:
                try:
                    point_cam.header.stamp = rclpy.time.Time().to_msg()
                    point_target = self.tf_buffer.transform(
                        point_cam,
                        self.target_frame,
                        timeout=rclpy.duration.Duration(seconds=0.05)
                    )
                    return (
                        point_target.point.x,
                        point_target.point.y,
                        point_target.point.z
                    )
                except Exception as e2:
                    self.get_logger().warn(f"TF transform failed (both stamps): {e2}")
                    return None

            self.get_logger().warn(f"TF transform failed: {e}")
            return None

    def rgb_callback(self, msg):
        # DROP SKIPPED FRAMES IMMEDIATELY (0 CPU COST)
        self.frame_count += 1
        if self.frame_count % self.process_every_n_frames != 0:
            return

        t_cb_start = time.monotonic()
        msg_age = self.get_clock().now().nanoseconds * 1e-9 - (
            msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"RGB conversion failed: {e}")
            return

        display_frame = frame.copy()

        if self.latest_depth is None:
            cv2.putText(
                display_frame,
                "Waiting for depth...",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

            if self.show_debug_image:
                cv2.imshow("YOLO ByteTrack Position", display_frame)
                cv2.waitKey(1)

            return

    
        t0 = time.monotonic()
        results = self.model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                conf=0.70,
                imgsz=320,
                device ="cpu", # Use 0 for GPU, or "cpu" if running on CPU (gemini)
                verbose=False
            )
        inference_ms = (time.monotonic() - t0) * 1000.0

        if results is None or len(results) == 0:
            if self.show_debug_image:
                cv2.putText(
                    display_frame,
                    "No YOLO result",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )
                cv2.imshow("YOLO ByteTrack Position", display_frame)
                cv2.waitKey(1)
            return

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            if self.show_debug_image:
                cv2.putText(
                    display_frame,
                    "No person detected",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )
                cv2.imshow("YOLO ByteTrack Position", display_frame)
                cv2.waitKey(1)
            return

        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = xyxy.astype(int)

            conf = float(box.conf[0].cpu().numpy())
            track_id = -1 if box.id is None else int(box.id[0].cpu().numpy())

            depth, u, v = self.get_valid_depth_from_bbox(x1, y1, x2, y2)

            if depth is None:
                if self.show_debug_image:
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        display_frame,
                        f"ID:{track_id} bad depth",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2
                    )
                continue

            camera_xyz = self.pixel_to_camera_xyz(u, v, depth)
            if camera_xyz is None:
                continue

            cam_x, cam_y, cam_z = camera_xyz

            map_point = self.transform_camera_to_target(cam_x, cam_y, cam_z, stamp=msg.header.stamp)

            if map_point is None:
                if self.show_debug_image:
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cv2.putText(
                        display_frame,
                        f"ID:{track_id} TF fail",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 165, 255),
                        2
                    )
                continue

            map_x, map_y, map_z = map_point

            # Reject impossible map-frame jumps for the same ByteTrack ID.
            # Skip the check when the stored entry is stale — the person may
            # have genuinely moved or reappeared from occlusion.
            # Stamp from the image capture time, not the processing clock.
            # Processing time drifts relative to capture time once frames are
            # dropped (depth=1 BEST_EFFORT), producing dt values far smaller
            # than the true interval between the two observed positions --
            # measured implied_speed up to 11 m/s for a genuine 0.24 m step.
            now_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if track_id in self.last_positions:
                last_x, last_y, last_t = self.last_positions[track_id]
                # was:
                #     if now_sec - last_t <= self.jump_timeout:
                #         jump = math.hypot(map_x - last_x, map_y - last_y)
                #         if jump > self.max_jump:
                if 0.12 < (now_sec - last_t) <= self.jump_timeout:
                    jump = math.hypot(map_x - last_x, map_y - last_y)
                    implied_speed = jump / (now_sec - last_t)
                    if implied_speed > self.max_speed:
                        self.get_logger().warn(
                            f"Reject jump id:{track_id}, "
                            f"jump={jump:.2f} m, implied_speed={implied_speed:.2f} m/s, "
                            f"new=({map_x:.2f},{map_y:.2f}), "
                            f"last=({last_x:.2f},{last_y:.2f})"
                        )

                        if self.show_debug_image:
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(
                                display_frame,
                                f"ID:{track_id} rejected jump",
                                (x1, max(20, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 255),
                                2
                            )

                        # Anchor forward even on rejection: leaving last_positions
                        # stale makes the next comparison span a larger gap, so one
                        # rejection cascades into several. Observed 5 rejections per
                        # approach, in runs of 2-3 against an unchanged anchor.
                        self.last_positions[track_id] = (map_x, map_y, now_sec)
                        continue

            self.last_positions[track_id] = (map_x, map_y, now_sec)

            # THESIS ADDITION (group formation support): append the pixel
            # bbox corners as trailing fields. x1,y1,x2,y2 are already in
            # scope from this loop's xyxy.astype(int) above. Old consumers
            # parsing only the first 7 fields are unaffected since the
            # existing fields keep their original order/meaning.
            out = String()
            out.data = (
                f"{track_id},"
                f"{conf:.2f},"
                f"{map_x:.3f},"
                f"{map_y:.3f},"
                f"{depth:.3f},"
                f"{u},"
                f"{v},"
                f"{x1},"
                f"{y1},"
                f"{x2},"
                f"{y2}"
            )
            self.pub.publish(out)

            self.get_logger().info(
                f"id:{track_id} conf:{conf:.2f} "
                f"pixel=({u},{v}) "
                f"depth={depth:.2f} m "
                f"camera_xyz=({cam_x:.2f},{cam_y:.2f},{cam_z:.2f}) m "
                f"map_xy=({map_x:.2f},{map_y:.2f}) m "
                f"bbox=({x1},{y1},{x2},{y2}) h={y2-y1}px"   # TEMP DIAGNOSTIC
            )

            if self.show_debug_image:
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(display_frame, (u, v), 4, (255, 0, 0), -1)

                cv2.putText(
                    display_frame,
                    f"ID:{track_id} conf:{conf:.2f}",
                    (x1, max(20, y1 - 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    display_frame,
                    f"map x={map_x:.2f}, y={map_y:.2f}, depth={depth:.2f}m",
                    (x1, max(40, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

        if self.show_debug_image:
            cv2.imshow("YOLO ByteTrack Position", display_frame)
            cv2.waitKey(1)

        cb_ms = (time.monotonic() - t_cb_start) * 1000.0


def main(args=None):
    n = subprocess.run(["pgrep", "-fc", "python3.*yolo_detector"],
                       capture_output=True, text=True).stdout.strip()
    if n and int(n) > 1:
        print(f"WARNING: {n} yolo_detector processes running - kill the others first")

    rclpy.init(args=args)

    node = YoloByteTrackPositionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == "__main__":
    main()