#!/usr/bin/env python3
import math
import time
import cv2
import numpy as np
import os
import torch
import subprocess
torch.set_num_threads(2)

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge

from ultralytics import YOLO


class YoloByteTrackPositionNode(Node):
    def __init__(self):
        super().__init__("yolo_bytetrack_position_node")

        self.bridge = CvBridge()

        # THESIS FIX (topic mismatch): "/oakd/rgb/preview/image_raw/
        # compressed" has ZERO publishers in this sim setup (confirmed
        # via `ros2 topic info --verbose`) - it only appeared in
        # `ros2 topic list` because this node's own subscription
        # registers it. Nothing compresses/republishes the camera feed
        # here. The raw topic below has a real publisher (camera_bridge
        # node, sensor_msgs/Image, RELIABLE) - subscribing there
        # instead. A BEST_EFFORT subscriber can connect to a RELIABLE
        # publisher (downgrade is allowex`d by ROS2's QoS compatibility
        # rules), so sensor_qos below doesn't need to change.
        self.rgb_topic = "/oakd/rgb/preview/image_raw"

        export_path = "/root/thesis_social_navigation_ws/src/social_perception/social_perception/Lidar/yolov8n-pose_openvino_model/"
        if not os.path.isdir(export_path):
            self.get_logger().info("OpenVINO export not found, exporting once...")
            YOLO("yolov8n-pose.pt").export(format="openvino", imgsz=320)
        self.model = YOLO(export_path)

        # COCO keypoint indices: 13=L knee,14=R knee,15=L ankle,16=R ankle
        self.leg_kpt_idx = [13, 14, 15, 16]
        self.leg_kpt_min_conf = 0.3
        self.min_publish_conf = 0.2

        self.frame_count = 0
        self.process_every_n_frames = 2
        self.show_debug_image = True

        self.pub = self.create_publisher(String, "/person_positions_map", 10)

        # ==========================================================
        # THESIS ADDITION (bearing-based fusion)
        #
        # This node has no depth/TF, so it cannot compute a position -
        # only a 2D pixel location. To let identity_fusion_node.py
        # match a camera detection to a LIDAR cluster without a
        # position, we convert pixel x to a BEARING (angle from the
        # camera's optical axis) and match by angle instead of by
        # (x,y) distance.
        #
        # Proper way: pinhole model using camera_info's fx, cx -
        # bearing = atan2(px - cx, fx). Subscribed below; if
        # camera_info never arrives (unconfirmed whether this camera
        # publishes it), falls back to a linear pixel-fraction-of-FOV
        # approximation using FALLBACK_HFOV_DEG. That fallback is a
        # rough approximation (not a true pinhole projection) and its
        # FOV value is an assumed OAK-D RGB preview-stream figure, NOT
        # verified against this specific camera/crop - calibrate
        # against a known real-world bearing if the angular gate in
        # identity_fusion_node.py needs tightening.
        # ==========================================================
        self.camera_info_received = False
        self.fx = None
        self.cx = None
        self.FALLBACK_HFOV_DEG = 69.0  # unverified assumption - see note above
        self.create_subscription(
            CameraInfo, "/oakd/rgb/preview/camera_info",
            self._camera_info_callback, 10)

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

        self.get_logger().info("YOLO + ByteTrack leg-keypoint node started (no depth/TF - LIDAR fusion handles range)")
        self.get_logger().info(f"RGB topic: {self.rgb_topic}")
        self.get_logger().info(f"Publishing: /person_positions_map")

    def _camera_info_callback(self, msg):
        if not self.camera_info_received:
            self.fx = msg.k[0]
            self.cx = msg.k[2]
            self.camera_info_received = True
            self.get_logger().info(
                f"camera_info received: fx={self.fx:.1f}, cx={self.cx:.1f} "
                f"- using true pinhole bearing from here on")

    def _pixel_x_to_bearing(self, px, image_width):
        """Bearing in radians, positive = right of camera centre."""
        if self.camera_info_received:
            return math.atan2(px - self.cx, self.fx)
        # Fallback: linear pixel-fraction-of-FOV, not a true pinhole
        # projection - acceptable approximation only near image centre.
        frac = (px - image_width / 2.0) / image_width
        return frac * math.radians(self.FALLBACK_HFOV_DEG)

    def rgb_callback(self, msg):
        # DROP SKIPPED FRAMES IMMEDIATELY (0 CPU COST)
        self.frame_count += 1
        if self.frame_count % self.process_every_n_frames != 0:
            return

        # THESIS ADDITION (readability): this node previously logged
        # NOTHING at all when no person was confidently detected -
        # matching the "(idle)" heartbeat pattern already used in
        # lidar_person_detector.py so total silence in the terminal is
        # diagnosable (frames arriving, no person seen) rather than
        # ambiguous with "rgb_callback never fires at all".
        now_s = time.monotonic()
        if not hasattr(self, "_last_heartbeat"):
            self._last_heartbeat = 0.0
        if now_s - self._last_heartbeat > 5.0:
            self.get_logger().info(
                f"(alive) frame #{self.frame_count} processed - "
                f"camera_info={'yes' if self.camera_info_received else 'NOT YET'}")
            self._last_heartbeat = now_s

        t_cb_start = time.monotonic()
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"RGB conversion failed: {e}")
            return

        display_frame = frame.copy()

        t0 = time.monotonic()
        results = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=0.15,
            imgsz=320,
            device="cpu",
            verbose=False
        )
        inference_ms = (time.monotonic() - t0) * 1000.0

        if results is None or len(results) == 0:
            if self.show_debug_image:
                cv2.putText(display_frame, "No YOLO result", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow("YOLO ByteTrack Position", display_frame)
                cv2.waitKey(1)
            return

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            if self.show_debug_image:
                cv2.putText(display_frame, "No person detected", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow("YOLO ByteTrack Position", display_frame)
                cv2.waitKey(1)
            return

        for box_i, box in enumerate(result.boxes):
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = xyxy.astype(int)

            conf = float(box.conf[0].cpu().numpy())
            track_id = -1 if box.id is None else int(box.id[0].cpu().numpy())

            # Only publish/draw high-confidence detections. Kept the
            # model's own conf=0.15 threshold above so ByteTrack still
            # sees lower-confidence boxes for track continuity; this is
            # a separate, stricter gate on what actually gets used.
            if conf < self.min_publish_conf:
                continue

            # THESIS FIX (unassigned-track sentinel leak) - perception v2
            # -1 means ByteTrack did not associate this detection with any
            # existing track. It is a sentinel, not an identity. Publishing
            # it makes every unassigned detection collide in the same
            # downstream dict slot, producing one phantom "person" whose
            # position jumps between real people. Discard recall to
            # protect identity consistency.
            if track_id < 0:
                continue

            # Leg keypoints in pixel space -- LIDAR fusion node matches
            # these against LIDAR leg clusters by bearing; no depth or
            # camera->map transform needed here anymore.
            leg_pts = []
            if result.keypoints is not None:
                kpts_xy = result.keypoints.xy[box_i].cpu().numpy()
                kpts_conf = result.keypoints.conf[box_i].cpu().numpy()
                for idx in self.leg_kpt_idx:
                    if kpts_conf[idx] >= self.leg_kpt_min_conf:
                        px, py = kpts_xy[idx]
                        leg_pts.append((float(px), float(py)))


            # Fields: track_id, conf, x1, y1, x2, y2, leg keypoints
            # as "px:py" pairs separated by ';' (variable count, 0-4).
            # No bearing here anymore - identity_fusion_node_lidar.py
            # computes it from bbox center, matching the real-robot
            # architecture (see that file for the fixed-HFOV method).
            out = String()
            out.data = (
                f"{track_id},"
                f"{conf:.2f},"
                f"{x1},"
                f"{y1},"
                f"{x2},"
                f"{y2},"
                + ";".join(f"{px:.1f}:{py:.1f}" for px, py in leg_pts)
            )
            self.pub.publish(out)

            self.get_logger().info(
                f"id:{track_id} conf:{conf:.2f} "
                f"bbox=({x1},{y1},{x2},{y2}) h={y2-y1}px "
                f"leg_kpts={len(leg_pts)}"
            )

            if self.show_debug_image:
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                for px, py in leg_pts:
                    cv2.circle(display_frame, (int(px), int(py)), 4, (255, 0, 0), -1)
                cv2.putText(
                    display_frame,
                    f"ID:{track_id} conf:{conf:.2f}",
                    (x1, max(20, y1 - 10)),
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
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()