#!/usr/bin/env python3
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

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

from ultralytics import YOLO


class YoloByteTrackPositionNode(Node):

    @staticmethod
    def _iou(box_a, box_b):
        xa1, ya1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
        xa2, ya2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
        inter = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def __init__(self):
        super().__init__("yolo_bytetrack_position_node")

        self.bridge = CvBridge()

        # Simulation uses raw Image on /oakd/rgb/preview/image_raw
        # (real robot uses CompressedImage on /turtlebot4/oakd/...)
        self.rgb_topic = "/oakd/rgb/preview/image_raw"

        export_path = "/root/thesis_social_navigation_ws/src/social_perception/social_perception/Lidar/yolov8n-pose_openvino_model/"
        if not os.path.isdir(export_path):
            self.get_logger().info("OpenVINO export not found, exporting once...")
            YOLO("yolov8n-pose.pt").export(format="openvino", imgsz=320)
        self.model = YOLO(export_path)

        # COCO keypoint indices: 13=L knee,14=R knee,15=L ankle,16=R ankle
        self.leg_kpt_idx = [13, 14, 15, 16]
        self.leg_kpt_min_conf = 0.3
        # In simulation the camera sees the full person body so 0.2 is fine.
        # Real robot: 0.45 (camera pitched down, only legs in frame).
        self.min_publish_conf = 0.2

        self.frame_count = 0
        self.process_every_n_frames = 2
        self.show_debug_image = True

        self.pub = self.create_publisher(String, "/person_positions_map", 10)

        # Bearing is computed by identity_fusion_node_lidar.py from the
        # bbox centre — no camera_info subscription needed here.

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

        self.get_logger().info("YOLO + ByteTrack leg-keypoint node started (bearing computed by fusion node)")
        self.get_logger().info(f"RGB topic: {self.rgb_topic}")
        self.get_logger().info(f"Publishing: /person_positions_map")

    def rgb_callback(self, msg):
        # DROP SKIPPED FRAMES IMMEDIATELY (0 CPU COST)
        self.frame_count += 1
        if self.frame_count % self.process_every_n_frames != 0:
            return

        now_s = time.monotonic()
        if not hasattr(self, "_last_heartbeat"):
            self._last_heartbeat = 0.0
        if now_s - self._last_heartbeat > 5.0:
            self.get_logger().info(f"(alive) frame #{self.frame_count} processed")
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

        # IoU dedup: drop lower-confidence boxes that overlap a
        # higher-confidence one above threshold (same-person split boxes).
        DEDUP_IOU_THRESHOLD = 0.45
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        order = sorted(range(len(boxes_xyxy)), key=lambda i: -confs[i])
        keep = []
        for i in order:
            if any(self._iou(boxes_xyxy[i], boxes_xyxy[j]) > DEDUP_IOU_THRESHOLD
                   for j in keep):
                continue
            keep.append(i)
        keep_set = set(keep)

        for box_i, box in enumerate(result.boxes):
            if box_i not in keep_set:
                continue

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
            # Bearing computed by identity_fusion_node_lidar.py from bbox centre.
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