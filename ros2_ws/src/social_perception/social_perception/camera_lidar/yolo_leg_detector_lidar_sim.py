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
        inter_w, inter_h = max(0.0, xa2 - xa1), max(0.0, ya2 - ya1)
        inter = inter_w * inter_h
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0
        
    @staticmethod
    def _contained(inner, outer):
        ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
        iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
        area = (inner[2] - inner[0]) * (inner[3] - inner[1])
        return ix * iy / area if area > 0 else 0.0

    def __init__(self):

        
        super().__init__("yolo_bytetrack_position_node")

        self.bridge = CvBridge()

        self.rgb_topic = "/oakd/rgb/preview/image_raw"  # SIM: raw Image, no namespace

        export_path = "/root/thesis_social_navigation_ws/src/social_perception/social_perception/camera_lidar/yolov8n-pose_openvino_model/"
        if not os.path.isdir(export_path):
            self.get_logger().info("OpenVINO export not found, exporting once...")
            YOLO("yolov8n-pose.pt").export(format="openvino", imgsz=320)
        self.model = YOLO(export_path)

        # COCO keypoint indices: 13=L knee,14=R knee,15=L ankle,16=R ankle
        self.leg_kpt_idx = [13, 14, 15, 16]
        self.leg_kpt_min_conf = 0.3
        # THESIS TUNE 0.75 -> 0.45. Measured on hallway_7m_06: the
        # camera published detections for 5.5 s of a 50.7 s run and
        # fused output covered 3.8 s as a direct result, while lidar
        # tracked the person throughout. ~757 inferences produced 66
        # published detections (~9% pass rate). The mount is pitched
        # at the floor, so a person is cropped to knees-and-boots and
        # scores far below 0.75 - the gate was tuned for whole-body
        # boxes this camera does not produce. Uncorroborated camera
        # detections are already handled downstream by
        # CAMERA_ONLY_GRACE_PERIOD and the bearing blacklist in
        # identity_fusion_node_lidar.py, both verified working.
        # Root fix remains mounting geometry: tilt the camera up.
        self.min_publish_conf = 0.45

        self.frame_count = 0
        self.process_every_n_frames = 2
        self.show_debug_image = True

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

        self.get_logger().info("YOLO + ByteTrack leg-keypoint node started (no depth/TF - LIDAR fusion handles range)")
        self.get_logger().info(f"RGB topic: {self.rgb_topic}")
        self.get_logger().info(f"Publishing: /person_positions_map")

    def rgb_callback(self, msg):
        # DROP SKIPPED FRAMES IMMEDIATELY (0 CPU COST)
        self.frame_count += 1
        if self.frame_count % self.process_every_n_frames != 0:
            return

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

        # THESIS FIX (duplicate/split detections): occasionally YOLO
        # produces two overlapping boxes for the same person's legs in
        # one frame, each getting its own ByteTrack id - this looks
        # like two people to everything downstream. Drop lower-confidence
        # boxes that overlap a higher-confidence one above threshold.
        DEDUP_IOU_THRESHOLD = 0.45
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        order = sorted(range(len(boxes_xyxy)), key=lambda i: -confs[i])
        keep = []
        for i in order:
            # THESIS FIX (16 Sep, proto_K3): a box partly hiding the legs split one
            # person into id:22 (98,0,147,76) fully inside id:20 (92,0,157,153).
            # IoU was 0.37 < 0.45, so it survived and the same-frame rule created
            # a new stable id. Also drop a box that is >= 85% inside a kept one.
            if any(self._iou(boxes_xyxy[i], boxes_xyxy[j]) > DEDUP_IOU_THRESHOLD
                   or self._contained(boxes_xyxy[i], boxes_xyxy[j]) >= 0.85 for j in keep):
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
            # THESIS INSTRUMENTATION (reject logging). Detections between
            # the model's conf=0.15 and min_publish_conf are produced and
            # tracked but never published. Log them before dropping so the
            # 0.15-0.45 population can be counted offline from a bag
            # replay. Purely diagnostic - no behaviour change.
            if conf < self.min_publish_conf:
                _k = 0
                if result.keypoints is not None:
                    _kc = result.keypoints.conf[box_i].cpu().numpy()
                    _k = sum(1 for _i in self.leg_kpt_idx
                             if _kc[_i] >= self.leg_kpt_min_conf)
                self.get_logger().info(
                    f"REJECT conf id:{track_id} conf:{conf:.2f} "
                    f"bbox=({x1},{y1},{x2},{y2}) h={y2-y1}px leg_kpts={_k}")
                continue

            # THESIS FIX (unassigned-track sentinel leak) - perception v2
            # -1 means ByteTrack did not associate this detection with any
            # existing track. It is a sentinel, not an identity. Publishing
            # it makes every unassigned detection collide in the same
            # downstream dict slot, producing one phantom "person" whose
            # position jumps between real people. Discard recall to
            # protect identity consistency.
            if track_id < 0:
                self.get_logger().info(
                    f"REJECT track id:-1 conf:{conf:.2f} "
                    f"bbox=({x1},{y1},{x2},{y2}) h={y2-y1}px")
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

            # Fields: track_id, conf, x1, y1, x2, y2, then leg keypoints
            # as "px:py" pairs separated by ';' (variable count, 0-4).
            # THESIS FIX (16 Sep, ray timing): field [7] = image header
            # stamp (s). camera_ray_person_node pairs the detection with
            # the scan closest to THIS time instead of the newest scan;
            # otherwise a moving robot casts the ray from the wrong pose.
            # Appended last so older consumers (parts[0..6]) are unaffected.
            img_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            out = String()
            out.data = (
                f"{track_id},"
                f"{conf:.2f},"
                f"{x1},"
                f"{y1},"
                f"{x2},"
                f"{y2},"
                + ";".join(f"{px:.1f}:{py:.1f}" for px, py in leg_pts)
                + f",{img_stamp:.3f}"
            )
            self.pub.publish(out)

            self.get_logger().info(
                f"id:{track_id} conf:{conf:.2f} "
                f"bbox=({x1},{y1},{x2},{y2}) h={y2-y1}px "
                f"leg_kpts={len(leg_pts)} "
                f"age={(self.get_clock().now().nanoseconds * 1e-9 - img_stamp) * 1000.0:.0f}ms"
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