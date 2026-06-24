#!/usr/bin/env python3

import math
import cv2
import numpy as np

import torch
torch.set_num_threads(2)  # or 1 — start low and measure

import rclpy
from rclpy.node import Node

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

        self.model = YOLO("/root/thesis_social_navigation_ws/yolov8s.pt")

        self.frame_count = 0
        self.process_every_n_frames = 2
        self.show_debug_image = True

        self.latest_depth = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # Depth filtering
        self.min_depth = 0.2
        self.max_depth = 5.0
        self.min_depth_pixels = 20

        # Jump rejection in map frame
        self.last_positions = {}   # track_id -> (x, y, timestamp)
        self.max_jump = 0.8
        self.jump_timeout = 2.0    # seconds; stale entry skips jump check

        self.pub = self.create_publisher(String, "/person_positions_map", 10)

        self.create_subscription(Image, self.rgb_topic, self.rgb_callback, 10)
        self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)
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

        # Use lower-middle body region instead of bbox center.
        # This reduces background/wall/floor depth contamination.
        u1 = int(x1 + 0.35 * (x2 - x1))
        u2 = int(x1 + 0.65 * (x2 - x1))
        v1 = int(y1 + 0.60 * (y2 - y1))
        v2 = int(y1 + 0.90 * (y2 - y1))

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

    def transform_camera_to_target(self, x, y, z):
        point_cam = PointStamped()

        # Use latest available TF.
        # This is usually more stable in simulation than using image stamp
        # when camera/depth/TF timestamps are slightly mismatched.
        point_cam.header.stamp = rclpy.time.Time().to_msg()
        point_cam.header.frame_id = self.camera_frame

        point_cam.point.x = float(x)
        point_cam.point.y = float(y)
        point_cam.point.z = float(z)

        try:
            point_target = self.tf_buffer.transform(
                point_cam,
                self.target_frame,
                timeout=rclpy.duration.Duration(seconds=0.2)
            )

            return (
                point_target.point.x,
                point_target.point.y,
                point_target.point.z
            )

        except Exception as e:
            self.get_logger().warn(f"TF transform failed: {e}")
            return None

    def rgb_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"RGB conversion failed: {e}")
            return

        display_frame = frame.copy()

        self.frame_count += 1
        if self.frame_count % self.process_every_n_frames != 0:
            return

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

        results = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=0.65,
            imgsz=320,
            verbose=False
        )

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

            map_point = self.transform_camera_to_target(cam_x, cam_y, cam_z)

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
            now_sec = self.get_clock().now().nanoseconds * 1e-9
            if track_id in self.last_positions:
                last_x, last_y, last_t = self.last_positions[track_id]
                if now_sec - last_t <= self.jump_timeout:
                    jump = math.hypot(map_x - last_x, map_y - last_y)

                    if jump > self.max_jump:
                        self.get_logger().warn(
                            f"Reject jump id:{track_id}, "
                            f"jump={jump:.2f} m, "
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

                        continue

            self.last_positions[track_id] = (map_x, map_y, now_sec)

            out = String()
            out.data = (
                f"{track_id},"
                f"{conf:.2f},"
                f"{map_x:.3f},"
                f"{map_y:.3f},"
                f"{depth:.3f},"
                f"{u},"
                f"{v}"
            )
            self.pub.publish(out)

            self.get_logger().info(
                f"id:{track_id} conf:{conf:.2f} "
                f"pixel=({u},{v}) "
                f"depth={depth:.2f} m "
                f"camera_xyz=({cam_x:.2f},{cam_y:.2f},{cam_z:.2f}) m "
                f"map_xy=({map_x:.2f},{map_y:.2f}) m"
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


def main(args=None):
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