#!/usr/bin/env python3
"""
social_group_detector_node.py

Combines combined_facing.py's gaze-first + VLM-fallback facing
classifier with group_formation_detector.py's live tracking structure,
plus Option 1 (corridor-aware zone buffer shrinking) using the local
costmap.

SCOPE - what this does and does not do, read before deploying:
  - Dyad detection only (facing / side_by_side / diverging). Triad
    (3-person huddle) support from combined_facing.py is NOT ported
    here yet - would need group candidate formation from streaming
    positions, not just a single image crop, which is real additional
    work.
  - Queue detection from the original group_formation_detector.py is
    NOT included. This file is scoped to conversation/companion
    dyads + Option 1 only, to keep it reviewable.
  - UNTESTED against real hardware or a real costmap. The facing
    classifier logic (gaze + VLM, dedup, confidence gating, flipped
    ear convention, side_by_side/diverging labels) is the same code
    validated against 6 real photos in combined_facing.py - that part
    is trustworthy. The costmap clearance probing and buffer-shrink
    logic (Option 1) is new and has not been run against a real
    OccupancyGrid at all. Test in open space first, narrow corridor
    second.
  - Costmap topic/frame defaults below are GUESSES based on your
    /robot1 namespace convention from earlier bring-up - verify
    against your actual controller_server / costmap config before
    trusting the clearance numbers.

INPUT
  /predicted_person_positions (String) - same format as the existing
      pipeline: track_id,conf,x,y,vx,vy,pred_x,pred_y,horizon,
      rotation_gate,bbox("x1;y1;x2;y2" or "none")
  RGB camera topic - for cropping the facing classifier's input
  local costmap (OccupancyGrid) - for Option 1 clearance probing

OUTPUT
  /social_groups (String), same wire format as group_formation_detector.py:
      group_id,group_type,cx,cy,axis_x,axis_y,half_length,half_width,
      member_ids,member_xy
  group_type is now "conversation" (facing), "side_by_side", or absent
  (diverging / unknown pairs are not published, same as before).
"""

import math
import time

import numpy as np
import cv2
import open_clip
import torch
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.qos import DurabilityPolicy
from std_msgs.msg import String
from sensor_msgs.msg import Image as RosImage   # SIM: gz publishes raw Image, not CompressedImage
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge
from ultralytics import YOLO


# =======================================================================
# Tunables - geometry/timing gates reused verbatim from
# group_formation_detector.py (already tuned/validated there).
# =======================================================================
CONV_MAX_DIST = 1.8              # m; max separation to be a candidate pair
CONV_MIN_DIST = 0.3              # m; below this, same-person noise/overlap
CONV_MAX_SPEED = 0.30            # m/s; "near-stationary" threshold
CONV_MIN_DURATION = 0.75        # s; sustained closeness before flagging
FRESH_POSITION_TIMEOUT = 1.0     # s; position trusted for geometry
IDENTITY_RETENTION_TIMEOUT = 300.0  # s; state kept for re-identification
CONV_MAX_CONTINUITY_GAP = 1.0    # s; gap still counted as continuous
BBOX_MAX_AGE = 5.0               # s; loosened from the original 0.5s -
                                  # that assumed a dense ~25Hz bbox stream,
                                  # but bbox here only refreshes on
                                  # camera_confirmed cycles (intermittent,
                                  # observed 20-90s+ gaps on hardware).
                                  # Safe to loosen since pairs are already
                                  # gated to near-stationary (CONV_MAX_SPEED)

# Facing classifier tunables, from combined_facing.py (validated against
# 6 real photos there - see that file's history for how each was found).
KPT_MIN_CONF = 0.30
KPT_PRESENCE_CONF = 0.20
VLM_MIN_CONFIDENCE = 0.55
DEDUP_IOU_THRESHOLD = 0.45
PROMPTS = ["people facing each other", "people not facing each other"]

# Zone geometry
ZONE_BUFFER = 0.4                # m; nominal buffer (group_formation_detector.py)
MIN_ZONE_BUFFER = 0.05           # m; floor - never shrink below this
SIDE_BY_SIDE_FORWARD_EXTENT = 1.2  # m; UNVALIDATED placeholder, see prior
                                    # discussion - not from proxemics
                                    # measurement, just a starting guess

# Option 1: corridor-aware buffer shrinking
CLEARANCE_PROBE_MAX = 2.0        # m; don't probe further than this
COSTMAP_OCCUPIED_THRESHOLD = 99  # occupancy value (0-100) considered 
                                  # -1 (unknown) is also treated as blocked -
                                  # conservative, since unmapped space near a
                                  # group shouldn't be assumed free


GROUP_BREAKUP_DIST_HOLD = 1.0  # s; pair must exceed CONV_MAX_DIST for
                                  # this long (not just one noisy sample)
                                  # before the cached group is cleared


class TrackState:
    def __init__(self):
        self.x = None
        self.y = None
        self.vx = 0.0
        self.vy = 0.0
        self.last_update = 0.0
        self.bbox = None
        self.bbox_time = None
        self.close_since = {}  # other_track_id -> [accumulated_s, last_seen]


# =======================================================================
# Facing classifier - ported from combined_facing.py, unchanged logic.
# =======================================================================
class FacingClassifier:
    def __init__(self, pose_model_path="yolov8n-pose.pt"):
        self.clip_model = None  # lazy-loaded on first VLM fallback use
        self.clip_preprocess = None
        self.clip_text_features = None
        self.pose_model = YOLO(pose_model_path)
        self._no_signal = False

    def _ensure_vlm_loaded(self):
        if self.clip_model is not None:
            return
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            "MobileCLIP-S1", pretrained="datacompdr"
        )
        self.clip_model.eval()
        tokenizer = open_clip.get_tokenizer("MobileCLIP-S1")
        text_tokens = tokenizer(PROMPTS)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            self.clip_text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    def classify_vlm(self, crop_bgr):
        self._ensure_vlm_loaded()
        crop_rgb = crop_bgr[:, :, ::-1]
        pil_image = PILImage.fromarray(crop_rgb)
        image_input = self.clip_preprocess(pil_image).unsqueeze(0)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarities = (100.0 * image_features @ self.clip_text_features.T).softmax(dim=-1)
        best_idx = int(similarities.argmax())
        best_score = float(similarities[0, best_idx])
        if best_score < VLM_MIN_CONFIDENCE:
            return None
        return best_idx == 0

    @staticmethod
    def _extract_head_heading(kpts):
        nose = kpts[0]
        l_ear, r_ear = kpts[3], kpts[4]
        if l_ear[0] > 0 and r_ear[0] == 0 and nose[0] > 0:
            return np.array([-1.0, 0.0])
        if r_ear[0] > 0 and l_ear[0] == 0 and nose[0] > 0:
            return np.array([1.0, 0.0])
        if l_ear[0] > 0 and r_ear[0] > 0 and nose[0] > 0:
            ear_mid_x = (l_ear[0] + r_ear[0]) / 2.0
            yaw_px = nose[0] - ear_mid_x
            if yaw_px > 3.0:
                return np.array([1.0, 0.0])
            elif yaw_px < -3.0:
                return np.array([-1.0, 0.0])
            return np.array([0.0, 0.0])
        return np.array([0.0, 0.0])

    @staticmethod
    def _extract_hip_heading(kpts):
        l_hip, r_hip = kpts[11], kpts[12]
        if l_hip[0] > 0 and r_hip[0] == 0:
            return np.array([1.0, 0.0])
        if r_hip[0] > 0 and l_hip[0] == 0:
            return np.array([-1.0, 0.0])
        return np.array([0.0, 0.0])

    @staticmethod
    def _has_orientation_data(conf_row):
        if conf_row is None:
            return False
        idxs = [0, 3, 4]  # nose, l_ear, r_ear only - hip excluded, see
                          # combined_facing.py history for why
        return bool(np.any(conf_row[idxs] >= KPT_PRESENCE_CONF))

    def _person_heading(self, kpts, conf_row):
        kpts_gated = kpts.copy()
        if conf_row is not None:
            kpts_gated[conf_row < KPT_MIN_CONF] = 0.0
        head_vec = self._extract_head_heading(kpts_gated)
        if head_vec[0] == 0.0:
            hip_vec = self._extract_hip_heading(kpts_gated)
            if hip_vec[0] != 0.0:
                head_vec = hip_vec
        return head_vec

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

    def _deduplicate(self, kpts_list, boxes_list, conf_list, box_confs):
        order = sorted(range(len(boxes_list)), key=lambda i: -box_confs[i])
        keep = []
        for i in order:
            if any(self._iou(boxes_list[i], boxes_list[j]) > DEDUP_IOU_THRESHOLD for j in keep):
                continue
            keep.append(i)
        keep.sort()
        return (
            [kpts_list[i] for i in keep],
            [boxes_list[i] for i in keep],
            [conf_list[i] for i in keep] if conf_list is not None else None,
        )

    def classify_gaze_dyad(self, crop_bgr, debug_log=None):
        """Returns "facing" / "side_by_side" / "diverging" / None."""
        self._no_signal = False
        results = self.pose_model(crop_bgr, verbose=False)[0]
        kpts_list = results.keypoints.xy.cpu().numpy()
        boxes_list = results.boxes.xyxy.cpu().numpy()
        box_confs = results.boxes.conf.cpu().numpy()
        conf_list = results.keypoints.conf.cpu().numpy() if results.keypoints.conf is not None else None

        kpts_list, boxes_list, conf_list = self._deduplicate(kpts_list, boxes_list, conf_list, box_confs)
        if len(kpts_list) != 2:
            self._no_signal = True
            return None

        people = []
        for idx, (kpts, box) in enumerate(zip(kpts_list, boxes_list)):
            cx = (box[0] + box[2]) / 2.0
            conf_row = conf_list[idx] if conf_list is not None else None
            head_vec = self._person_heading(kpts, conf_row)
            has_data = self._has_orientation_data(conf_row)
            people.append({"cx": cx, "head_vec": head_vec, "has_data": has_data})

        left_p, right_p = sorted(people, key=lambda p: p["cx"])
        lx, rx = left_p["head_vec"][0], right_p["head_vec"][0]

        if debug_log is not None:
            debug_log(
                f"left_cx={left_p['cx']:.1f} heading={lx:.0f} "
                f"right_cx={right_p['cx']:.1f} heading={rx:.0f}"
            )

        if lx > 0.5 and rx < -0.5:
            return "facing"
        if lx == 0.0 and rx == 0.0:
            if left_p["has_data"] and right_p["has_data"]:
                return "side_by_side"
            self._no_signal = True
            return None
        if lx < -0.5 and rx > 0.5:
            return "diverging"
        return "side_by_side"

    def classify(self, crop_bgr, debug_log=None):
        gaze_result = self.classify_gaze_dyad(crop_bgr, debug_log=debug_log)
        if gaze_result is not None:
            return gaze_result
        if self._no_signal:
            return None  # VLM would guess on the same missing info - skip
        vlm_result = self.classify_vlm(crop_bgr)
        if vlm_result is True:
            return "facing"
        if vlm_result is False:
            return "side_by_side"
        return None

    @staticmethod
    def is_group(label):
        return label in ("facing", "side_by_side")


# =======================================================================
# Node
# =======================================================================
class SocialGroupDetector(Node):
    def __init__(self):
        super().__init__("social_group_detector")

        self.declare_parameter("input_topic", "/predicted_person_positions")
        self.declare_parameter("bbox_topic", "/person_positions_fused")
        self.declare_parameter("output_topic", "/social_groups")
        self.declare_parameter("rgb_topic", "/oakd/rgb/preview/image_raw")
        self.declare_parameter("costmap_topic", "/local_costmap/costmap")
        self.declare_parameter("show_debug_image", True)

        self.input_topic = self.get_parameter("input_topic").value
        self.bbox_topic = self.get_parameter("bbox_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.costmap_topic = self.get_parameter("costmap_topic").value
        self.show_debug_image = self.get_parameter("show_debug_image").value

        self.tracks = {}
        self.bridge = CvBridge()
        self.latest_frame = None
        self.latest_costmap = None  # (grid np.array, resolution, origin_x, origin_y)
        self.confirmed_pairs = {}   # frozenset({id_a,id_b}) -> (label, last_confirmed_time)
        self.pairs_out_of_range_since = {}  # pair_key -> timestamp first seen out of range

        self.get_logger().info("Loading YOLOv8-pose...")
        self.classifier = FacingClassifier("yolov8n-pose.pt")
        self.get_logger().info("Ready (MobileCLIP lazy-loads on first fallback use).")

        self.create_subscription(String, self.input_topic, self.position_callback, 10)
        self.create_subscription(String, self.bbox_topic, self.bbox_callback, 10)
        # Same fix as yolo_leg_detector_lidar.py: default queue depth let
        # frames back up (measured 325ms staleness there). depth=1 +
        # BEST_EFFORT drops late frames instead of queueing them.
        sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(RosImage, self.rgb_topic, self.image_callback, sensor_qos)
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(OccupancyGrid, self.costmap_topic, self.costmap_callback, 10)
        self.pub = self.create_publisher(String, self.output_topic, 10)

        self.create_timer(0.5, self.detect_groups)
        self.create_timer(2.0, self._log_diagnostics)

    def _log_diagnostics(self):
        now = self.now()
        n_tracks = len(self.tracks)
        n_fresh_bbox = sum(
            1 for t in self.tracks.values()
            if t.bbox is not None and t.bbox_time is not None
            and now - t.bbox_time <= FRESH_POSITION_TIMEOUT
        )
        has_frame = self.latest_frame is not None
        has_costmap = self.latest_costmap is not None
        self.get_logger().info(
            f"[diag] tracks={n_tracks} fresh_bbox={n_fresh_bbox} "
            f"latest_frame={'yes' if has_frame else 'NO'} "
            f"latest_costmap={'yes' if has_costmap else 'NO'}"
        )
        # THESIS FIX: drop confirmed_pairs entries referencing a pruned
        # track (person gone entirely) OR a track that's gone stale
        # (no update within FRESH_POSITION_TIMEOUT) - previously this
        # only ran at IDENTITY_RETENTION_TIMEOUT (300s), leaving stale
        # "confirmed" pairs sitting in memory for up to 5 minutes after
        # either member actually left camera/lidar range.
        stale_or_gone = []
        for pair_key in self.confirmed_pairs:
            ids = list(pair_key)
            if len(ids) != 2:
                stale_or_gone.append(pair_key)
                continue
            for tid in ids:
                t = self.tracks.get(tid)
                if t is None or now - t.last_update > FRESH_POSITION_TIMEOUT:
                    stale_or_gone.append(pair_key)
                    break
        for pair_key in stale_or_gone:
            del self.confirmed_pairs[pair_key]
            self.pairs_out_of_range_since.pop(pair_key, None)

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # -------------------------------------------------------------
    def image_callback(self, msg):
        self.latest_frame = msg
        if self.show_debug_image:
            self._draw_debug(msg)

    # Colors match a facing/side_by_side/diverging legend, BGR for cv2.
    _LABEL_COLOR = {
        "facing": (0, 255, 0),        # green
        "side_by_side": (255, 200, 0),  # cyan-ish
        "diverging": (0, 0, 255),     # red
    }

    def _draw_debug(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"Debug decode failed: {e}")
            return

        now = self.now()
        # Draw every track with a fresh bbox
        for tid, t in self.tracks.items():
            if t.bbox is None or t.bbox_time is None or now - t.bbox_time > FRESH_POSITION_TIMEOUT:
                continue
            x1, y1, x2, y2 = (int(v) for v in t.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 200), 1)
            cv2.putText(frame, f"id:{tid}", (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 200), 1)

        # Draw a connecting line + label for every currently-confirmed pair
        for pair_key, (label, _) in self.confirmed_pairs.items():
            ids = list(pair_key)
            if len(ids) != 2 or ids[0] not in self.tracks or ids[1] not in self.tracks:
                continue
            ta, tb = self.tracks[ids[0]], self.tracks[ids[1]]
            if ta.bbox is None or tb.bbox is None:
                continue
            ca = (int((ta.bbox[0] + ta.bbox[2]) / 2), int((ta.bbox[1] + ta.bbox[3]) / 2))
            cb = (int((tb.bbox[0] + tb.bbox[2]) / 2), int((tb.bbox[1] + tb.bbox[3]) / 2))
            color = self._LABEL_COLOR.get(label, (200, 200, 200))
            cv2.line(frame, ca, cb, color, 2)
            mid = ((ca[0] + cb[0]) // 2, (ca[1] + cb[1]) // 2)
            cv2.putText(frame, label, mid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Social Group Detector", frame)
        cv2.waitKey(1)

    def costmap_callback(self, msg):
        w, h = msg.info.width, msg.info.height
        grid = np.array(msg.data, dtype=np.int16).reshape(h, w)
        self.latest_costmap = (
            grid, msg.info.resolution, msg.info.origin.position.x, msg.info.origin.position.y
        )

    def bbox_callback(self, msg):
        """/person_positions_fused, from identity_fusion_node_lidar.py:
        [stable_id, conf, lidar_x, lidar_y, "0.0","0","0", x1,y1,x2,y2,
        source] - source is "camera_confirmed" (real bbox) or
        "lidar_only" (bbox fields zeroed - skip). Same stable_id space
        as /predicted_person_positions, unlike the raw camera topic
        which uses pre-fusion ByteTrack ids."""
        fields = msg.data.split(",")
        if len(fields) < 12:
            return
        if fields[11] != "camera_confirmed":
            return
        try:
            tid = int(float(fields[0]))
            x1, y1, x2, y2 = (float(fields[i]) for i in (7, 8, 9, 10))
        except ValueError:
            return

        t = self.tracks.get(tid)
        if t is None:
            t = TrackState()
            self.tracks[tid] = t
        t.bbox = (x1, y1, x2, y2)
        t.bbox_time = self.now()

    def position_callback(self, msg):
        parts = msg.data.split(",")
        # human_kf_predictor_lidar.py always publishes exactly 10 fields
        # (no bbox) - bbox comes separately from bbox_callback, keyed by
        # the same stable id, since /person_positions_fused (identity
        # fusion's output) is where stable_id and bbox last coexist.
        if len(parts) < 10:
            return
        try:
            tid = int(float(parts[0]))
            x, y = float(parts[2]), float(parts[3])
            vx, vy = float(parts[4]), float(parts[5])
        except ValueError:
            return

        now = self.now()
        t = self.tracks.get(tid)
        if t is None:
            t = TrackState()
            self.tracks[tid] = t
        t.x, t.y, t.vx, t.vy = x, y, vx, vy
        t.last_update = now

    # -------------------------------------------------------------
    # Option 1: costmap clearance probing
    # -------------------------------------------------------------
    def _costmap_value_at(self, wx, wy):
        if self.latest_costmap is None:
            return None
        grid, res, ox, oy = self.latest_costmap
        gx = int((wx - ox) / res)
        gy = int((wy - oy) / res)
        h, w = grid.shape
        if gx < 0 or gx >= w or gy < 0 or gy >= h:
            return None
        return int(grid[gy, gx])

    def _probe_clearance(self, cx, cy, perp_x, perp_y, sign):
        """March outward from (cx,cy) along (perp_x,perp_y)*sign until an
        occupied/unknown cell or CLEARANCE_PROBE_MAX. Returns distance (m)."""
        if self.latest_costmap is None:
            return CLEARANCE_PROBE_MAX  # no costmap yet - assume open, but
                                          # this is a real gap, see caveat below
        step = max(self.latest_costmap[1], 0.05)
        dist = 0.0
        while dist < CLEARANCE_PROBE_MAX:
            wx = cx + perp_x * sign * dist
            wy = cy + perp_y * sign * dist
            val = self._costmap_value_at(wx, wy)
            if val is None or val >= COSTMAP_OCCUPIED_THRESHOLD or val < 0:
                return dist
            dist += step
        return CLEARANCE_PROBE_MAX

    def _effective_buffer(self, cx, cy, axis_x, axis_y):
        """Option 1: shrink ZONE_BUFFER to fit available corridor space.
        Probes perpendicular to the given axis (the direction the zone's
        half_width extends into) in both directions, sums to get total
        available width, and caps the buffer so 2*(dist/2+buffer) <=
        available width. Floors at MIN_ZONE_BUFFER rather than going to
        zero - see prior discussion on when this should instead trigger
        a yield/wait behavior (Option 2, NOT implemented in this file)."""
        perp_x, perp_y = -axis_y, axis_x
        clearance = (
            self._probe_clearance(cx, cy, perp_x, perp_y, 1.0)
            + self._probe_clearance(cx, cy, perp_x, perp_y, -1.0)
        )
        buffer = min(ZONE_BUFFER, max(MIN_ZONE_BUFFER, clearance / 2.0))
        if buffer < ZONE_BUFFER - 1e-3:
            self.get_logger().info(
                f"Zone buffer shrunk {ZONE_BUFFER:.2f}m -> {buffer:.2f}m "
                f"(measured clearance {clearance:.2f}m)"
            )
        return buffer

    # -------------------------------------------------------------
    # Zone builders
    # -------------------------------------------------------------
    def _build_facing_zone(self, ta, tb, id_a, id_b):
        cx, cy = (ta.x + tb.x) / 2.0, (ta.y + tb.y) / 2.0
        dx, dy = tb.x - ta.x, tb.y - ta.y
        dist = math.hypot(dx, dy)
        axis_x = dx / dist if dist > 1e-6 else 1.0
        axis_y = dy / dist if dist > 1e-6 else 0.0

        buffer = self._effective_buffer(cx, cy, axis_x, axis_y)
        half_length = dist / 2.0 + buffer
        half_width = buffer

        group_id = f"conv_{min(id_a, id_b)}_{max(id_a, id_b)}"
        return (group_id, "conversation", cx, cy, axis_x, axis_y,
                half_length, half_width, [id_a, id_b], [(ta.x, ta.y), (tb.x, tb.y)], buffer)

    def _build_side_by_side_zone(self, ta, tb, id_a, id_b):
        cx, cy = (ta.x + tb.x) / 2.0, (ta.y + tb.y) / 2.0
        dx, dy = tb.x - ta.x, tb.y - ta.y
        dist = math.hypot(dx, dy)
        # Connecting-axis unit vector (across the pair)
        conn_x = dx / dist if dist > 1e-6 else 1.0
        conn_y = dy / dist if dist > 1e-6 else 0.0
        # Facing axis is perpendicular to the connecting line
        axis_x, axis_y = -conn_y, conn_x

        # The hallway-constrained dimension here is ACROSS the pair
        # (conn direction), same reasoning as the facing zone's
        # half_width - so Option 1 probes along conn, not along axis.
        buffer = self._effective_buffer(cx, cy, conn_x, conn_y)
        half_width = dist / 2.0 + buffer
        half_length = SIDE_BY_SIDE_FORWARD_EXTENT  # not costmap-shrunk

        group_id = f"sbs_{min(id_a, id_b)}_{max(id_a, id_b)}"
        return (group_id, "side_by_side", cx, cy, axis_x, axis_y,
                half_length, half_width, [id_a, id_b], [(ta.x, ta.y), (tb.x, tb.y)], buffer)

    # -------------------------------------------------------------
    def _save_debug_crop(self, crop, id_a, id_b, label, now):
        """Save the exact crop the classifier saw for a newly-confirmed
        pair, so a wrong verdict (like Pair(2,5) 'facing' when they were
        actually back-to-back) can be visually checked against what the
        classifier actually had to work with, instead of guessing."""
        try:
            import os
            out_dir = "/root/thesis_social_navigation_ws/debug_crops"
            os.makedirs(out_dir, exist_ok=True)
            fname = f"{out_dir}/{now:.0f}_{id_a}_{id_b}_{label}.jpg"
            cv2.imwrite(fname, crop)
            self.get_logger().info(f"Saved debug crop: {fname}")
        except Exception as e:
            self.get_logger().warn(f"Could not save debug crop: {e}")

    # def _crop_union(self, frame, bbox_a, bbox_b, pad=20):
    #     h, w = frame.shape[:2]
    #     x1 = max(0, int(min(bbox_a[0], bbox_b[0]) - pad))
    #     y1 = max(0, int(min(bbox_a[1], bbox_b[1]) - pad))
    #     x2 = min(w - 1, int(max(bbox_a[2], bbox_b[2]) + pad))
    #     y2 = min(h - 1, int(max(bbox_a[3], bbox_b[3]) + pad))
    #     if x2 <= x1 or y2 <= y1:
    #         return None
    #     return frame[y1:y2, x1:x2]
    def _crop_union(self, frame, bbox_a, bbox_b, pad=20):
        h, w = frame.shape[:2]
        x1 = max(0, int(min(bbox_a[0], bbox_b[0]) - pad))
        # TEMP TEST: extend crop all the way to top of frame, since at
        # close range the leg-bbox sits well below the head, and a fixed
        # pixel pad isn't enough to reach it.
        y1 = 0
        x2 = min(w - 1, int(max(bbox_a[2], bbox_b[2]) + pad))
        y2 = min(h - 1, int(max(bbox_a[3], bbox_b[3]) + pad))
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    def detect_groups(self):
        now = self.now()

        # Prune fully stale tracks (long gone, no re-ID window relevant)
        for tid in [t for t, s in self.tracks.items()
                    if now - s.last_update > IDENTITY_RETENTION_TIMEOUT]:
            del self.tracks[tid]

        fresh_ids = [t for t, s in self.tracks.items()
                     if now - s.last_update <= FRESH_POSITION_TIMEOUT]

        groups = []
        for i, id_a in enumerate(fresh_ids):
            for id_b in fresh_ids[i + 1:]:
                ta, tb = self.tracks[id_a], self.tracks[id_b]
                dist = math.hypot(ta.x - tb.x, ta.y - tb.y)
                pair_key = frozenset((id_a, id_b))
                if not (CONV_MIN_DIST <= dist <= CONV_MAX_DIST):
                    if pair_key in self.confirmed_pairs:
                        first_out = self.pairs_out_of_range_since.get(pair_key)
                        if first_out is None:
                            self.pairs_out_of_range_since[pair_key] = now
                            self.get_logger().info(
                                f"[breakup] ({id_a},{id_b}) dist={dist:.2f}m "
                                f"exceeds range - starting breakup hold"
                            )
                        elif now - first_out > GROUP_BREAKUP_DIST_HOLD:
                            self.confirmed_pairs.pop(pair_key, None)
                            self.pairs_out_of_range_since.pop(pair_key, None)
                            self.get_logger().info(
                                f"[breakup] ({id_a},{id_b}) cleared after "
                                f"{now - first_out:.2f}s out of range"
                            )
                        else:
                            self.get_logger().info(
                                f"[breakup] ({id_a},{id_b}) dist={dist:.2f}m "
                                f"still out of range ({now - first_out:.2f}/"
                                f"{GROUP_BREAKUP_DIST_HOLD}s)"
                            )
                    continue
                else:
                    self.pairs_out_of_range_since.pop(pair_key, None)

                speed_a = math.hypot(ta.vx, ta.vy)
                speed_b = math.hypot(tb.vx, tb.vy)
                if speed_a > CONV_MAX_SPEED or speed_b > CONV_MAX_SPEED:
                    self.get_logger().info(
                        f"[gate] ({id_a},{id_b}) dist={dist:.2f}m PASSED distance, "
                        f"FAILED speed (a={speed_a:.2f} b={speed_b:.2f}, "
                        f"limit={CONV_MAX_SPEED})"
                    )
                    continue

                entry = ta.close_since.get(id_b)
                if entry is None:
                    entry = [0.0, now]
                else:
                    gap = now - entry[1]
                    if gap <= CONV_MAX_CONTINUITY_GAP:
                        entry[0] += gap
                    entry[1] = now
                ta.close_since[id_b] = entry
                tb.close_since[id_a] = list(entry)

                if entry[0] < CONV_MIN_DURATION:
                    self.get_logger().info(
                        f"[gate] ({id_a},{id_b}) dist={dist:.2f}m speed OK "
                        f"(a={speed_a:.2f} b={speed_b:.2f}) - accumulating duration "
                        f"{entry[0]:.2f}/{CONV_MIN_DURATION}s"
                    )
                    continue

                cached = self.confirmed_pairs.get(pair_key)
                if cached is not None:
                    label = cached[0]
                else:
                    if (ta.bbox is None or tb.bbox is None or self.latest_frame is None
                            or ta.bbox_time is None or tb.bbox_time is None
                            or now - ta.bbox_time > BBOX_MAX_AGE
                            or now - tb.bbox_time > BBOX_MAX_AGE):
                        continue
                    frame = self.bridge.imgmsg_to_cv2(self.latest_frame, desired_encoding="bgr8")
                    crop = self._crop_union(frame, ta.bbox, tb.bbox)
                    if crop is None:
                        continue
                    # label = self.classifier.classify(
                    #     crop, debug_log=lambda s: self.get_logger().info(f"[classify] {s}")
                    # )
                    # if label is not None and FacingClassifier.is_group(label):
                    #     self.confirmed_pairs[pair_key] = (label, now)
                    #     self.get_logger().info(f"Pair ({id_a},{id_b}) confirmed: {label}")
                    #     self._save_debug_crop(crop, id_a, id_b, label, now)
                    label = self.classifier.classify(
                        crop, debug_log=lambda s: self.get_logger().info(f"[classify] {s}")
                    )
                    # TEMP DEBUG: save every attempted crop, not just confirmed ones
                    self._save_debug_crop(crop, id_a, id_b, label or "none", now)
                    if label is not None and FacingClassifier.is_group(label):
                        self.confirmed_pairs[pair_key] = (label, now)
                        self.get_logger().info(f"Pair ({id_a},{id_b}) confirmed: {label}")

                if label == "facing":
                    groups.append(self._build_facing_zone(ta, tb, id_a, id_b))
                elif label == "side_by_side":
                    groups.append(self._build_side_by_side_zone(ta, tb, id_a, id_b))
                # "diverging" / None -> not a group, publish nothing

        self._publish_groups(groups)

    def _publish_groups(self, groups):
        for (group_id, group_type, cx, cy, axis_x, axis_y,
            half_length, half_width, member_ids, member_xy, buffer) in groups:
            xy_str = "|".join(f"{x:.3f};{y:.3f}" for x, y in member_xy)
            out = String()
            out.data = (
                f"{group_id},{group_type},{cx:.3f},{cy:.3f},"
                f"{axis_x:.3f},{axis_y:.3f},{half_length:.3f},{half_width:.3f},"
                f"{';'.join(str(i) for i in member_ids)},{xy_str},{buffer:.3f}"
            )
            self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = SocialGroupDetector()
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