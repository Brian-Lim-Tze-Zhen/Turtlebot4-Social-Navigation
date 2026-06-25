#!/usr/bin/env python3
"""
group_formation_detector.py

THESIS ADDITION — social formation detection (conversation pairs, queues)

-----------------------------------------------------------------------
WHY THIS EXISTS
-----------------------------------------------------------------------
The existing pipeline tracks each person independently:

    yolo_detector.py        -> /person_positions_map      (per-person, pixel+map)
    human_kf_predictor.py   -> /predicted_person_positions (per-person, map+vel+bbox)

Nothing currently looks at RELATIONSHIPS between tracked people. This
node buffers the latest state of every active track and, on each cycle,
checks for two social formations:

  - CONVERSATION (dyad): two people close together and both
    near-stationary for a sustained duration.
  - QUEUE (3+): three or more people roughly collinear, similarly
    spaced, in a corridor-like arrangement.

-----------------------------------------------------------------------
DESIGN DECISION: geometry first, VLM only for genuine ambiguity
-----------------------------------------------------------------------
Queue detection is done ENTIRELY geometrically (collinearity + spacing
regularity computed exactly from x,y - no model uncertainty needed, and
a VLM is worse at fine spatial-relational judgments like this anyway).

Conversation detection is ALSO geometric for the actual flagging
(distance + stationary + duration). The one place geometry has a real
gap: confirming two stationary people are actually FACING each other
(true conversation) vs. e.g. standing back-to-back or side-by-side for
an unrelated reason. Velocity-based heading is meaningless at near-zero
speed, so this is a genuine blind spot for geometry alone.

MobileCLIP is invoked ONLY for that narrow, ambiguous case: a candidate
pair that passes distance+duration but can't be confirmed as
face-to-face from velocity. This keeps the VLM call rare (most frames
won't have an ambiguous pair at all), fast, and scoped to the thing it's
actually suited for - classifying an image - rather than asked to do the
geometric reasoning it's bad at.

-----------------------------------------------------------------------
INPUT
-----------------------------------------------------------------------
Subscribes to /predicted_person_positions (String), CSV format from the
current human_kf_predictor.py:

    track_id,conf,x,y,vx,vy,pred_x,pred_y,horizon,bbox

    bbox is "x1;y1;x2;y2" (semicolons) or the literal string "none" if
    no fresh detection bbox is available this cycle (e.g. a coasted
    prediction). Only tracks with a fresh, non-"none" bbox are eligible
    for the MobileCLIP confirmatory crop.

Optionally subscribes to the raw camera topic (for cropping) - see
RGB_TOPIC below. This is the same topic yolo_detector.py already reads
from; subscribing to it again here is just a second consumer of an
existing topic, no change needed on the publisher side for this part.

-----------------------------------------------------------------------
OUTPUT
-----------------------------------------------------------------------
Publishes /social_groups (String), one line per detected group:

    group_id,group_type,cx,cy,axis_x,axis_y,half_length,half_width,member_ids

  - group_type     : "conversation" or "queue"
  - cx, cy         : center of the group's shared zone (map frame)
  - axis_x, axis_y : unit vector along the group's social axis
  - half_length    : zone half-extent along the axis
  - half_width     : zone half-extent perpendicular to the axis
  - member_ids     : semicolon-separated track IDs, e.g. "12;15"

Intentionally similar in spirit to the existing CSV topics so a future
social_group_cloud_node.py (the next stage - costmap injection) can
parse it the same way predicted_person_cloud_node.py already parses
/predicted_person_positions.

-----------------------------------------------------------------------
WHAT THIS FILE DOES NOT DO YET
-----------------------------------------------------------------------
1. It does not feed the costmap. /social_groups is published but has no
   consumer yet - that's social_group_cloud_node.py, the next piece.
2. MobileCLIP loading/inference is stubbed behind a clear interface
   (classify_facing) so this runs and is testable BEFORE MobileCLIP is
   set up in Docker. Swap the stub body once the model is available.
3. Queue detection here uses a simple "fit a line, check residuals"
   approach. It will need real-world tuning (DBSCAN-style clustering
   first, if you ever have multiple simultaneous queues in frame at
   once) - this version assumes at most one queue-like cluster at a
   time, which matches a single-corridor test scenario.
"""

import math
import itertools

import open_clip
import torch
from PIL import Image

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge


# =======================================================================
# Tunable thresholds - starting guesses, not measured optima. Re-tune
# against your actual staged scenes (two_human.sdf etc.) once running.
# =======================================================================

# --- Conversation (dyad) ---
CONV_MAX_DIST = 1.8          # m; max separation to be a candidate pair
CONV_MIN_DIST = 0.3          # m; below this, treat as same-person noise/overlap
CONV_MAX_SPEED = 0.15        # m/s; "near-stationary" threshold
CONV_MIN_DURATION = 1.5      # s; sustained closeness before flagging

# --- Queue (3+, collinear) ---
QUEUE_MIN_MEMBERS = 3
QUEUE_MAX_SPACING = 1.5      # m; max gap between consecutive queue members
QUEUE_MIN_SPACING = 0.3      # m; below this, treat as crowd/overlap, not a queue
QUEUE_MAX_PERP_DEV = 0.4     # m; max perpendicular deviation from fitted line
QUEUE_MAX_SPEED = 0.4        # m/s; queues can shuffle forward slowly

# --- Zone sizing (applied to both group types) ---
ZONE_BUFFER = 0.4            # m; extra margin added around the raw extent

# --- MobileCLIP confirmatory step ---
# Only invoked when a conversation candidate's facing direction can't be
# resolved from velocity (both members below CONV_MAX_SPEED, so heading
# is meaningless) AND both members have a fresh bbox this cycle.
ENABLE_VLM_CONFIRMATION = True
VLM_MIN_CONFIDENCE = 0.55    # below this similarity margin, fall back to "no group"
RGB_TOPIC = "/oakd/rgb/preview/image_raw"


class TrackState:
    """Latest known state for one tracked person, plus a short history
    used to test how long a candidate pair has been close (CONV_MIN_DURATION)."""

    def __init__(self):
        self.x = None
        self.y = None
        self.vx = 0.0
        self.vy = 0.0
        self.bbox = None          # (x1, y1, x2, y2) or None if stale/unavailable
        self.last_update = None
        # Timestamp at which this track first became a member of *some*
        # close-pair candidate; reset to None when it stops qualifying.
        self.close_since = {}     # other_track_id -> first_close_timestamp


class GroupFormationDetector(Node):
    def __init__(self):
        super().__init__("group_formation_detector")

        self.declare_parameter("input_topic", "/predicted_person_positions")
        self.declare_parameter("output_topic", "/social_groups")

        self.input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        self.output_topic = self.get_parameter("output_topic").get_parameter_value().string_value

        self.tracks = {}  # track_id -> TrackState
        self.bridge = CvBridge()
        self.latest_frame = None  # most recent raw RGB frame, for VLM cropping

        self.get_logger().info("Loading MobileCLIP-S1...")
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            'MobileCLIP-S1', pretrained='datacompdr'
        )
        self.clip_model.eval()
        self.clip_tokenizer = open_clip.get_tokenizer('MobileCLIP-S1')
        self.clip_prompts = [
            "two people facing each other talking",
            "two people standing back to back",
            "two people standing apart not interacting",
        ]
        self.clip_text_tokens = self.clip_tokenizer(self.clip_prompts)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(self.clip_text_tokens)
            self.clip_text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.get_logger().info("MobileCLIP-S1 loaded.")

        self.sub = self.create_subscription(
            String, self.input_topic, self.position_callback, 10
        )

        if ENABLE_VLM_CONFIRMATION:
            self.create_subscription(RosImage, RGB_TOPIC, self.image_callback, 10)

        self.pub = self.create_publisher(String, self.output_topic, 10)

        # Detection runs on its own timer, decoupled from message arrival
        # rate, same pattern as predicted_person_cloud_node.py's publish
        # timer - keeps group detection at a fixed, predictable cadence.
        self.detect_timer = self.create_timer(0.3, self.detect_groups)

        self.get_logger().info("Group formation detector started")
        self.get_logger().info(f"Input : {self.input_topic}")
        self.get_logger().info(f"Output: {self.output_topic}")
        self.get_logger().info(f"VLM confirmation: {'ON' if ENABLE_VLM_CONFIRMATION else 'OFF'}")

    # -------------------------------------------------------------
    # Input handling
    # -------------------------------------------------------------
    def get_ros_time_seconds(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def image_callback(self, msg):
        # Stored only for on-demand cropping in classify_facing(); not
        # processed here to avoid doing image conversion work on every
        # frame when no ambiguous pair currently needs it.
        self.latest_frame = msg

        # TEMP DEBUG: save the full raw frame every 30th callback so we
        # can inspect the camera's actual vertical field of view.
        if not hasattr(self, '_debug_frame_counter'):
            self._debug_frame_counter = 0
        self._debug_frame_counter += 1
        if self._debug_frame_counter % 30 == 0:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            import cv2
            cv2.imwrite("/root/thesis_social_navigation_ws/debug_full_frame.png", frame)

    def position_callback(self, msg):
        now = self.get_ros_time_seconds()
        parts = msg.data.split(",")

        if len(parts) < 9:
            self.get_logger().warn(f"Invalid /predicted_person_positions msg: {msg.data}")
            return

        try:
            track_id = int(float(parts[0]))
            x = float(parts[2])
            y = float(parts[3])
            vx = float(parts[4])
            vy = float(parts[5])
        except ValueError:
            self.get_logger().warn(f"Parse failed: {msg.data}")
            return

        bbox = None
        if len(parts) >= 10 and parts[9] != "none":
            try:
                bx1, by1, bx2, by2 = (int(v) for v in parts[9].split(";"))
                bbox = (bx1, by1, bx2, by2)
            except ValueError:
                bbox = None

        if track_id not in self.tracks:
            self.tracks[track_id] = TrackState()

        t = self.tracks[track_id]
        t.x, t.y, t.vx, t.vy = x, y, vx, vy
        t.bbox = bbox
        t.last_update = now

    # -------------------------------------------------------------
    # Main detection cycle
    # -------------------------------------------------------------
    def detect_groups(self):
        now = self.get_ros_time_seconds()

        # Drop tracks that have gone silent - same staleness pattern used
        # in predicted_person_cloud_node.py / human_kf_predictor.py.
        stale = [tid for tid, t in self.tracks.items()
                 if t.last_update is None or now - t.last_update > 1.0]
        for tid in stale:
            del self.tracks[tid]

        active_ids = list(self.tracks.keys())
        groups = []
        used_in_conversation = set()

        # --- Conversation candidates: all pairs ---
        for id_a, id_b in itertools.combinations(active_ids, 2):
            ta, tb = self.tracks[id_a], self.tracks[id_b]

            dist = math.hypot(ta.x - tb.x, ta.y - tb.y)
            if not (CONV_MIN_DIST <= dist <= CONV_MAX_DIST):
                self._clear_close_since(ta, tb, id_a, id_b)
                continue

            speed_a = math.hypot(ta.vx, ta.vy)
            speed_b = math.hypot(tb.vx, tb.vy)
            if speed_a > CONV_MAX_SPEED or speed_b > CONV_MAX_SPEED:
                self._clear_close_since(ta, tb, id_a, id_b)
                continue

            # Track how long this pair has been continuously close+slow.
            first_seen = ta.close_since.get(id_b)
            if first_seen is None:
                ta.close_since[id_b] = now
                tb.close_since[id_a] = now
                first_seen = now

            duration = now - first_seen
            if duration < CONV_MIN_DURATION:
                continue

            # Geometry alone got us this far (close, stationary, sustained).
            # Facing direction is the genuine ambiguity - velocity heading
            # is meaningless at near-zero speed for both members.
            confirmed = True
            if ENABLE_VLM_CONFIRMATION:
                confirmed = self.classify_facing(ta, tb)
                if confirmed is None:
                    # VLM unavailable/inconclusive this cycle - fall back
                    # to NOT flagging, rather than guessing. Conservative
                    # choice: a missed conversation zone is recoverable
                    # next cycle; a wrongly-claimed one wastes costmap
                    # space for no reason.
                    continue

            if confirmed:
                groups.append(self._build_conversation_zone(ta, tb, id_a, id_b))
                used_in_conversation.add(id_a)
                used_in_conversation.add(id_b)

        # --- Queue candidates: remaining tracks not already in a conversation ---
        queue_candidates = [tid for tid in active_ids if tid not in used_in_conversation]
        queue_group = self._detect_queue(queue_candidates)
        if queue_group is not None:
            groups.append(queue_group)

        self._publish_groups(groups)

    def _clear_close_since(self, ta, tb, id_a, id_b):
        ta.close_since.pop(id_b, None)
        tb.close_since.pop(id_a, None)

    # -------------------------------------------------------------
    # Conversation zone geometry
    # -------------------------------------------------------------
    def _build_conversation_zone(self, ta, tb, id_a, id_b):
        cx = (ta.x + tb.x) / 2.0
        cy = (ta.y + tb.y) / 2.0

        dx = tb.x - ta.x
        dy = tb.y - ta.y
        dist = math.hypot(dx, dy)

        axis_x = dx / dist if dist > 1e-6 else 1.0
        axis_y = dy / dist if dist > 1e-6 else 0.0

        half_length = dist / 2.0 + ZONE_BUFFER
        half_width = ZONE_BUFFER

        group_id = f"conv_{min(id_a, id_b)}_{max(id_a, id_b)}"
        return (group_id, "conversation", cx, cy, axis_x, axis_y,
                half_length, half_width, [id_a, id_b])

    # -------------------------------------------------------------
    # Queue detection - simple line fit + spacing/residual check.
    #
    # NOTE: this assumes at most one queue-like cluster among the
    # candidates at a time. If you need multiple simultaneous queues,
    # cluster `queue_candidates` first (e.g. by mutual distance) and run
    # this per-cluster instead of on the whole list.
    # -------------------------------------------------------------
    def _detect_queue(self, candidate_ids):
        if len(candidate_ids) < QUEUE_MIN_MEMBERS:
            return None

        pts = [(tid, self.tracks[tid].x, self.tracks[tid].y) for tid in candidate_ids]

        for tid, _, _ in pts:
            t = self.tracks[tid]
            if math.hypot(t.vx, t.vy) > QUEUE_MAX_SPEED:
                return None  # someone's moving too fast to be queuing

        # Fit a line through the points via simple PCA (principal axis),
        # avoids needing numpy as a hard new dependency for just this.
        n = len(pts)
        mean_x = sum(p[1] for p in pts) / n
        mean_y = sum(p[2] for p in pts) / n

        sxx = sum((p[1] - mean_x) ** 2 for p in pts)
        syy = sum((p[2] - mean_y) ** 2 for p in pts)
        sxy = sum((p[1] - mean_x) * (p[2] - mean_y) for p in pts)

        # Principal axis angle from the 2x2 covariance matrix.
        theta = 0.5 * math.atan2(2 * sxy, sxx - syy)
        axis_x, axis_y = math.cos(theta), math.sin(theta)

        # Project each point onto the axis and check perpendicular deviation.
        projections = []
        max_perp = 0.0
        for tid, px, py in pts:
            rel_x, rel_y = px - mean_x, py - mean_y
            along = rel_x * axis_x + rel_y * axis_y
            perp = abs(-rel_x * axis_y + rel_y * axis_x)
            max_perp = max(max_perp, perp)
            projections.append((along, tid))

        if max_perp > QUEUE_MAX_PERP_DEV:
            return None  # not collinear enough

        projections.sort()

        # Check consecutive spacing is regular and within bounds.
        for i in range(len(projections) - 1):
            gap = projections[i + 1][0] - projections[i][0]
            if not (QUEUE_MIN_SPACING <= gap <= QUEUE_MAX_SPACING):
                return None

        along_values = [p[0] for p in projections]
        span = along_values[-1] - along_values[0]
        half_length = span / 2.0 + ZONE_BUFFER
        half_width = ZONE_BUFFER

        member_ids = [tid for _, tid in projections]
        group_id = "queue_" + "_".join(str(i) for i in sorted(member_ids))

        return (group_id, "queue", mean_x, mean_y, axis_x, axis_y,
                half_length, half_width, member_ids)

    # -------------------------------------------------------------
    # MobileCLIP confirmatory classification (STUB)
    #
    # Returns:
    #   True  -> confirmed facing each other (conversation)
    #   False -> confirmed NOT facing each other (e.g. back-to-back)
    #   None  -> inconclusive / model unavailable this cycle
    #
    # Swap the body of this method once MobileCLIP is set up in Docker.
    # Everything around it (when it's called, how its result is used)
    # already matches the final integration point.
    # -------------------------------------------------------------
    def classify_facing(self, ta, tb):
        if ta.bbox is None or tb.bbox is None or self.latest_frame is None:
            return None

        frame = self.bridge.imgmsg_to_cv2(self.latest_frame, desired_encoding="bgr8")
        crop = self._crop_union(frame, ta.bbox, tb.bbox, pad=20)
        if crop is None:
            return None

        crop_rgb = crop[:, :, ::-1]
        pil_image = Image.fromarray(crop_rgb)
        pil_image.save("/root/thesis_social_navigation_ws/debug_clip_crop.png")  # TEMP DEBUG
        image_input = self.clip_preprocess(pil_image).unsqueeze(0)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarities = (100.0 * image_features @ self.clip_text_features.T).softmax(dim=-1)

        best_idx = int(similarities.argmax())
        best_score = float(similarities[0, best_idx])

        self.get_logger().info(
            f"classify_facing: best='{self.clip_prompts[best_idx]}' "
            f"score={best_score:.3f}"
        )

        if best_score < VLM_MIN_CONFIDENCE:
            return None

        return best_idx == 0

    @staticmethod
    def _crop_union(frame, bbox_a, bbox_b, pad=20):
        h, w = frame.shape[:2]
        x1 = max(0, min(bbox_a[0], bbox_b[0]) - pad)
        y1 = max(0, min(bbox_a[1], bbox_b[1]) - pad)
        x2 = min(w - 1, max(bbox_a[2], bbox_b[2]) + pad)
        y2 = min(h - 1, max(bbox_a[3], bbox_b[3]) + pad)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    # -------------------------------------------------------------
    # Output
    # -------------------------------------------------------------
    def _publish_groups(self, groups):
        for (group_id, group_type, cx, cy, axis_x, axis_y,
             half_length, half_width, member_ids) in groups:

            out = String()
            out.data = (
                f"{group_id},"
                f"{group_type},"
                f"{cx:.3f},{cy:.3f},"
                f"{axis_x:.3f},{axis_y:.3f},"
                f"{half_length:.3f},{half_width:.3f},"
                f"{';'.join(str(i) for i in member_ids)}"
            )
            self.pub.publish(out)

            self.get_logger().info(
                f"[{group_type}] {group_id} center=({cx:.2f},{cy:.2f}) "
                f"axis=({axis_x:.2f},{axis_y:.2f}) "
                f"half_len={half_length:.2f} half_wid={half_width:.2f} "
                f"members={member_ids}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = GroupFormationDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
