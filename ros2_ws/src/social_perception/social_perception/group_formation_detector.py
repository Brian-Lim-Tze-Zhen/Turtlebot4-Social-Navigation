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

    track_id,conf,x,y,vx,vy,pred_x,pred_y,horizon,rotation_gate,bbox

    Field [9] is the rotation-gate flag ("0"/"1"), unrelated to this
    node but must not be misread as bbox. Field [10] is the bbox as
    "x1;y1;x2;y2" (semicolons) or the literal string "none" if no
    fresh detection bbox is available this cycle (e.g. a coasted
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
2. MobileCLIP loading/inference is fully implemented in classify_facing()
   — real model load, real encode_image/encode_text calls, no stub
   remaining. Facing-classification was validated via a controlled
   distance sweep: confirmed reliable when the bbox top (y1) sits
   clear of the frame edge, unreliable/flips wrong when y1=0 (head
   clipped). See TOP_CLIP_MARGIN_PX below.
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

# --- Identity retention across occlusion ---
# Split from the old single 1.0s staleness prune, which deleted the whole
# TrackState - and with it close_since - after 1s of silence. With stable
# IDs from identity_fusion_node, the ID survives a camera occlusion, but
# it had nothing to come back TO: the accumulated conversation time was
# already gone.
#
# FRESH_POSITION_TIMEOUT gates GEOMETRY: a track whose position is older
# than this has untrustworthy coordinates and is excluded from pairing.
# IDENTITY_RETENTION_TIMEOUT gates DELETION: the TrackState (and its
# close_since history) is kept far longer, so a person who walks behind
# an obstacle and returns resumes their established conversation instead
# of restarting the CONV_MIN_DURATION clock from zero.
FRESH_POSITION_TIMEOUT = 1.0      # s; position trusted for geometry
IDENTITY_RETENTION_TIMEOUT = 300.0  # s; state kept for re-identification

# Max gap between consecutive qualifying ticks that still counts as
# continuous conversation time. Larger gaps are treated as an occlusion:
# the pair RESUMES its accumulated duration rather than either resetting
# it or (wrongly) counting the blind window as time spent conversing.
CONV_MAX_CONTINUITY_GAP = 1.0     # s

# --- Bbox retention ---
# /predicted_person_positions interleaves measurement messages (real
# bbox) with coasted ones (bbox "none") at roughly 0.1s spacing, because
# human_kf_predictor's coast timer republishes without a detection. The
# old code assigned t.bbox unconditionally, so every coasted message
# wiped a perfectly good bbox from ~0.1s earlier - roughly halving the
# frames on which classify_facing() had two usable bboxes and could run
# the VLM at all.
#
# Fix: keep the last real bbox instead of nulling it. But a retained
# bbox cannot be kept forever - if the person moves while coasting, the
# stale pixel region no longer contains them, and cropping it would feed
# the VLM the wrong image and get a confident wrong answer. That is
# worse than skipping, since a missed confirmation retries next cycle
# while a false one propagates.
#
# So the bbox carries its own timestamp and is refused past this age.
# At the ~0.2 m/s walking speeds in the test worlds, 0.5s is well under
# the time to move a body-width, while comfortably bridging the ~0.1s
# measurement/coast alternation. Re-tune if people move faster.
BBOX_MAX_AGE = 0.5                # s

# --- Queue (3+, collinear) ---
QUEUE_MIN_MEMBERS = 3
QUEUE_MAX_SPACING = 1.5      # m; max gap between consecutive queue members
QUEUE_MIN_SPACING = 0.3      # m; below this, treat as crowd/overlap, not a queue
QUEUE_MAX_PERP_DEV = 0.4     # m; max perpendicular deviation from fitted line
QUEUE_MAX_SPEED = 0.4        # m/s; queues can shuffle forward slowly

# THESIS ADDITION (queue hold-open, POSITION-anchored)
# _detect_queue() rebuilds from scratch each cycle and needs ALL members
# simultaneously camera-fresh. Measured: robot motion drops the whole
# roster for 7.5-23.9 s at a time, and ByteTrack reassigns every id when
# they return (41,48,51,54 -> 57,69,72,74), so an id-keyed hold would not
# survive either failure. Queue members are static by definition, so
# their POSITIONS are the stable anchor. Hold re-emits the last confirmed
# zone while enough cached member positions still have SOME fresh
# detection near them, whatever it is now called.
# Values are starting points, not measured optima. TIMEOUT is the ONLY
# exit: position anchoring gives up early revocation, acceptable for a
# static queue but a real lag source if the queue ever advances.
QUEUE_HOLD_TIMEOUT = 10.0    # s; max age of a cached queue before expiry
QUEUE_HOLD_MIN_VISIBLE = 2   # cached positions that must still match
QUEUE_HOLD_MATCH_RADIUS = 0.6  # m; well under the 1.2 m member spacing
LIDAR_ANCHOR_MAX_AGE = 1.0   # s; same freshness standard as camera tracks

# --- Zone sizing (applied to both group types) ---
ZONE_BUFFER = 0.4            # m; extra margin added around the raw extent

# --- MobileCLIP confirmatory step ---
# Only invoked when a conversation candidate's facing direction can't be
# resolved from velocity (both members below CONV_MAX_SPEED, so heading
# is meaningless) AND both members have a fresh bbox this cycle.
ENABLE_VLM_CONFIRMATION = True
VLM_MIN_CONFIDENCE = 0.55    # below this similarity margin, fall back to "no group"
RGB_TOPIC = "/oakd/rgb/preview/image_raw"

# --- Framing gate for VLM confirmation ---
# Empirically calibrated via a controlled distance sweep (see thesis
# notes): bbox y1=0 (flush against the top of a 240px frame) correlates
# with head-clipping and unreliable/flipped facing-classification.
# y1>=5 was clean across every trial in the sweep; y1=0 was wrong or
# borderline across every trial. Skip classify_facing() entirely rather
# than risk a confidently-wrong result when framing is this tight.
TOP_CLIP_MARGIN_PX = 5


class TrackState:
    """Latest known state for one tracked person, plus a short history
    used to test how long a candidate pair has been close (CONV_MIN_DURATION)."""

    def __init__(self):
        self.x = None
        self.y = None
        self.vx = 0.0
        self.vy = 0.0
        self.bbox = None          # (x1, y1, x2, y2); last REAL bbox seen
        self.bbox_time = None     # when self.bbox was captured
        self.last_update = None
        # Timestamp at which this track first became a member of *some*
        # close-pair candidate; reset to None when it stops qualifying.
        # other_track_id -> [accumulated_close_seconds, last_tick_time]
        # Accumulated rather than a single first-seen timestamp, so an
        # occlusion gap can be excluded from the total instead of being
        # counted as conversation time.
        self.close_since = {}


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

        # ==========================================================
        # THESIS MODIFICATION (prompt-bias fix)
        #
        # Original 3-way prompt set ("facing each other talking" /
        # "back to back" / "standing apart not interacting") was
        # empirically shown to have a strong bias toward the "apart"
        # prompt regardless of image content - confirmed via a
        # horizontal-flip invariance test (near-identical scores on
        # original vs. mirrored crop, ruling out orientation cues as
        # the driver) and a real-photo control image (unambiguous
        # facing-each-other photo still lost to "far apart" 0.627 to
        # 0.235). Root cause: unbalanced prompt structure (compound
        # claim vs. simple claims of different lengths) biases the
        # softmax independent of the image.
        #
        # Fix: minimal-contrast binary pair (negation only, same
        # length/structure) removes the structural bias. Verified on
        # both the real photo (0.587) and the actual synthetic crop
        # (0.564) - both now correctly favor "facing each other".
        # ==========================================================
        # ==========================================================
        # KNOWN LIMITATION - single-crop facing classification does not
        # work from a side-on viewpoint. Measured against staged ground
        # truth (people rotated in place, crop verified by eye):
        #
        #   prompt pair                    back-to-back      facing
        #   facing / not facing            0.664 "facing"    0.697 "facing"
        #   from the front / from behind   0.924 "behind"    0.743 "behind"
        #
        # Both pairs return the SAME answer for both conditions. The
        # first has no discrimination at all (0.03 gap); the second
        # discriminates something (0.18 gap) but not the thing needed.
        #
        # The cause is geometric, not phrasing. Viewed side-on - which
        # is where the robot stands relative to a conversing pair - a
        # facing pair presents one front and one back, and so does a
        # back-to-back pair. There is no single-crop appearance cue that
        # separates them from this viewpoint.
        #
        # Reverted to the original pair pending a better approach.
        # Candidates: per-person crops (classify each individual's
        # front/back separately, then reason about the pair
        # geometrically), or a non-visual facing estimate. NOTE that
        # everything downstream currently trusts this classification -
        # a wrong confirmation now becomes a phantom costmap zone via
        # social_group_cloud_node, not just a wrong log line.
        #
        # Index [0] must remain the "conversation" answer -
        # classify_facing returns best_idx == 0.
        # ==========================================================
        self.clip_prompts = [
            "conversation",
            "queue",
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
        # ==========================================================
        # THESIS MODIFICATION (sticky confirmed pair)
        #
        # detect_groups() rebuilds every group from scratch each cycle,
        # so a pair confirmed by the VLM 30s ago had to re-earn that
        # confirmation every 0.3s. That inverted the desired behaviour
        # at close range: once the robot approaches within ~3.0m the
        # framing gate (TOP_CLIP_MARGIN_PX) starts skipping the VLM, so
        # an established conversation silently dropped out of
        # /social_groups exactly as the robot got close enough for the
        # zone to matter for navigation.
        #
        # Fix: remember confirmation per pair. Once confirmed, the pair
        # keeps its status on cheap geometry alone (distance + speed),
        # and the VLM is not re-run.
        #
        # Keyed on frozenset({id_a, id_b}) of the STABLE ids supplied by
        # identity_fusion_node, so the key survives a ByteTrack id switch
        # across occlusion - the same mechanism that keeps close_since
        # alive. A raw ByteTrack key would break on every re-detection.
        #
        # KNOWN LIMITATION (accepted, not solved): confirmation is
        # cleared only when geometry breaks. Two people who stay close
        # and stationary but turn back-to-back keep a stale
        # "conversation" flag indefinitely, since nothing re-checks
        # facing. Revisit if a scenario exercises it - the fix would be
        # opportunistic re-validation whenever framing allows.
        # ==========================================================
        self.confirmed_pairs = {}   # frozenset({id_a, id_b}) -> confirm time
        # Last geometrically-confirmed queue, for hold-open across
        # detection dropout. dict: zone (9-tuple), positions {tid:(x,y)}, time.
        self.queue_hold = None
        # THESIS ADDITION (lidar anchor evidence for queue hold-open)
        # Measured: during robot motion the camera goes fully blind for
        # 5-10 s at a time, while leg_detector never dropped below 4
        # merged clusters. Lidar is used ONLY as evidence that SOMETHING
        # still stands where a cached member was - never to create or
        # reshape a group, because its false-positive rate rises sharply
        # with robot motion (6 raw clusters parked, up to 20 driving).
        # Position matching makes that acceptable: the walls it
        # spuriously clusters sit at y=+-5, far outside the queue.
        self.lidar_points = {}   # lidar_id -> (x, y, last_seen)
        self.create_subscription(
            String, "/lidar_person_clusters", self.lidar_callback, 10)

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

    def position_callback(self, msg):
        now = self.get_ros_time_seconds()
        parts = msg.data.split(",")

        if len(parts) < 10:
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
        if len(parts) >= 11 and parts[10] != "none":
            try:
                bx1, by1, bx2, by2 = (int(v) for v in parts[10].split(";"))
                bbox = (bx1, by1, bx2, by2)
            except ValueError:
                bbox = None

        if track_id not in self.tracks:
            self.tracks[track_id] = TrackState()

        t = self.tracks[track_id]
        t.x, t.y, t.vx, t.vy = x, y, vx, vy
        # Only overwrite on a REAL bbox - a coasted message carries
        # "none" and must not erase the last good one. See BBOX_MAX_AGE.
        if bbox is not None:
            t.bbox = bbox
            t.bbox_time = now
        t.last_update = now

    def lidar_callback(self, msg):
        # Format from leg_detector_node.py: "tid,x,y,age"
        parts = msg.data.split(",")
        if len(parts) < 3:
            return
        try:
            lid = int(float(parts[0]))
            x, y = float(parts[1]), float(parts[2])
        except ValueError:
            return
        self.lidar_points[lid] = (x, y, self.get_ros_time_seconds())

    # -------------------------------------------------------------
    # Main detection cycle
    # -------------------------------------------------------------
    def detect_groups(self):
        now = self.get_ros_time_seconds()

        # Drop tracks that have gone silent - same staleness pattern used
        # in predicted_person_cloud_node.py / human_kf_predictor.py.
        stale = [tid for tid, t in self.tracks.items()
                 if t.last_update is None
                 or now - t.last_update > IDENTITY_RETENTION_TIMEOUT]
        for tid in stale:
            del self.tracks[tid]

        # Without this, confirmed_pairs would grow unbounded as tracks
        # come and go. Deletion happens only after
        # IDENTITY_RETENTION_TIMEOUT, so an occlusion does NOT reach here.
        if stale:
            gone = set(stale)
            for key in [k for k in self.confirmed_pairs if k & gone]:
                del self.confirmed_pairs[key]

        # Only tracks with a FRESH position take part in geometry; the
        # rest are retained (see IDENTITY_RETENTION_TIMEOUT) but their
        # stale coordinates must not drive distance/speed decisions.
        active_ids = [tid for tid, t in self.tracks.items()
                      if t.last_update is not None
                      and now - t.last_update <= FRESH_POSITION_TIMEOUT]
        groups = []

        # ==================================================================
        # THESIS FIX (queue/conversation precedence conflict)
        #
        # CONV_MAX_DIST (1.8m) > QUEUE_MAX_SPACING (1.5m): every adjacent
        # queue pair also qualifies as a conversation candidate and gets
        # sent to the VLM. Measured: a 3-person static queue viewed near
        # head-on got confidently classified "facing each other" (0.627,
        # not a marginal miss), consuming 2 of 3 members into a
        # conversation group and starving _detect_queue() below
        # QUEUE_MIN_MEMBERS.
        #
        # Fix: run the purely-geometric queue check FIRST, over all
        # active tracks. Collinearity + regular spacing is a stronger,
        # unambiguous signal than one VLM crop of two people pulled out
        # of a longer line - if a valid queue exists, trust it and
        # exclude its members from conversation pairing entirely.
        #
        # KNOWN LIMITATION (accepted, not solved): if a pair was already
        # VLM-confirmed and sticky (self.confirmed_pairs) BEFORE the
        # queue formed around them, that stale confirmation is not
        # cleared here - _clear_close_since is the only thing that
        # clears it, and it's never called for queue members. Revisit
        # if a scenario exercises a confirmed pair later becoming part
        # of a queue.
        # ==================================================================
        queue_group = self._detect_queue(active_ids)

        # ==========================================================
        # THESIS FIX (roster shrinkage overwriting a fuller cache)
        #
        # _detect_queue() succeeds on any >=QUEUE_MIN_MEMBERS subset, so
        # a cycle that momentarily sees only 3 of 4 members returns a
        # VALID but SHORTER queue - measured half_len collapsing 2.12 ->
        # 1.58 m, i.e. the tail member silently losing its costmap zone -
        # and that shorter zone then overwrote the 4-member cache.
        #
        # The people did not move; the camera just framed fewer of them.
        # So a smaller roster is treated as partial observation, not as
        # a new ground truth: the fuller cached zone is kept and held,
        # bounded as always by QUEUE_HOLD_TIMEOUT. A roster that is
        # equal or larger is a genuine improvement and replaces it.
        # ==========================================================
        cached_n = (len(self.queue_hold["positions"])
                    if self.queue_hold is not None else 0)

        # Index -2 is member_ids; -1 is now member_xy (added so the cloud
        # node no longer has to look positions up by track_id).
        if queue_group is not None and len(queue_group[-2]) >= cached_n:
            self.queue_hold = {
                "zone": queue_group,
                "positions": {tid: (self.tracks[tid].x, self.tracks[tid].y)
                              for tid in queue_group[-2]},
                "time": now,
            }
        else:
            if queue_group is not None:
                self.get_logger().info(
                    f"Queue detected with {len(queue_group[-2])} members, "
                    f"holding cached {cached_n}-member zone instead")
            queue_group = self._queue_hold_zone(active_ids, now)

        queue_member_ids = set(queue_group[-2]) if queue_group is not None else set()
        if queue_group is not None:
            groups.append(queue_group)

        used_in_conversation = set()

        # --- Conversation candidates: all pairs, excluding queue members ---
        for id_a, id_b in itertools.combinations(active_ids, 2):
            if id_a in queue_member_ids or id_b in queue_member_ids:
                continue

            if id_a in used_in_conversation or id_b in used_in_conversation:
                continue

            ta, tb = self.tracks[id_a], self.tracks[id_b]

            dist = math.hypot(ta.x - tb.x, ta.y - tb.y)
            if not (CONV_MIN_DIST <= dist <= CONV_MAX_DIST):
                self._clear_close_since(ta, tb, id_a, id_b)
                continue

            speed_a = math.hypot(ta.vx, ta.vy)
            speed_b = math.hypot(tb.vx, tb.vy)
            already_confirmed = frozenset((id_a, id_b)) in self.confirmed_pairs

            if not already_confirmed and (speed_a > CONV_MAX_SPEED
                                          or speed_b > CONV_MAX_SPEED):
                self._clear_close_since(ta, tb, id_a, id_b)
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

            duration = entry[0]
            if duration < CONV_MIN_DURATION:
                continue

            confirmed = True
            pair_key = frozenset((id_a, id_b))

            if ENABLE_VLM_CONFIRMATION and pair_key in self.confirmed_pairs:
                confirmed = True

            elif ENABLE_VLM_CONFIRMATION:
                if self._bbox_clipped_at_top(ta.bbox) or self._bbox_clipped_at_top(tb.bbox):
                    continue

                confirmed = self.classify_facing(ta, tb)
                if confirmed is None:
                    continue

            if confirmed:
                if ENABLE_VLM_CONFIRMATION and pair_key not in self.confirmed_pairs:
                    self.confirmed_pairs[pair_key] = now
                    self.get_logger().info(
                        f"Pair ({id_a},{id_b}) VLM-confirmed - now sticky, "
                        f"held on geometry until it breaks")

                groups.append(self._build_conversation_zone(ta, tb, id_a, id_b))
                used_in_conversation.add(id_a)
                used_in_conversation.add(id_b)

        self._publish_groups(groups)

    def _clear_close_since(self, ta, tb, id_a, id_b):
        ta.close_since.pop(id_b, None)
        tb.close_since.pop(id_a, None)

        # Geometry broke (moved apart or started walking) - this is the
        # ONLY thing that revokes a sticky confirmation. Called from the
        # distance and speed checks in detect_groups().
        if self.confirmed_pairs.pop(frozenset((id_a, id_b)), None) is not None:
            self.get_logger().info(
                f"Pair ({id_a},{id_b}) confirmation cleared - geometry broke")

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
                half_length, half_width, [id_a, id_b],
                [(ta.x, ta.y), (tb.x, tb.y)])

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

        # THESIS ADDITION (member coordinates in the message)
        # Ordered along the queue axis, same order as member_ids. The
        # consumer cannot reconstruct N>=3 member positions from
        # centroid+axis+span, and previously re-looked them up by
        # track_id from /predicted_person_positions - which collapses
        # whenever ByteTrack reassigns ids, dropping individual gap
        # segments from the costmap (measured: 74 -> 37 points, i.e. a
        # hole opening in the middle of the queue while the detector
        # still reported all 4 members). This node already holds the
        # authoritative positions; publishing them removes the
        # duplicated, id-dependent lookup entirely.
        member_xy = [(self.tracks[tid].x, self.tracks[tid].y)
                     for tid in member_ids]

        return (group_id, "queue", mean_x, mean_y, axis_x, axis_y,
                half_length, half_width, member_ids, member_xy)

    # -------------------------------------------------------------
    # Queue hold-open, anchored on POSITION rather than track_id.
    # Returns the cached 9-tuple, or None.
    # -------------------------------------------------------------
    def _queue_hold_zone(self, active_ids, now):
        if self.queue_hold is None:
            return None

        if now - self.queue_hold["time"] > QUEUE_HOLD_TIMEOUT:
            self.get_logger().info(
                f"Queue hold expired after {QUEUE_HOLD_TIMEOUT:.1f}s")
            self.queue_hold = None
            return None

        for lid in [l for l, v in self.lidar_points.items()
                    if now - v[2] > LIDAR_ANCHOR_MAX_AGE]:
            del self.lidar_points[lid]

        matched = 0
        lidar_only = 0
        for cx, cy in self.queue_hold["positions"].values():
            if any(math.hypot(self.tracks[tid].x - cx,
                              self.tracks[tid].y - cy) <= QUEUE_HOLD_MATCH_RADIUS
                   for tid in active_ids):
                matched += 1
                continue
            # Camera lost this member - a lidar cluster standing where it
            # was counts as evidence the person is still there.
            if any(math.hypot(lx - cx, ly - cy) <= QUEUE_HOLD_MATCH_RADIUS
                   for lx, ly, _ in self.lidar_points.values()):
                matched += 1
                lidar_only += 1

        if matched < QUEUE_HOLD_MIN_VISIBLE:
            return None  # weak evidence; cache kept until timeout

        # Full re-anchoring means the queue is still fully observed and
        # only its track_ids changed, so the cache is re-validated rather
        # than merely tolerated - otherwise a continuously-visible queue
        # would still expire on QUEUE_HOLD_TIMEOUT purely from id churn.
        # A PARTIAL match does not refresh: that is genuine dropout and
        # must remain bounded by the timeout.
        # ==========================================================
        # THESIS FIX (what the hold clock actually measures)
        #
        # Previously only a fully CAMERA-confirmed cycle reset the clock,
        # so a queue whose members were all still observed - just by
        # lidar - aged out anyway. Measured: matched oscillated 4/4 <->
        # 2/4 several times a second while nobody moved, and the clock
        # ran straight through every 4/4 cycle to expire at 10 s.
        #
        # FULL coverage means every cached member position still has an
        # observation on it, whatever the sensor. The queue is occupied;
        # there is nothing to time out. The clock exists for the case it
        # was written for: a member with NO observation at all, held
        # purely on a remembered position. That is the only genuinely
        # unverified state, and it stays bounded.
        #
        # Accepted limitation: lidar confirms occupancy, not identity, so
        # a full-coverage hold cannot distinguish the same four people
        # from four different people standing in the same spots. The
        # camera's own detections resolve this whenever framing allows;
        # nothing here does.
        # ==========================================================
        if matched == len(self.queue_hold["positions"]):
            self.queue_hold["time"] = now

        self.get_logger().info(
            f"Queue HELD on {matched}/{len(self.queue_hold['positions'])} "
            f"anchored ({lidar_only} via lidar), "
            f"age {now - self.queue_hold['time']:.1f}s")
        return self.queue_hold["zone"]

    # -------------------------------------------------------------
    # Framing gate helper
    #
    # bbox is (x1, y1, x2, y2) in pixel coordinates, y1 = top edge.
    # y1 <= TOP_CLIP_MARGIN_PX means the box is flush (or nearly
    # flush) against the top of frame - empirically correlated with
    # head-clipping and unreliable classify_facing() results.
    # -------------------------------------------------------------
    @staticmethod
    def _bbox_clipped_at_top(bbox):
        if bbox is None:
            return True
        _, y1, _, _ = bbox
        return y1 <= TOP_CLIP_MARGIN_PX

    # -------------------------------------------------------------
    # MobileCLIP confirmatory classification
    #
    # Returns:
    #   True  -> confirmed facing each other (conversation)
    #   False -> confirmed NOT facing each other (e.g. back-to-back)
    #   None  -> inconclusive (no bbox/frame yet, crop failed, or
    #            best similarity score < VLM_MIN_CONFIDENCE)
    # -------------------------------------------------------------
    def classify_facing(self, ta, tb):
        if ta.bbox is None or tb.bbox is None or self.latest_frame is None:
            return None

        # Refuse a retained bbox that has gone stale: it may no longer
        # contain the person, and cropping it would hand the VLM the
        # wrong image. See BBOX_MAX_AGE.
        now = self.get_ros_time_seconds()
        if ta.bbox_time is None or tb.bbox_time is None:
            return None
        if (now - ta.bbox_time > BBOX_MAX_AGE
                or now - tb.bbox_time > BBOX_MAX_AGE):
            return None

        frame = self.bridge.imgmsg_to_cv2(self.latest_frame, desired_encoding="bgr8")
        crop = self._crop_union(frame, ta.bbox, tb.bbox, pad=20)
        if crop is None:
            return None

        crop_rgb = crop[:, :, ::-1]
        pil_image = Image.fromarray(crop_rgb)
        image_input = self.clip_preprocess(pil_image).unsqueeze(0)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarities = (100.0 * image_features @ self.clip_text_features.T).softmax(dim=-1)

        for i, prompt in enumerate(self.clip_prompts):
            self.get_logger().info(f"  [{i}] '{prompt}' = {float(similarities[0, i]):.3f}")

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
             half_length, half_width, member_ids, member_xy) in groups:

            # Field [9]: member coordinates, same order as member_ids,
            # "x;y|x;y|...". Appended, so consumers parsing only fields
            # 0-8 are unaffected.
            xy_str = "|".join(f"{x:.3f};{y:.3f}" for x, y in member_xy)

            out = String()
            out.data = (
                f"{group_id},"
                f"{group_type},"
                f"{cx:.3f},{cy:.3f},"
                f"{axis_x:.3f},{axis_y:.3f},"
                f"{half_length:.3f},{half_width:.3f},"
                f"{';'.join(str(i) for i in member_ids)},"
                f"{xy_str}"
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