#!/usr/bin/env python3
"""
social_group_cloud_node.py

THESIS ADDITION - injects social group zones into the Nav2 costmap.

-----------------------------------------------------------------------
WHAT THIS DOES
-----------------------------------------------------------------------
group_formation_detector.py publishes /social_groups but nothing
consumed it, so detected conversations had no effect on how the robot
drove. This node closes that loop: it converts each conversation group
into a synthetic PointCloud2 obstacle region that Nav2's costmap layer
consumes, so the planner routes around the group instead of through it.

-----------------------------------------------------------------------
WHAT REGION IS FILLED, AND WHY
-----------------------------------------------------------------------
This fills the O-SPACE only - the shared space BETWEEN the two people -
not their bodies.

The o-space is the term from Kendon's F-formation work for the inner
region a conversing group encircles and orients toward. Walking through
it is the socially disruptive act; passing behind one member is not.
Filling only the o-space means the robot refuses to thread between two
people in conversation, while still being able to pass close behind
either of them. That keeps tight spaces navigable.

The alternative (filling the whole pair, bodies included) was considered
and rejected: it produces a much larger blocked region, which combined
with Nav2's inflation layer can wall off a corridor entirely, and risks
the same local-planner stalling already seen with oversized predicted-
person ellipses.

The people's own bodies are NOT covered here on purpose -
predicted_person_cloud_node.py already places a disk on each person.
Double-covering them would only distort the cost gradient.

-----------------------------------------------------------------------
SIZING, AND THE PROXEMIC BASIS
-----------------------------------------------------------------------
Hall's proxemic zones: intimate <0.45m, personal 0.45-1.2m, social
1.2-3.6m. Conversation concentrates in the personal zone and the near
half of social, which is why group_formation_detector uses
CONV_MAX_DIST=1.8m - it covers all of personal plus the lower part of
social. Pairs beyond that are more plausibly co-located than conversing.

The o-space grows with the pair's actual separation (verified):

    separation 0.60m -> skipped (no meaningful gap)
    separation 0.84m -> o-space 0.34m long, 17 points
    separation 1.00m -> o-space 0.50m long, 31 points
    separation 1.80m -> o-space 1.30m long, 75 points

Below MIN_O_SPACE_HALF_LENGTH the gap is too small to be worth blocking
and nothing is published for that group - better than emitting a
degenerate sliver.

NOTE ON CULTURAL VARIATION (worth stating in the thesis limitations):
preferred conversational distance varies substantially between cultures,
so a fixed 1.8m threshold encodes one norm rather than a universal.

-----------------------------------------------------------------------
COUPLING - READ BEFORE CHANGING CONSTANTS
-----------------------------------------------------------------------
/social_groups publishes half_length = separation/2 + ZONE_BUFFER, where
ZONE_BUFFER is defined in group_formation_detector.py (currently 0.4).
This node subtracts that buffer back out to recover the raw separation,
so SOURCE_ZONE_BUFFER below MUST match ZONE_BUFFER there. If they drift
apart, every zone silently becomes the wrong length - there is no error,
just wrong geometry. Change them together. This coupling only applies
to the CONVERSATION path (queue gap positions come from live lookups,
not from decoding half_length - see QUEUE HANDLING below).

-----------------------------------------------------------------------
QUEUE HANDLING
-----------------------------------------------------------------------
A queue's /social_groups message carries only a centroid and total
span - unlike a conversation's exactly-2-member case, individual
member positions cannot be reconstructed from that alone for N>=3.

So queue members' live positions are looked up separately from
/predicted_person_positions (see QUEUE_POSITIONS_TOPIC), keyed by the
same stable track_id used in /social_groups' member_ids field. This
node then fills the gap between each CONSECUTIVE pair in that list
(the order group_formation_detector's _detect_queue publishes them in,
sorted by projection along the queue's axis), using the exact same
o-space ellipse shape/sizing as a conversation gap - the queue's social
space decomposes into a chain of pairwise gaps rather than one big
region, consistent with only blocking gaps, never bodies.

KNOWN LIMITATION (accepted, not solved): no lidar hold-open for queues
yet, unlike the conversation path's anchor_a/anchor_b mechanism. If the
camera loses ANY queue member, that gap (and only that gap - see
make_queue_gap_points) simply stops updating and the whole group
expires GROUP_TIMEOUT after its last message, rather than being held
open on lidar the way a conversation is. Revisit if queue scenarios
start exercising the camera's occlusion/FOV limits the way conversation
testing already has.

-----------------------------------------------------------------------
INPUT / OUTPUT
-----------------------------------------------------------------------
Subscribes:
  /social_groups (String)
      group_id,group_type,cx,cy,axis_x,axis_y,half_length,half_width,members
  /predicted_person_positions (String) - queue member position lookup only

Publishes:
  /social_group_cloud (PointCloud2, frame "map")

Costmap plugin: use NonPersistentVoxelLayer, NOT VoxelLayer. VoxelLayer
clears via raytracing, which fails for synthetic clouds that have no
real sensor origin, leaving ghost obstacles behind forever.
NonPersistentVoxelLayer rebuilds from scratch each cycle. This is the
same lesson already recorded for predicted_person_cloud_node.py.
"""

import math
import struct

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

import tf2_ros

from std_msgs.msg import String, Header
from sensor_msgs.msg import PointCloud2, PointField


# =======================================================================
# Geometry
# =======================================================================

# MUST match ZONE_BUFFER in group_formation_detector.py - see COUPLING.
SOURCE_ZONE_BUFFER = 0.4

# Backed off from each person so the zone covers the GAP, not the bodies.
# Roughly a body radius; predicted_person_cloud_node already marks the
# people themselves with a 0.40m disk.
BODY_CLEARANCE = 0.25

# Half-extent across the pair's axis. Narrower than the source message's
# half_width (which is just ZONE_BUFFER) because Nav2's inflation layer
# expands whatever is published - this is the raw obstacle, not the
# final keep-out distance the robot will actually respect.
O_SPACE_HALF_WIDTH = 0.35

# Below this the pair are effectively shoulder to shoulder and there is
# no meaningful gap to protect. Publish nothing rather than a sliver.
MIN_O_SPACE_HALF_LENGTH = 0.10

POINT_SPACING = 0.10   # m between synthetic points
POINT_HEIGHT = 0.30    # m; matches predicted_person_cloud_node


# =======================================================================
# Queue gap-filling (camera-fresh only, no lidar hold-open yet)
# =======================================================================
# See QUEUE HANDLING in the module docstring for why this is a separate
# topic subscription rather than a /social_groups schema change.
QUEUE_POSITIONS_TOPIC = "/predicted_person_positions"

# How stale a cached member position may be before it's refused for gap
# geometry. Matches FRESH_POSITION_TIMEOUT in group_formation_detector.py
# - same freshness standard, different file.
QUEUE_POSITION_MAX_AGE = 1.0   # s


# =======================================================================
# Timing / safety
# =======================================================================

PUBLISH_RATE_HZ = 10.0

# =======================================================================
# Lidar-anchored persistence (CONVERSATION path only - see QUEUE
# HANDLING above for why queues don't use this yet)
# =======================================================================
# PROBLEM: the zone vanished exactly when the robot got close, which is
# when it matters most. group_formation_detector only emits a group
# while BOTH members are in active_ids, which needs fresh camera
# positions. The OAK-D preview is narrow - two people 1m apart only both
# fit from ~3m back - so approaching or rotating drops one of them,
# /social_groups goes quiet, and GROUP_TIMEOUT cleared the zone.
#
# That is a camera FOV limit, not a logic error, and the sticky
# confirmed-pair flag cannot fix it: the flag preserves the CONFIRMATION,
# but with no fresh positions there is no pair to publish at all.
#
# FIX: the lidar does not share the camera's occlusion geometry (the same
# asymmetry identity_fusion_node exploits for re-ID). While the camera is
# blind, hold the zone open as long as lidar tracks still sit near where
# the members were, and MOVE the zone to follow those tracks so it stays
# correct if the pair drifts.
#
# /social_groups carries stable CAMERA ids while /lidar_person_clusters
# carries lidar track ids - different numbering, and the binding between
# them lives inside identity_fusion_node and is not published. So members
# are matched to anchors BY POSITION instead, using the camera-lidar
# offset measured empirically (0.043-0.114m, hence a 0.35m gate with
# room to spare, well under the 1.0m person-to-person spacing).
LIDAR_TOPIC = "/lidar_person_clusters"

# Max distance from a member's last known position to a lidar track for
# that track to count as its anchor.
LIDAR_ANCHOR_GATE = 0.35

# Drop a cached lidar track this long after its last message.
LIDAR_TRACK_TIMEOUT = 1.0

# How long a group may be held open on lidar anchors alone, with no
# camera confirmation at all. Bounded so a group cannot persist
# indefinitely on geometry the VLM has never re-examined.
MAX_LIDAR_ONLY_HOLD = 60.0

# Recomputed separation limit while running on lidar anchors. MUST match
# CONV_MAX_DIST in group_formation_detector.py - if the pair walks apart
# while the camera is blind, the zone must die on the same criterion the
# camera path would have used.
CONV_MAX_DIST = 1.8

# Drop a group this long after its last message.
#
# THESIS FIX (timeout sized against the MEASURED publish period)
# The previous value assumed the detector's nominal 0.3 s detect_timer.
# It does not achieve that: with MobileCLIP inference, the PCA fit and
# the hold-open anchoring in the same callback, measured inter-message
# intervals on /social_groups were 0.7 s typical and up to 2.8 s. A 1.0 s
# timeout therefore expired the group on any slow cycle, and the zone was
# rebuilt on the next message - producing repeated costmap dropouts
# (11.1 s total in one 113 s trial) that looked like perception failures
# but were entirely this consumer discarding a live group.
#
# 4.0 s covers the worst measured interval with margin. The cost of the
# larger value is bounded: a genuinely dispersed group persists for up to
# 4 s, and queue members are static, so a stale zone here does not chase
# a moving person. Re-tune if the detector is ever made faster - and
# measure the interval rather than reading detect_timer.
GROUP_TIMEOUT = 4.0

# Robot keep-out. Without this, a zone can place lethal cost under the
# robot's own footprint - the controller then scores every trajectory as
# blocked and the robot stalls or oscillates. This already happened with
# predicted_person_cloud_node's ellipse in head-on encounters.
#
# A conversation zone makes it MORE likely, not less: it is static and
# persistent, and the robot is expected to pass close by. Keep the radius
# footprint-sized - making it large would carve a moving hole through the
# zone and let the robot drive straight through the conversation.
ROBOT_FRAME = "base_link"
ROBOT_KEEPOUT_RADIUS = 0.25


class SocialGroupCloudNode(Node):
    def __init__(self):
        super().__init__("social_group_cloud_node")

        self.declare_parameter("input_topic", "/social_groups")
        self.declare_parameter("output_topic", "/social_group_cloud")
        self.declare_parameter("frame_id", "map")

        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.frame_id = self.get_parameter("frame_id").value

        # group_id -> dict(...); shape differs by group type, see
        # group_callback.
        self.active_groups = {}
        self.lidar_tracks = {}      # lidar_id -> (x, y, last_seen)
        self.person_positions = {}  # track_id -> (x, y, last_seen); queue lookup only
        self.last_robot_xy = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.sub = self.create_subscription(
            String, self.input_topic, self.group_callback, 10)

        self.create_subscription(
            String, LIDAR_TOPIC, self.lidar_callback, 10)

        self.create_subscription(
            String, QUEUE_POSITIONS_TOPIC, self.position_callback, 10)

        self.pub = self.create_publisher(
            PointCloud2, self.output_topic, 10)

        # Publish on a timer rather than per-message, so all active groups
        # appear together in one cloud. Publishing per-message would make
        # each group overwrite the previous one, so only the most recent
        # would ever reach the costmap - the same multi-track bug already
        # fixed in predicted_person_cloud_node.py.
        self.create_timer(1.0 / PUBLISH_RATE_HZ, self.publish_cloud)

        self.get_logger().info("Social group cloud node started")
        self.get_logger().info(f"Input : {self.input_topic}")
        self.get_logger().info(f"Output: {self.output_topic}")
        self.get_logger().info(f"Frame : {self.frame_id}")
        self.get_logger().info(
            f"O-space width: {O_SPACE_HALF_WIDTH * 2:.2f} m | "
            f"body clearance: {BODY_CLEARANCE:.2f} m")
        self.get_logger().info(
            f"Robot keep-out: {ROBOT_KEEPOUT_RADIUS:.2f} m "
            f"around '{ROBOT_FRAME}'")
        self.get_logger().info(
            f"Queue gap-fill: ON (positions from {QUEUE_POSITIONS_TOPIC}, "
            f"no lidar hold-open yet)")

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # -----------------------------------------------------------------
    def lidar_callback(self, msg):
        parts = msg.data.split(",")
        if len(parts) < 3:
            return
        try:
            lid = int(float(parts[0]))
            x = float(parts[1])
            y = float(parts[2])
        except ValueError:
            return
        self.lidar_tracks[lid] = (x, y, self.now())

    # -----------------------------------------------------------------
    # THESIS ADDITION (queue cloud support)
    #
    # Caches live positions from /predicted_person_positions, keyed by
    # the same stable track_id /social_groups' queue member_ids uses.
    # Only consumed by make_queue_gap_points() - conversation zones
    # still get their positions decoded from centre+axis+span, unchanged.
    # -----------------------------------------------------------------
    def position_callback(self, msg):
        parts = msg.data.split(",")
        if len(parts) < 10:
            return
        try:
            track_id = int(float(parts[0]))
            x = float(parts[2])
            y = float(parts[3])
        except ValueError:
            return
        self.person_positions[track_id] = (x, y, self.now())

    def find_anchor(self, x, y, exclude=None):
        """Nearest lidar track to (x, y) within LIDAR_ANCHOR_GATE.
        `exclude` prevents both members of a pair matching the same
        track, which would otherwise collapse the zone to a point."""
        best_id, best_d = None, LIDAR_ANCHOR_GATE
        for lid, (lx, ly, _) in self.lidar_tracks.items():
            if lid == exclude:
                continue
            d = math.hypot(x - lx, y - ly)
            if d < best_d:
                best_id, best_d = lid, d
        return best_id

    # -----------------------------------------------------------------
    def group_callback(self, msg):
        parts = msg.data.split(",")
        if len(parts) < 9:
            self.get_logger().warn(f"Invalid /social_groups msg: {msg.data}")
            return

        try:
            group_id = parts[0].strip()
            group_type = parts[1].strip()
            cx = float(parts[2])
            cy = float(parts[3])
            axis_x = float(parts[4])
            axis_y = float(parts[5])
            half_length = float(parts[6])
        except ValueError:
            self.get_logger().warn(f"Parse failed: {msg.data}")
            return

        if group_type == "conversation":
            # Recover the two members' positions from centre + axis +
            # separation, so they can be matched to lidar anchors later.
            separation = 2.0 * (half_length - SOURCE_ZONE_BUFFER)
            norm = math.hypot(axis_x, axis_y)
            if norm < 1e-6:
                return
            ax, ay = axis_x / norm, axis_y / norm
            h = separation / 2.0

            existing = self.active_groups.get(group_id, {})

            self.active_groups[group_id] = {
                "type": "conversation",
                "member_a": (cx - h * ax, cy - h * ay),
                "member_b": (cx + h * ax, cy + h * ay),
                # Anchor ids are LEARNED while the camera is fresh (see
                # publish_cloud) and reused directly once it goes blind, so a
                # lost member is not re-matched by proximity to whatever
                # happens to be nearby - which is how a member previously
                # ended up 0.9m away, i.e. locked onto the OTHER person.
                "anchor_a": existing.get("anchor_a"),
                "anchor_b": existing.get("anchor_b"),
                "last_seen": self.now(),
                "camera_fresh": True,
            }

        elif group_type == "queue":
            # See QUEUE HANDLING in the module docstring.
            member_ids_str = parts[8].strip()
            try:
                member_ids = [int(v) for v in member_ids_str.split(";") if v]
            except ValueError:
                self.get_logger().warn(f"Bad member_ids in queue msg: {msg.data}")
                return

            if len(member_ids) < 2:
                return

            # ==================================================================
            # THESIS FIX (queue group-identity churn under camera FOV limits)
            #
            # Previously keyed by the literal group_id string, which bakes the
            # CURRENT roster into the key (queue_1_2_3_4 vs queue_1_2_3 are
            # different dict entries). Measured: a 4-person queue outside the
            # camera's simultaneous FOV capacity produces a roster that
            # legitimately fluctuates between 3 and 4 visible members every
            # cycle - not a rare glitch, the NORMAL state for a queue wider
            # than the camera can frame at once. Every fluctuation was
            # creating a brand-new group that then expired via GROUP_TIMEOUT
            # almost immediately, even though the underlying queue was
            # continuously, correctly detected the whole time.
            #
            # Fix: match an incoming queue message to an EXISTING active
            # queue group by membership overlap rather than exact id match -
            # same continuity principle the conversation path already uses
            # (confirmed_pairs keyed by frozenset, anchors surviving
            # occlusion), applied to group identity instead of pair
            # confirmation. A shrinking or growing roster updates the same
            # group in place instead of orphaning the old one.
            #
            # Single-queue assumption carried over unchanged from
            # group_formation_detector.py's own docstring: this only
            # disambiguates safely because at most one queue-like cluster is
            # assumed at a time. Would need real work (matching by centroid
            # proximity, not just any overlap) if that assumption changes.
            # ==================================================================
            new_members = set(member_ids)
            matched_gid = None
            for gid, g in self.active_groups.items():
                if g.get("type") == "queue" and set(g["member_ids"]) & new_members:
                    matched_gid = gid
                    break

            target_gid = matched_gid if matched_gid is not None else group_id

            # THESIS FIX (positions from the message, not an id lookup)
            # Field [9] carries the members' coordinates in member_ids
            # order. Previously these were re-looked-up by track_id from
            # /predicted_person_positions, which fails on every ByteTrack
            # id reassignment and silently drops gap segments. Falls back
            # to the old lookup only if the field is absent (older
            # publisher), so this stays backward compatible.
            member_xy = None
            if len(parts) >= 10 and parts[9].strip():
                try:
                    member_xy = [tuple(float(v) for v in pair.split(";"))
                                 for pair in parts[9].strip().split("|")]
                except ValueError:
                    member_xy = None
                if member_xy is not None and len(member_xy) != len(member_ids):
                    self.get_logger().warn(
                        "member_xy length does not match member_ids - ignoring")
                    member_xy = None

            self.active_groups[target_gid] = {
                "type": "queue",
                "member_ids": member_ids,
                "member_xy": member_xy,
                "last_seen": self.now(),
            }

        else:
            # Unknown group_type - ignore rather than guess a shape for it.
            return

    # -----------------------------------------------------------------
    def o_space_half_length(self, source_half_length):
        """Recover the pair's raw separation from the published
        half_length, then back off a body radius at each end so the zone
        covers only the gap between them. See COUPLING in the docstring."""
        separation = 2.0 * (source_half_length - SOURCE_ZONE_BUFFER)
        return separation / 2.0 - BODY_CLEARANCE

    def make_o_space_points(self, cx, cy, axis_x, axis_y, half_length):
        """Fill an ellipse centred on the pair's midpoint, elongated along
        their shared axis."""
        points = []

        norm = math.hypot(axis_x, axis_y)
        if norm < 1e-6:
            return points
        ax, ay = axis_x / norm, axis_y / norm

        a = half_length
        b = O_SPACE_HALF_WIDTH

        steps_u = int(a / POINT_SPACING)
        steps_v = int(b / POINT_SPACING)

        for iu in range(-steps_u, steps_u + 1):
            u = iu * POINT_SPACING
            for iv in range(-steps_v, steps_v + 1):
                v = iv * POINT_SPACING

                if (u / a) ** 2 + (v / b) ** 2 <= 1.0:
                    # rotate local (u, v) onto the group's axis
                    x = cx + u * ax - v * ay
                    y = cy + u * ay + v * ax
                    points.append((x, y, POINT_HEIGHT))

        return points

    # -----------------------------------------------------------------
    # THESIS ADDITION (queue cloud support)
    #
    # Fills the gap between each CONSECUTIVE pair in member_ids (the
    # order _detect_queue publishes them, sorted along the queue's
    # axis) using the same o-space shape as a conversation.
    #
    # Deliberately looks up each pair fresh from the ORIGINAL member_ids
    # list, not from a pre-filtered "positions we have" list - if one
    # member's position is missing/stale, only the ONE gap touching
    # that member is skipped. Pairing across a missing middle member
    # instead would silently bridge two non-adjacent people and produce
    # an oversized, wrongly-shaped fill - worse than just omitting that
    # gap.
    # -----------------------------------------------------------------
    def make_queue_gap_points(self, member_ids, member_xy=None):
        points = []
        now = self.now()

        # Preferred path: coordinates came with the group message, so
        # every gap is fillable regardless of id churn.
        if member_xy is not None:
            pairs = list(zip(member_xy, member_xy[1:]))
        else:
            pairs = None

        if pairs is not None:
            for (xa, ya), (xb, yb) in pairs:
                dx, dy = xb - xa, yb - ya
                separation = math.hypot(dx, dy)
                a = separation / 2.0 - BODY_CLEARANCE
                if a < MIN_O_SPACE_HALF_LENGTH:
                    continue
                points.extend(self.make_o_space_points(
                    (xa + xb) / 2.0, (ya + yb) / 2.0, dx, dy, a))
            return points

        # Legacy fallback: look positions up by track_id.
        for tid_a, tid_b in zip(member_ids, member_ids[1:]):
            cached_a = self.person_positions.get(tid_a)
            cached_b = self.person_positions.get(tid_b)
            if cached_a is None or cached_b is None:
                continue

            xa, ya, ta = cached_a
            xb, yb, tb = cached_b
            if (now - ta > QUEUE_POSITION_MAX_AGE
                    or now - tb > QUEUE_POSITION_MAX_AGE):
                continue

            dx, dy = xb - xa, yb - ya
            separation = math.hypot(dx, dy)

            a = separation / 2.0 - BODY_CLEARANCE
            if a < MIN_O_SPACE_HALF_LENGTH:
                continue

            gx = (xa + xb) / 2.0
            gy = (ya + yb) / 2.0
            points.extend(self.make_o_space_points(gx, gy, dx, dy, a))

        return points

    # -----------------------------------------------------------------
    def publish_cloud(self):
        now = self.now()

        # Expire cached lidar tracks.
        for lid in [l for l, v in self.lidar_tracks.items()
                    if now - v[2] > LIDAR_TRACK_TIMEOUT]:
            del self.lidar_tracks[lid]

        # Expire cached person positions (queue gap lookups).
        for tid in [t for t, v in self.person_positions.items()
                    if now - v[2] > QUEUE_POSITION_MAX_AGE]:
            del self.person_positions[tid]

        points = []
        published = []
        drop = []

        for gid, g in self.active_groups.items():
            if g["type"] == "queue":
                # No lidar hold-open for queues yet - see QUEUE HANDLING
                # in the module docstring. Expires on the same rule the
                # conversation path uses for "camera fresh".
                if now - g["last_seen"] > GROUP_TIMEOUT:
                    drop.append((gid, "queue: no update within GROUP_TIMEOUT"))
                    continue

                gap_points = self.make_queue_gap_points(
                    g["member_ids"], g.get("member_xy"))
                points.extend(gap_points)
                if gap_points:
                    published.append(("queue", gid, len(g["member_ids"])))
                continue

            # --- conversation path, unchanged ---
            camera_age = now - g["last_seen"]
            camera_fresh = camera_age <= GROUP_TIMEOUT

            if camera_fresh:
                ax_pos, bx_pos = g["member_a"], g["member_b"]

                # Re-bind anchors continuously while positions are
                # trustworthy. Previously anchors were only looked up
                # AFTER the camera went blind, by which point the
                # remembered position had drifted relative to the moving
                # lidar centroid - measured misses of 0.30/0.34/0.36m
                # against a 0.35m gate even when both tracks were
                # present, and 0.90m once a member's own track had gone.
                # Binding while fresh means the association is made
                # under good conditions and simply carried forward.
                a_id = self.find_anchor(*ax_pos)
                b_id = self.find_anchor(*bx_pos, exclude=a_id)
                if a_id is not None:
                    g["anchor_a"] = a_id
                if b_id is not None:
                    g["anchor_b"] = b_id
            else:
                # Camera has gone quiet - hold the zone open only while
                # BOTH members still have a lidar anchor, and move it to
                # follow those anchors so it stays correct if the pair
                # drifts. Either member losing its anchor ends the group.
                if camera_age > MAX_LIDAR_ONLY_HOLD:
                    drop.append((gid, "lidar-only hold expired"))
                    continue

                # Use the anchors learned while the camera was fresh.
                # Only fall back to a proximity search if one was never
                # bound - and never re-search for an anchor that has
                # simply disappeared, since the nearest remaining track
                # is most likely the OTHER member.
                lid_a = g.get("anchor_a")
                lid_b = g.get("anchor_b")

                if lid_a is None:
                    lid_a = self.find_anchor(*g["member_a"], exclude=lid_b)
                if lid_b is None:
                    lid_b = self.find_anchor(*g["member_b"], exclude=lid_a)

                if lid_a not in self.lidar_tracks:
                    drop.append((gid, f"anchor {lid_a} for member a gone"))
                    continue
                if lid_b not in self.lidar_tracks:
                    drop.append((gid, f"anchor {lid_b} for member b gone"))
                    continue
                if lid_a == lid_b:
                    drop.append((gid, "both members collapsed onto one anchor"))
                    continue

                ax_pos = self.lidar_tracks[lid_a][:2]
                bx_pos = self.lidar_tracks[lid_b][:2]

                # Positions are now lidar-derived, so re-check the
                # separation the camera path would have checked.
                if math.hypot(ax_pos[0] - bx_pos[0],
                              ax_pos[1] - bx_pos[1]) > CONV_MAX_DIST:
                    drop.append((gid, "members moved apart"))
                    continue

                if g["camera_fresh"]:
                    g["camera_fresh"] = False
                    self.get_logger().info(
                        f"Group {gid}: camera lost, holding zone on "
                        f"lidar anchors {lid_a}/{lid_b}")

                g["member_a"], g["member_b"] = ax_pos, bx_pos

            cx = (ax_pos[0] + bx_pos[0]) / 2.0
            cy = (ax_pos[1] + bx_pos[1]) / 2.0
            dx, dy = bx_pos[0] - ax_pos[0], bx_pos[1] - ax_pos[1]
            separation = math.hypot(dx, dy)

            a = separation / 2.0 - BODY_CLEARANCE
            if a < MIN_O_SPACE_HALF_LENGTH:
                continue

            points.extend(self.make_o_space_points(cx, cy, dx, dy, a))
            published.append(("conversation", gid, a, camera_fresh))

        for gid, reason in drop:
            del self.active_groups[gid]
            self.get_logger().info(f"Group {gid} cleared - {reason}")

        # --- robot keep-out ------------------------------------------
        try:
            tfm = self.tf_buffer.lookup_transform(
                self.frame_id, ROBOT_FRAME,
                rclpy.time.Time(), timeout=Duration(seconds=0.05))
            self.last_robot_xy = (tfm.transform.translation.x,
                                  tfm.transform.translation.y)
        except Exception:
            pass  # reuse last known pose

        if self.last_robot_xy is not None and points:
            rx, ry = self.last_robot_xy
            r2 = ROBOT_KEEPOUT_RADIUS ** 2
            before = len(points)
            points = [p for p in points
                      if (p[0] - rx) ** 2 + (p[1] - ry) ** 2 > r2]
            removed = before - len(points)
            if removed:
                self.get_logger().info(
                    f"Keep-out: removed {removed} point(s) within "
                    f"{ROBOT_KEEPOUT_RADIUS:.2f} m of robot")

        self.pub.publish(self.create_cloud(points))

        if published:
            parts_desc = []
            for entry in published:
                if entry[0] == "queue":
                    _, gid, n_members = entry
                    parts_desc.append(f"{gid}(queue,{n_members} members)")
                else:
                    _, gid, a, fresh = entry
                    parts_desc.append(
                        f"{gid}(len={2*a:.2f}m{'' if fresh else ',LIDAR-HELD'})")
            desc = ", ".join(parts_desc)
            self.get_logger().info(
                f"Published {len(published)} zone(s): {desc} "
                f"points={len(points)}")

    # -----------------------------------------------------------------
    def create_cloud(self, points):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.frame_id

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        cloud = PointCloud2()
        cloud.header = header
        cloud.height = 1
        cloud.width = len(points)
        cloud.fields = fields
        cloud.is_bigendian = False
        cloud.point_step = 12
        cloud.row_step = cloud.point_step * len(points)
        cloud.data = b"".join(struct.pack("fff", x, y, z) for x, y, z in points)
        cloud.is_dense = True
        return cloud

    def destroy_node(self):
        self.pub.publish(self.create_cloud([]))
        self.get_logger().info("Published empty cloud to clear costmap on shutdown")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SocialGroupCloudNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    node.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
