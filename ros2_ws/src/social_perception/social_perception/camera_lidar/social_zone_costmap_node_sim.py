#!/usr/bin/env python3
"""
social_zone_costmap_node.py

Publishes conversation/side-by-side social zones as a GRADED-COST
nav_msgs/OccupancyGrid for the global costmap.

=======================================================================
WHY THIS EXISTS (ablation E -> F)
=======================================================================
social_group_cloud_node.py rasterised the o-space into a PointCloud2
consumed by a NonPersistentVoxelLayer. Every marked cell became
LETHAL_OBSTACLE (254), so the o-space was not "expensive", it was
FORBIDDEN. In a corridor too narrow to route around the pair that left
no valid plan and the robot halted.

The distinction this node encodes:

  PHYSICAL occupancy - a person's body, their predicted path - is
    lethal. Occupying it is a collision. Nothing to weigh. That stays
    with predicted_person_cloud_node_lidar.py, untouched.

  SOCIAL occupancy - the space a group has claimed between them - is
    GRADED. Crossing it is a rudeness, not a collision. A rudeness has
    a price, and a price can be outbid by a corridor with no
    alternative. A lethal encoding cannot express that; this can.

=======================================================================
WHAT THIS NODE DOES *NOT* PAINT
=======================================================================
Bodies. Group members keep their lethal disks from
predicted_person_cloud_node_lidar.py - the `!= "queue"` guard in that
node's group_callback stays exactly as it is, so conversation members
are NOT deferred to this layer.

That is deliberate and was learned the hard way: deferring conversation
bodies to the group layer left a pair represented by a small patch of
gap and nothing else, and the global plan did not reroute at all
because there was almost nothing to route around. This node marks only
the space BETWEEN and AROUND people, never the people.

=======================================================================
THE COST LADDER
=======================================================================
Flipped by field 10 of /social_groups - the effective buffer that
social_group_detector_node.py used after its Option 1 costmap clearance
probe. That probe is the single source of truth for "is there room to
go around"; no second threshold is invented here.

  WIDE corridor (buffer == ZONE_BUFFER, no shrink):
      o-space  90  (~229)      flanks  unpainted
      -> going around is cheap. Robot goes around. Correct default.

  NARROW corridor (buffer shrunk by the probe):
      o-space  35  (~89)       flanks  60  (~152)
      -> the gap between them is now the cheapest FINITE route.
         Robot passes through, deliberately and (with
         social_zone_speed_limiter.py) slowly.

Ceiling is 90 so nothing reaches the inscribed (253) or lethal (254)
band. A zone is ALWAYS crossable; the planner chooses on cost, never on
legality. Overlapping regions keep the HIGHER value so a flank lobe
cannot dilute an o-space it touches.

On the narrow case: where the corridor is truly tight, the flanks are
already walls at 254 from the static and obstacle layers and
FLANK_COST_NARROW is redundant. It earns its place in the MIDDLING
case - passable but tight - where a shoulder-squeeze would otherwise be
free and would beat a discouraged o-space.

=======================================================================
STATIONARY GROUPS ONLY
=======================================================================
The detector gates on both members under 0.30 m/s for 0.75 s, so a
walking pair never becomes a group and never reaches this node - they
fall through to the individual swept ellipses instead. That gate is
also what stops two strangers passing in opposite directions being
misread as a conversation.

This is why PUBLISH_RATE_HZ = 2.0 is safe: a zone that takes ~1 s to
form does not need 10 Hz refresh. If walking-group support is ever
added, this rate assumption breaks first.

=======================================================================
GEOMETRY MIRRORING - THE RESIZE GUARD
=======================================================================
StaticLayer calls resizeMap() on the master grid whenever the costmap
is not rolling. The global costmap is NOT rolling. So this node copies
resolution / width / height / origin VERBATIM from /turtlebot4/map and
paints into a zero grid of that exact shape, making the resize a no-op.

If this node ever published a grid of different geometry, StaticLayer
would shrink the entire global costmap to the size of one conversation
zone. Nothing is published until the map arrives - guessing is worse
than silence.

=======================================================================
COSTMAP PLUGIN (see ablation F yaml)
=======================================================================
      social_zone_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_topic: /social_zone_map
        trinary_costmap: false     # REQUIRED: keeps 0-100 graded
        use_maximum: true          # REQUIRED: do not erase other layers
        lethal_cost_threshold: 100
        subscribe_to_updates: false
        map_subscribe_transient_local: false

trinary_costmap: false is the whole point. Left at its default (true),
every value above the threshold collapses to 254 and this is ablation E
again with extra steps.

UNTESTED on hardware. StaticLayer is normally fed a latched SLAM map,
not a 2 Hz stream. If it misbehaves the honest fallback is a custom C++
layer.
"""

import math

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid


# =======================================================================
# Cost ladder. Occupancy 0-100; StaticLayer with trinary_costmap:false
# scales to roughly value * 2.54 in costmap units. Keep everything <= 90.
# =======================================================================
# THESIS CHANGE (ablation F, narrow-corridor pass-through): the two
# members' bodies are now painted HERE, as a small lethal core, instead
# of being marked by predicted_person_cloud_node_lidar.py. Reason: that
# node paints a directional ellipse whose inflation, at a 1.5 m pair
# separation, merged across the gap and made the o-space impassable -
# measured cost 100 at the pair midpoint. With no valid plan through a
# narrow corridor the whole graded ladder was unreachable. A 0.25 m core
# still makes driving INTO a person illegal; the space between them is
# priced, not forbidden.
BODY_CORE_RADIUS_M = 0.25   # m; lethal core per conversation member
BODY_CORE_COST = 100        # lethal - never crossable, unlike the zone

# THESIS ADDITION (personal-space halo). Costmap FILTERS receive no
# inflation from the inflation_layer - the layer runs inside the layer
# stack, filters are applied above it. So the lethal cores above get NO
# berth at all, and the robot skimmed a stationary person at 0.622 m
# (measured, bag conv_F_20260920_214023, t=29.2 s) while still
# correctly routing around the o-space. Proxemics puts that inside
# personal distance.
#
# The berth is therefore painted explicitly, as a graded ring around
# each member out to PERSONAL_SPACE_RADIUS_M. Graded, not lethal: a
# corridor narrower than the halo must stay passable, which is the
# whole argument of ablation F. The cost sits just below the o-space so
# that when both are crossable the robot still prefers to pass a person
# on the outside rather than walk through their conversation.
PERSONAL_SPACE_RADIUS_M = 0.8   # m; target minimum clearance to a person
PERSONAL_SPACE_COST = 80        # ~204 - below O_SPACE_COST_WIDE (90)

O_SPACE_COST_WIDE = 90      # ~229 - strongly avoided, still legal
O_SPACE_COST_NARROW = 35    # ~89  - cheapest finite route when boxed in
FLANK_COST_NARROW = 60      # ~152 - MUST sit above O_SPACE_COST_NARROW

# MUST match ZONE_BUFFER in social_group_detector_node.py. Used only to
# detect that the detector shrank it - never for geometry.
NOMINAL_ZONE_BUFFER = 0.4
SHRINK_EPS = 0.01

# =======================================================================
# Zone geometry. Mirrors social_group_cloud_node.py's constants so the
# painted region matches what ablation E marked, minus the lethality.
# =======================================================================
BODY_CLEARANCE = 0.25          # back off from each person; paint the GAP
O_SPACE_HALF_WIDTH = 0.35
MIN_O_SPACE_HALF_LENGTH = 0.10

# Flank lobes sit BEYOND each person along the pair's axis - that is the
# "going around" route for a pair standing across a corridor.
FLANK_OFFSET = 0.45            # m from person to lobe centre, along axis
FLANK_HALF_LENGTH = 0.40       # along the axis
FLANK_HALF_WIDTH = 0.45        # across it

GROUP_TIMEOUT = 4.0            # s; matches social_group_cloud_node.py
PUBLISH_RATE_HZ = 2.0          # see STATIONARY GROUPS ONLY above

# Margin around the pair. FIXED, deliberately not the published buffer:
# the buffer decides COST (narrow vs wide), not size. Letting it shrink
# the zone too would make a narrow-corridor zone nearly vanish, which
# defeats the whole point - the robot should pass through it slowly, not
# find nothing there.
ZONE_MARGIN = 0.4

MAP_TOPIC = "/map"   # SIM: root namespace
GROUP_TOPIC = "/social_groups"
OUTPUT_TOPIC = "/social_zone_map"

# =======================================================================
# THESIS CHANGE (ablation F integration): KEEPOUT FILTER MASK
#
# This grid is a costmap FILTER MASK, consumed by
# nav2_costmap_2d::KeepoutFilter, not by a costmap layer.
#
# Two StaticLayer routes were measured and both failed, for the same
# reason:
#   1. A second StaticLayer alongside the map's own: StaticLayer
#      OVERWRITES the master costmap rather than merging, and Jazzy's
#      StaticLayer has no use_maximum (runtime: "Parameter not set").
#      Zone layer last wiped the costmap to uniform 0; zone layer first
#      left 32 of ~550 painted cells and cost 0 at the pair midpoint.
#   2. Baking the map's walls into this grid and pointing the single
#      static_layer at it: the walls arrived, but the graded values did
#      not. Jazzy's StaticLayer declares only enabled,
#      footprint_clearing_enabled, map_subscribe_transient_local,
#      map_topic, plugin, subscribe_to_updates, transform_tolerance -
#      there is NO trinary_costmap, so every value below lethal is read
#      as free space. Measured: this grid held 559 cells at 90 and the
#      global costmap read 0 at the same coordinate.
#
# KeepoutFilter is the mechanism that does work. Costmap filters are
# applied ON TOP of the combined layers, so no layer can overwrite
# them, and with base 0.0 / multiplier 1.0 the mask's OccupancyGrid
# values pass through one-to-one as cost - a range of costs, not just
# binary occupation. So this node publishes ZONES ONLY: the map's own
# walls stay with the map's own static_layer where they belong.
# =======================================================================
MAP_UNKNOWN_COST = -1    # preserved so track_unknown_space still works


class SocialZoneCostmapNode(Node):
    def __init__(self):
        super().__init__("social_zone_costmap_node")

        self.declare_parameter("map_topic", MAP_TOPIC)
        self.declare_parameter("input_topic", GROUP_TOPIC)
        self.declare_parameter("output_topic", OUTPUT_TOPIC)
        self.declare_parameter("frame_id", "map")

        self.map_topic = self.get_parameter("map_topic").value
        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.frame_id = self.get_parameter("frame_id").value

        self.map_info = None       # geometry to mirror; None until /map lands
        self.map_base = None       # map occupancy, used ONLY to keep unknown
                                   # cells unknown - never painted into the mask
        self.zones = {}            # group_id -> dict
        self.published_empty = False

        # The map is latched (TRANSIENT_LOCAL) and sent once. Matching
        # that durability is what lets a late-starting node still receive
        # it - with VOLATILE we would wait forever for a message that was
        # already sent.
        map_qos = QoSProfile(depth=1)
        map_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        map_qos.reliability = QoSReliabilityPolicy.RELIABLE

        self.create_subscription(
            OccupancyGrid, self.map_topic, self.map_callback, map_qos)
        self.create_subscription(
            String, self.input_topic, self.group_callback, 10)
        # TRANSIENT_LOCAL so a late-subscribing StaticLayer still receives
        # the most recent grid. StaticLayer BLOCKS the entire costmap
        # update until it has a map - a silent publisher here stops global
        # planning outright, it does not merely contribute no cost.
        pub_qos = QoSProfile(depth=1)
        pub_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        pub_qos.reliability = QoSReliabilityPolicy.RELIABLE
        self.pub = self.create_publisher(
            OccupancyGrid, self.output_topic, pub_qos)

        self.create_timer(1.0 / PUBLISH_RATE_HZ, self.publish_grid)

        self.get_logger().info("Social zone costmap node started")
        self.get_logger().info(
            f"Mirroring geometry from {self.map_topic} -> "
            f"{self.output_topic} (KeepoutFilter mask)")
        self.get_logger().info(
            f"Personal space: {PERSONAL_SPACE_RADIUS_M:.2f} m halo at "
            f"cost {PERSONAL_SPACE_COST}, {BODY_CORE_RADIUS_M:.2f} m "
            f"lethal core")
        self.get_logger().info(
            f"Cost ladder | wide: o-space {O_SPACE_COST_WIDE} | "
            f"narrow: o-space {O_SPACE_COST_NARROW}, "
            f"flanks {FLANK_COST_NARROW} | rate {PUBLISH_RATE_HZ:.1f} Hz")
        self.get_logger().warn(
            "Waiting for map - nothing published until geometry is known")

    def get_ros_time_seconds(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # -----------------------------------------------------------------
    def map_callback(self, msg):
        info = msg.info
        if self.map_info is not None:
            same = (info.width == self.map_info.width
                    and info.height == self.map_info.height
                    and abs(info.resolution - self.map_info.resolution) < 1e-9)
            if same:
                return
            self.get_logger().warn(
                "Map geometry CHANGED - re-mirroring. Any grid published "
                "with the old geometry would have resized the global "
                "costmap.")

        self.map_info = info
        # Keep the map's own occupancy as the base layer of every grid we
        # publish. np.int8 matches the OccupancyGrid wire type, and -1
        # (unknown) survives the copy unchanged.
        self.map_base = np.array(msg.data, dtype=np.int8).reshape(
            info.height, info.width)
        cells = info.width * info.height
        self.get_logger().info(
            f"Map geometry: {info.width}x{info.height} @ "
            f"{info.resolution:.3f} m ({cells} cells), origin "
            f"({info.origin.position.x:.2f}, {info.origin.position.y:.2f})")
        if cells > 4_000_000:
            self.get_logger().warn(
                f"{cells} cells at {PUBLISH_RATE_HZ:.1f} Hz may not keep "
                f"up - watch for publish lag in the throttled log below")

    # -----------------------------------------------------------------
    def group_callback(self, msg):
        parts = msg.data.split(",")
        if len(parts) < 11:
            # Field 10 (effective buffer) is the entire basis for the
            # narrow/wide flip. Without it there is no way to tell a
            # shrunk zone from a nominal one, so refuse rather than guess.
            self.get_logger().warn(
                "Ignoring /social_groups msg without buffer field - is "
                "social_group_detector_node.py the patched version?",
                throttle_duration_sec=10.0)
            return

        try:
            group_id = parts[0].strip()
            group_type = parts[1].strip()
            buffer = float(parts[10])
            # Field 9 is the members' own map positions,
            # "x;y|x;y" - the detector's ground truth. Using them
            # directly sidesteps the axis inconsistency entirely: the
            # detector points `axis` THROUGH the pair for a facing
            # group but PERPENDICULAR to it for side-by-side, so any
            # geometry derived from axis is right for one type and
            # wrong for the other. The connecting line between two
            # people is the same geometric thing either way.
            mem = parts[9].split("|")
            if len(mem) != 2:
                return
            ax_, ay_ = (float(v) for v in mem[0].split(";"))
            bx_, by_ = (float(v) for v in mem[1].split(";"))
        except (ValueError, IndexError):
            return

        if group_type not in ("conversation", "side_by_side"):
            return

        dx, dy = bx_ - ax_, by_ - ay_
        separation = math.hypot(dx, dy)
        if separation < 1e-6:
            return
        # Unit vector along the line joining the two people.
        ax, ay = dx / separation, dy / separation

        self.zones[group_id] = {
            "cx": (ax_ + bx_) / 2.0, "cy": (ay_ + by_) / 2.0,
            "ax": ax, "ay": ay,
            "separation": separation,
            "member_a": (ax_, ay_),
            "member_b": (bx_, by_),
            "buffer": buffer,
            "narrow": buffer < NOMINAL_ZONE_BUFFER - SHRINK_EPS,
            "type": group_type,
            "last_seen": self.get_ros_time_seconds(),
        }

    # -----------------------------------------------------------------
    def _regions(self, z):
        """Yield (cx, cy, half_length, half_width, cost) ellipses for one
        group, in world coordinates, oriented along the line joining the
        two people.

        FUSED ZONE (18 Sep). One shape for both group types, covering
        the pair and the space between them, rather than a type-specific
        F-formation o-space.

        The reason is measurement, not convenience: "side_by_side" is
        also what the classifier emits when its left/right sign test
        comes back zero - which happens for genuine side-by-side pairs
        AND for facing pairs viewed along the camera axis. So the label
        is partly noise, and branching the geometry on it would put
        zones in wrong places for the wrong reason. The classifier still
        runs and its verdict is still carried in /social_groups, so
        classification accuracy can be reported as a separate result
        without the navigation depending on it.

        What this gives up: the zone no longer models an F-formation
        o-space. It is a proximity region around a pair. State that
        plainly in the writeup rather than implying Kendon's model is
        being used.
        """
        half_length = z["separation"] / 2.0 + ZONE_MARGIN
        half_width = ZONE_MARGIN
        cost = O_SPACE_COST_NARROW if z["narrow"] else O_SPACE_COST_WIDE
        yield (z["cx"], z["cy"], half_length, half_width, cost)

        # THESIS FIX (23 Sep, wide-case bodies): halo + lethal core are
        # painted in BOTH branches. predicted_person_cloud_node_lidar.py
        # defers conversation members, so nothing else paints them; they
        # used to sit below the narrow-only return, which left a wide-case
        # pair with no body cost and no personal space (probe at spawn:
        # members 89, 0.6 m out 0). Halo first, core second: publish_grid()
        # takes the per-cell maximum, so the core wins inside its halo.
        for (px, py) in (z["member_a"], z["member_b"]):
            yield (px, py, PERSONAL_SPACE_RADIUS_M, PERSONAL_SPACE_RADIUS_M,
                   PERSONAL_SPACE_COST)

        # Lethal body cores, painted as circles (hl == hw).
        for (px, py) in (z["member_a"], z["member_b"]):
            yield (px, py, BODY_CORE_RADIUS_M, BODY_CORE_RADIUS_M,
                   BODY_CORE_COST)

        if not z["narrow"]:
            return  # wide corridor: flanks stay free, going around is cheap

        # Flank lobes beyond each person, along the connecting line -
        # that is the "going around" route for a pair standing across a
        # corridor.
        ax, ay = z["ax"], z["ay"]
        for (px, py), sign in ((z["member_a"], -1.0), (z["member_b"], 1.0)):
            yield (px + sign * FLANK_OFFSET * ax,
                   py + sign * FLANK_OFFSET * ay,
                   FLANK_HALF_LENGTH, FLANK_HALF_WIDTH, FLANK_COST_NARROW)

    # -----------------------------------------------------------------
    def publish_grid(self):
        if self.map_info is None:
            return  # geometry unknown - see RESIZE GUARD in the header

        now = self.get_ros_time_seconds()
        for gid in [g for g, z in self.zones.items()
                    if now - z["last_seen"] > GROUP_TIMEOUT]:
            del self.zones[gid]

        info = self.map_info
        res = info.resolution
        ox = info.origin.position.x
        oy = info.origin.position.y
        w, h = info.width, info.height

        if not self.zones:
            # Publish one empty grid to clear the layer, then go quiet.
            # Restreaming an all-zero map costs the same as a full one.
            if not self.published_empty:
                # A mask of all zeros adds no cost anywhere, which is
                # exactly "no social zones". The map's walls are not this
                # grid's business - static_layer still owns them.
                self._publish(np.zeros((h, w), dtype=np.int8))
                self.published_empty = True
                self.get_logger().info("No groups - published empty mask")
            return

        self.published_empty = False
        grid = np.zeros((h, w), dtype=np.int8)

        for z in self.zones.values():
            ax, ay = z["ax"], z["ay"]
            for (rcx, rcy, hl, hw, cost) in self._regions(z):
                # Paint only the bounding box of this ellipse, not the
                # whole map. The full grid is allocated once above; this
                # keeps per-region work proportional to the zone, not the
                # map.
                reach = math.hypot(hl, hw)
                i0 = max(0, int((rcx - reach - ox) / res))
                i1 = min(w - 1, int((rcx + reach - ox) / res))
                j0 = max(0, int((rcy - reach - oy) / res))
                j1 = min(h - 1, int((rcy + reach - oy) / res))
                if i1 < i0 or j1 < j0:
                    continue  # zone lies outside the map

                ii = np.arange(i0, i1 + 1)
                jj = np.arange(j0, j1 + 1)
                wx = ox + (ii + 0.5) * res
                wy = oy + (jj + 0.5) * res
                dx = wx[None, :] - rcx
                dy = wy[:, None] - rcy

                u = dx * ax + dy * ay          # along the pair's axis
                v = -dx * ay + dy * ax         # across it
                mask = (u / hl) ** 2 + (v / hw) ** 2 <= 1.0

                sub = grid[j0:j1 + 1, i0:i1 + 1]
                # Higher cost wins, so a flank lobe never dilutes an
                # o-space it overlaps. Walls (MAP_WALL_COST) therefore
                # survive automatically - no zone cost reaches 100.
                # Unknown cells (-1) are excluded: raising them to a zone
                # cost would silently declare unmapped space traversable.
                painted = np.where(mask, cost, 0).astype(np.int8)
                np.maximum(sub, np.where(sub >= 0, painted, 0).astype(np.int8),
                           out=sub)
                sub[self.map_base[j0:j1 + 1, i0:i1 + 1] < 0] = MAP_UNKNOWN_COST

        self._publish(grid)

        narrow = sum(1 for z in self.zones.values() if z["narrow"])
        self.get_logger().info(
            f"{len(self.zones)} zone(s), {narrow} narrow",
            throttle_duration_sec=5.0)

    # -----------------------------------------------------------------
    def _publish(self, grid):
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.info = self.map_info          # mirrored verbatim - resize guard
        msg.data = grid.ravel().tolist()
        self.pub.publish(msg)

    def destroy_node(self):
        if self.map_info is not None:
            self._publish(np.zeros(
                (self.map_info.height, self.map_info.width), dtype=np.int8))
            self.get_logger().info("Published empty grid to clear costmap")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SocialZoneCostmapNode()
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
