#!/usr/bin/env python3
"""
social_occupancy_grid_node_lidar.py

Simulation counterpart of the real-robot social_occupancy_grid_node
(humble_client/workspace/pytest/test.py). Same gradient-ellipse cost
grid design, wired to the plain (unprefixed) topic names used
throughout this Lidar/ sim pipeline:

  /predicted_person_positions (KF predictions, from human_kf_predictor_lidar.py)
  /map                        (map_server, latched)
  /social_cost_grid           (published here, consumed by nav2's social_layer)
"""
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header, String


class SocialOccupancyGridNode(Node):
    def __init__(self):
        super().__init__('social_occupancy_grid_node')

        self.kf_topic = '/predicted_person_positions'
        self.grid_topic = '/social_cost_grid'
        self.map_topic = '/map'

        # Cost Profile Parameters
        # THESIS FIX (robot stops instead of detouring): was semi_minor=1.10,
        # target_clearance=0.80 - the ramp from outer_rim_cost to
        # ellipse_peak_cost happened over just 0.3m (1.10-0.80), and with
        # trinary_costmap:false the StaticLayer interpolates peak_cost=92
        # to ~233/254 on the costmap - almost lethal. MPPI's rollouts had
        # no room to sample a graceful lateral swerve before hitting that
        # near-lethal wall, so it braked instead of steering around.
        #
        # THESIS FIX (ellipse filled the whole corridor): a first pass
        # widened semi_minor to 1.60 to fix the above, but this corridor
        # is only 2.0m wide wall-to-wall (+/-1.0m from centerline) - a
        # 1.60m LATERAL half-width blankets both walls at once, leaving
        # no gap for the controller to route through at all. semi_minor
        # is the sideways dimension and must stay well under the
        # corridor's half-width; semi_major (below, along the direction
        # of travel) is the one safe to extend for early warning, since
        # that's bounded by corridor LENGTH (15m), not width.
        self.semi_minor = 0.75             # Lateral half-width - must clear a corridor wall
        self.target_clearance = 0.55       # Desired clearance before the steep inner ramp
        self.semi_major_base = 2.00        # Longitudinal reach (safe to keep wide)
        self.vel_scale_factor = 1.2
        self.max_semi_major = 3.20

        # THESIS FIX (same detour issue): with trinary_costmap:false,
        # StaticLayer interpolates this 0-100 value to 0-254 on the
        # costmap - 92 mapped to ~233/254, indistinguishable from a real
        # lethal obstacle to MPPI's cost-based rollout scoring, so the
        # controller had no gradient left to prefer "close but survivable"
        # over "further and safer" - both looked equally catastrophic.
        # 70 -> ~178/254: still strongly discouraged, but leaves headroom
        # for the optimizer to actually rank nearby candidate paths by
        # cost instead of them all reading as equally forbidden.
        self.ellipse_peak_cost = 70        # Inner core cost
        self.outer_rim_cost = 25           # Maximum cost at the outer fringe

        # Stationary human settings
        # If speed == 0, give them an ellipse along corridor/yaw instead of a small dot
        self.stationary_speed_thresh = 0.05
        self.default_heading = math.radians(45.0)  # Angle of the corridor in map

        # Grid specifications (dynamically updated from /map)
        self.frame_id = 'map'
        self.resolution = 0.05
        self.grid_width = 300
        self.grid_height = 300
        self.origin_x = -10.0
        self.origin_y = -5.0
        self.map_initialized = False

        # QoS for latched map
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)
        self.sub_map = self.create_subscription(
            OccupancyGrid, self.map_topic, self.map_callback, map_qos)

        self.sub_kf = self.create_subscription(
            String, self.kf_topic, self.kf_callback, 10)

        grid_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)
        self.pub_grid = self.create_publisher(
            OccupancyGrid, self.grid_topic, grid_qos)

        self.active_tracks = {}
        self.track_timeout = 2.0

        self.timer = self.create_timer(0.1, self.publish_cost_grid)
        self.get_logger().info("Social Gradient Grid node initialized (sim).")

    def map_callback(self, msg: OccupancyGrid):
        if not self.map_initialized:
            self.resolution = msg.info.resolution
            self.grid_width = msg.info.width
            self.grid_height = msg.info.height
            self.origin_x = msg.info.origin.position.x
            self.origin_y = msg.info.origin.position.y
            self.frame_id = msg.header.frame_id
            self.map_initialized = True
            self.get_logger().info(
                f"Aligned social grid to map: {self.grid_width}x{self.grid_height} "
                f"origin=({self.origin_x:.2f}, {self.origin_y:.2f}) res={self.resolution:.3f}")

    def kf_callback(self, msg: String):
        parts = msg.data.split(',')
        if len(parts) < 8:
            return
        try:
            track_id = int(parts[0])
            x = float(parts[2])
            y = float(parts[3])
            vx = float(parts[4])
            vy = float(parts[5])
            speed = math.hypot(vx, vy)
            now = self.get_clock().now().nanoseconds * 1e-9

            self.active_tracks[track_id] = {
                'x': x, 'y': y, 'vx': vx, 'vy': vy,
                'speed': speed, 'last_seen': now
            }
        except ValueError:
            pass

    def _stamp_gradient_ellipse(self, grid, cx, cy, semi_a, semi_b, yaw, peak_cost):
        # THESIS FIX (edge clipping): the real-robot version bails out here
        # if the ellipse CENTER falls outside the grid, which also drops
        # any part of the ellipse that still overlaps the grid when the
        # person is just outside the mapped area (e.g. near a doorway).
        # inside_mask below already excludes out-of-grid cells correctly,
        # so no separate center-only bounds check is needed.
        y_cells, x_cells = np.ogrid[:self.grid_height, :self.grid_width]
        x_world = self.origin_x + x_cells * self.resolution
        y_world = self.origin_y + y_cells * self.resolution

        dx = x_world - cx
        dy = y_world - cy

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        x_rot = cos_yaw * dx + sin_yaw * dy
        y_rot = -sin_yaw * dx + cos_yaw * dy

        # Normalized ellipse distance: d_norm = 0.0 (center) to 1.0 (boundary)
        d_norm_sq = (x_rot / semi_a) ** 2 + (y_rot / semi_b) ** 2
        inside_mask = d_norm_sq <= 1.0

        if not np.any(inside_mask):
            return

        d_norm = np.sqrt(d_norm_sq[inside_mask])

        # Dual-zone profile:
        # - Outer region (d_norm > 0.72, corresponding to > 0.8m laterally):
        #   Tapers gently from ~15 down to 1. Robot can drive freely through it.
        # - Inner region (d_norm <= 0.72, inside 0.8m clearance):
        #   Ramps up steeply from 15 to 80 to strongly repel trajectory rollouts.
        clearance_norm = self.target_clearance / self.semi_minor  # ~0.727

        gradient_cost = np.zeros_like(d_norm)

        # Inner mask (d <= clearance_norm)
        inner = d_norm <= clearance_norm
        # Outer mask (clearance_norm < d <= 1.0)
        outer = ~inner

        # Inner ramp: 80 -> 15
        inner_progress = d_norm[inner] / clearance_norm
        gradient_cost[inner] = self.outer_rim_cost + (peak_cost - self.outer_rim_cost) * (1.0 - inner_progress)**2

        # Outer rim taper: 15 -> 0 (cubic dropoff, very soft)
        outer_progress = (d_norm[outer] - clearance_norm) / (1.0 - clearance_norm)
        gradient_cost[outer] = self.outer_rim_cost * np.power(1.0 - outer_progress, 2.0)

        gradient_cost = np.clip(gradient_cost, 0, peak_cost).astype(np.int8)
        grid[inside_mask] = np.maximum(grid[inside_mask], gradient_cost)

    def publish_cost_grid(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        self.active_tracks = {
            tid: trk for tid, trk in self.active_tracks.items()
            if now - trk['last_seen'] <= self.track_timeout
        }

        grid = np.zeros((self.grid_height, self.grid_width), dtype=np.int8)

        for trk in self.active_tracks.values():
            cx, cy = trk['x'], trk['y']
            speed = trk['speed']

            if speed >= self.stationary_speed_thresh:
                yaw = math.atan2(trk['vy'], trk['vx'])
                semi_a = min(self.max_semi_major, self.semi_major_base + speed * self.vel_scale_factor)
                proj_cx = cx + 0.3 * trk['vx']
                proj_cy = cy + 0.3 * trk['vy']
            else:
                # When stationary, orient the ellipse along the corridor diagonal (~45 deg)
                yaw = self.default_heading
                semi_a = self.semi_major_base
                proj_cx, proj_cy = cx, cy

            self._stamp_gradient_ellipse(
                grid, proj_cx, proj_cy,
                semi_a=semi_a,
                semi_b=self.semi_minor,
                yaw=yaw,
                peak_cost=self.ellipse_peak_cost
            )

        msg = OccupancyGrid()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.info.resolution = self.resolution
        msg.info.width = self.grid_width
        msg.info.height = self.grid_height
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.flatten().tolist()
        self.pub_grid.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SocialOccupancyGridNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
