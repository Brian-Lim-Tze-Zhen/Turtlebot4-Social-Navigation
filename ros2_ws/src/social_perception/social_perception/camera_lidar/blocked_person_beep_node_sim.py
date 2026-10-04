#!/usr/bin/env python3
"""
blocked_person_beep_node_sim.py - SIMULATION (Jazzy, gz sim).

Beeps when the robot has a goal, is making (almost) no headway, and a person
is in front of it.

It does not cancel or resend anything: Nav2 keeps the goal and drives on by
itself when the person has moved. It beeps within a few seconds instead of
after the goal has ABORTED, which is when beep_retry_node_sim.py acts. Run
both: this one for the prompt beep, that one to resend the goal if Nav2 gives
up before the person moves.

hold:=true (needs a config with the collision monitor's "BlockedHold" zone,
e.g. social_nav2_headon_F_hwreq_block_blockedhold_sim.yaml): once blocked, the
robot is also held at zero velocity until the person test (3) has been false
for its memory time. Without it the robot stays in place but keeps turning
and creeping while Nav2 looks for a way round.

Beep when ALL of these hold:
  1. a NavigateToPose goal is executing
  2. the robot's position has stayed within STILL_DIST_M over the last STILL_S
  3. "person in front", same two tests as beep_retry_node_sim.py:
       a) a person from /person_positions_fused within PERSON_BLOCK_M of the
          robot in the last PERSON_MEMORY_S, or
       b) a LiDAR point NOT on the static map within LIDAR_BLOCK_M,
          +/- FRONT_HALF_DEG ahead, in the last LIDAR_MEMORY_S
The beep repeats every BEEP_PERIOD_S while they hold.

Publishes:
  /cmd_audio             irobot_create_msgs/AudioNoteVector
  /blocked_person_event  std_msgs/String  "blocked,<sim t>,<why>" /
                                          "clear,<sim t>,<blocked for s>"
  /blocked_hold_polygon  geometry_msgs/PolygonStamped  (hold:=true only)

Run:
  python3 blocked_person_beep_node_sim.py --ros-args -p use_sim_time:=true
"""
import collections
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PolygonStamped, Point32
from nav_msgs.msg import Odometry, OccupancyGrid
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from action_msgs.msg import GoalStatusArray, GoalStatus
from irobot_create_msgs.msg import AudioNoteVector, AudioNote
from builtin_interfaces.msg import Duration as DurationMsg
import tf2_ros

NAV_STATUS_TOPIC = "/navigate_to_pose/_action/status"
ODOM_TOPIC = "/odom"
AUDIO_TOPIC = "/cmd_audio"
SCAN_TOPIC = "/scan"
MAP_TOPIC = "/map"
PERSON_TOPIC = "/person_positions_fused"
EVENT_TOPIC = "/blocked_person_event"
HOLD_TOPIC = "/blocked_hold_polygon"
HOLD_HALF_M = 1.6             # m; half side of the stop square (covers PERSON_BLOCK_M)

# "Not moving" is judged on headway, not on standing dead still. In front of
# two people closing the corridor the robot stood for ~4 s and then crept
# sideways at about 0.1 m/s looking for a way round (pilot
# headon_blockedbeep_pair_pilot1); with a 0.05 m / 2 s test the beeping
# stopped there although it was still blocked. 0.5 m in 4 s is a mean of
# 0.125 m/s: below the 0.15 m/s occlusion slow-down and the 0.31 m/s cruise.
STILL_S = 4.0                 # s window
STILL_DIST_M = 0.5            # m; position spread over STILL_S below this = no headway
BEEP_PERIOD_S = 3.0
# Person / unmapped-LiDAR tests: values of beep_retry_node_sim.py.
PERSON_BLOCK_M = 1.5
PERSON_MEMORY_S = 5.0
LIDAR_BLOCK_M = 1.0
LIDAR_MEMORY_S = 2.0
FRONT_HALF_DEG = 30.0
MAP_MARGIN_CELLS = 3

MAP_FRAME = "map"
BASE_FRAME = "base_link"
SCAN_FRAME = "rplidar_link"   # SIM: scan header frame is not in TF; this frame is


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class BlockedPersonBeepNode(Node):
    def __init__(self):
        super().__init__("blocked_person_beep_node")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST)
        sensor = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                            history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(GoalStatusArray, NAV_STATUS_TOPIC, self.on_status, latched)
        self.create_subscription(Odometry, ODOM_TOPIC, self.on_odom, sensor)
        self.create_subscription(String, PERSON_TOPIC, self.on_person, 10)
        self.create_subscription(LaserScan, SCAN_TOPIC, self.on_scan, sensor)
        self.create_subscription(OccupancyGrid, MAP_TOPIC, self.on_map, latched)
        self.audio_pub = self.create_publisher(AudioNoteVector, AUDIO_TOPIC, 10)
        self.event_pub = self.create_publisher(String, EVENT_TOPIC, 10)
        self.declare_parameter("hold", False)
        self.hold = bool(self.get_parameter("hold").value)
        self.hold_pub = (self.create_publisher(PolygonStamped, HOLD_TOPIC, 10)
                         if self.hold else None)

        self.goal_active = False
        self.odom = collections.deque()   # (sim t, x, y) over the last STILL_S
        self.last_person_block = -1e9
        self.last_lidar_block = -1e9
        self.grid = None
        self.mount_yaw = None
        self.blocked_since = None
        self.last_beep = None
        self.create_timer(0.2, self.tick)
        self.get_logger().info(
            f"blocked_person_beep_node: still = < {STILL_DIST_M} m in {STILL_S:.0f} s, "
            f"person {PERSON_BLOCK_M} m / {PERSON_MEMORY_S:.0f} s, "
            f"lidar unmapped {LIDAR_BLOCK_M} m +/-{FRONT_HALF_DEG:.0f} deg / {LIDAR_MEMORY_S:.0f} s, "
            f"beep every {BEEP_PERIOD_S:.0f} s, hold {'on' if self.hold else 'off'}")

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------ inputs
    def on_status(self, msg):
        if not msg.status_list:
            return
        newest = max(msg.status_list,
                     key=lambda s: (s.goal_info.stamp.sec, s.goal_info.stamp.nanosec))
        self.goal_active = newest.status in (GoalStatus.STATUS_ACCEPTED,
                                             GoalStatus.STATUS_EXECUTING)

    def on_odom(self, msg):
        now = self.now_s()
        p = msg.pose.pose.position
        self.odom.append((now, p.x, p.y))
        while self.odom and now - self.odom[0][0] > STILL_S + 0.5:
            self.odom.popleft()

    def on_map(self, msg):
        self.grid = msg

    def robot_pose(self):
        try:
            tr = self.tf_buffer.lookup_transform(MAP_FRAME, BASE_FRAME, rclpy.time.Time())
        except Exception:
            return None
        t = tr.transform
        return t.translation.x, t.translation.y, yaw_of(t.rotation)

    def on_person(self, msg):
        f = msg.data.split(",")
        try:
            px, py = float(f[2]), float(f[3])
        except (ValueError, IndexError):
            return
        rp = self.robot_pose()
        if rp is None:
            return
        if math.hypot(px - rp[0], py - rp[1]) <= PERSON_BLOCK_M:
            self.last_person_block = self.now_s()

    def on_map_cell_occupied(self, x, y):
        g = self.grid
        res = g.info.resolution
        cx = int((x - g.info.origin.position.x) / res)
        cy = int((y - g.info.origin.position.y) / res)
        W, H = g.info.width, g.info.height
        for dy in range(-MAP_MARGIN_CELLS, MAP_MARGIN_CELLS + 1):
            for dx in range(-MAP_MARGIN_CELLS, MAP_MARGIN_CELLS + 1):
                X, Y = cx + dx, cy + dy
                if 0 <= X < W and 0 <= Y < H and g.data[Y * W + X] >= 50:
                    return True
        return False

    def on_scan(self, msg):
        if self.grid is None:
            return
        if self.mount_yaw is None:
            try:
                tr = self.tf_buffer.lookup_transform(BASE_FRAME, SCAN_FRAME, rclpy.time.Time())
                self.mount_yaw = yaw_of(tr.transform.rotation)
            except Exception:
                return
        try:
            tr = self.tf_buffer.lookup_transform(MAP_FRAME, SCAN_FRAME, rclpy.time.Time())
        except Exception:
            return
        lx, ly = tr.transform.translation.x, tr.transform.translation.y
        lyaw = yaw_of(tr.transform.rotation)
        a = msg.angle_min
        for r in msg.ranges:
            if 0.15 < r <= LIDAR_BLOCK_M and not math.isinf(r) and not math.isnan(r):
                bearing = (math.degrees(a + self.mount_yaw) + 180.0) % 360.0 - 180.0
                if abs(bearing) <= FRONT_HALF_DEG:
                    x = lx + r * math.cos(lyaw + a)
                    y = ly + r * math.sin(lyaw + a)
                    if not self.on_map_cell_occupied(x, y):
                        self.last_lidar_block = self.now_s()
                        return
            a += msg.angle_increment

    # ------------------------------------------------------------ decision
    def is_still(self, now):
        if not self.odom or now - self.odom[0][0] < STILL_S:
            return False              # not enough history yet
        if now - self.odom[-1][0] > 1.0:
            return False              # odom is stale: unknown, do not beep
        xs = [o[1] for o in self.odom if now - o[0] <= STILL_S]
        ys = [o[2] for o in self.odom if now - o[0] <= STILL_S]
        return math.hypot(max(xs) - min(xs), max(ys) - min(ys)) < STILL_DIST_M

    def tick(self):
        now = self.now_s()
        person = now - self.last_person_block <= PERSON_MEMORY_S
        lidar = now - self.last_lidar_block <= LIDAR_MEMORY_S
        if self.hold and self.blocked_since is not None:
            # Held: the robot cannot move, so headway says nothing. Stay
            # blocked for as long as a person is still in front.
            blocked = self.goal_active and (person or lidar)
        else:
            blocked = self.goal_active and self.is_still(now) and (person or lidar)
        if self.hold:
            self.publish_hold(blocked)
        if blocked:
            if self.blocked_since is None:
                self.blocked_since = now
                self.last_beep = None
                why = "+".join(w for w, on in (("person", person), ("unmapped lidar", lidar)) if on)
                self.event_pub.publish(String(data=f"blocked,{now:.2f},{why}"))
                self.get_logger().info(f"Blocked by a person ({why}) - beeping.")
            if self.last_beep is None or now - self.last_beep >= BEEP_PERIOD_S:
                self.beep()
                self.last_beep = now
        elif self.blocked_since is not None:
            dur = now - self.blocked_since
            self.event_pub.publish(String(data=f"clear,{now:.2f},{dur:.2f}"))
            self.get_logger().info(f"No longer blocked after {dur:.1f} s.")
            self.blocked_since = None

    def publish_hold(self, on):
        # The collision monitor stops the robot while scan points are inside
        # this polygon: a square around the robot when holding, a sliver far
        # away (never any points in it) otherwise.
        msg = PolygonStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = BASE_FRAME
        h = HOLD_HALF_M
        corners = ([(h, h), (h, -h), (-h, -h), (-h, h)] if on
                   else [(50.0, 50.0), (50.1, 50.0), (50.0, 50.1)])
        msg.polygon.points = [Point32(x=float(x), y=float(y), z=0.0) for x, y in corners]
        self.hold_pub.publish(msg)

    def beep(self):
        # Same triple beep as beep_retry_node_sim.py's "blocked" signal.
        msg = AudioNoteVector()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.append = False
        for freq, dur in ((880, 0.2), (1100, 0.2), (880, 0.2)):
            n = AudioNote()
            n.frequency = freq
            n.max_runtime = DurationMsg(sec=0, nanosec=int(dur * 1e9))
            msg.notes.append(n)
        self.audio_pub.publish(msg)


def main():
    rclpy.init()
    node = BlockedPersonBeepNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    except Exception:
        # Teardown (SIGTERM) can invalidate the context mid-spin.
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
