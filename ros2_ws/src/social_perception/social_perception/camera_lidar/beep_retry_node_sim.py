#!/usr/bin/env python3
"""
beep_retry_node_sim.py - SIMULATION version (Jazzy, gz sim).

SIM CHANGES vs hardware:
  - no /turtlebot4 namespace (NS = "")
  - /cmd_vel is TwistStamped in Jazzy sim (hardware Humble: Twist)
  - LiDAR TF frame fixed to "rplidar_link" (scan header frame is not in TF)
  - all timing on the node clock (sim time), not self.now_s()

Original: blocked-goal handling for the real TurtleBot4 (Humble).

When a NavigateToPose goal ends ABORTED:
  BLOCKED (person or unmapped object ahead)
      -> beep, stay still, wait RETRY_WAIT_S, resend the same goal
  WALL TRAP (nothing unmapped ahead, no recent person)
      -> no beep, reverse 0.30 m once per goal on cmd_vel with a rear LiDAR
         check, resend the same goal. (Nav2 BackUp is not used: its costmap
         collision check fails at step 0 when the footprint already touches
         the wall - live 17 Sep: 'Collision Ahead - Exiting DriveOnHeading'.)
  retries used up
      -> give-up beep, stop

"Blocked" is true if EITHER
  a) a person from /person_positions_fused was within PERSON_BLOCK_M of the
     robot in the last PERSON_MEMORY_S (survives the camera losing a close
     person), OR
  b) the LiDAR saw a point NOT on the static map within LIDAR_BLOCK_M,
     +/- FRONT_HALF_DEG ahead, in the last LIDAR_MEMORY_S.

Goal source: the RViz Nav2 panel sends goals straight to the action server and
never publishes /turtlebot4/goal_pose, so the goal is taken from the END of the
latest /turtlebot4/plan. A new goal (a goal id this node did not send) resets
the retry counter.

Run:
  python3 beep_retry_node_sim.py --ros-args -p use_sim_time:=true
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.duration import Duration

from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from action_msgs.msg import GoalStatusArray, GoalStatus
from nav2_msgs.action import NavigateToPose
from irobot_create_msgs.msg import AudioNoteVector, AudioNote
from builtin_interfaces.msg import Duration as DurationMsg
import tf2_ros

NS = ""   # SIM: root namespace
PLAN_TOPIC = f"{NS}/plan"
NAV_ACTION = f"{NS}/navigate_to_pose"
CMD_VEL_TOPIC = f"{NS}/cmd_vel"
ODOM_TOPIC = f"{NS}/odom"
AUDIO_TOPIC = f"{NS}/cmd_audio"
SCAN_TOPIC = f"{NS}/scan"
MAP_TOPIC = f"{NS}/map"
PERSON_TOPIC = "/person_positions_fused"

MAX_RETRIES = 3
RETRY_WAIT_S = 5.0
PERSON_BLOCK_M = 1.5
PERSON_MEMORY_S = 5.0
LIDAR_BLOCK_M = 1.0
LIDAR_MEMORY_S = 2.0
FRONT_HALF_DEG = 30.0
MAP_MARGIN_CELLS = 3          # same as headon_metrics.py
BACKUP_DIST_M = 0.50
BACKUP_SPEED = 0.20          # m/s, same as the stock BT BackUp
BACKUP_TIMEOUT_S = 10.0
REAR_HALF_DEG = 30.0
REAR_STOP_RANGE_M = 0.40      # LiDAR range; ~0.25 m gap behind the robot body (lidar -0.04 m, radius 0.189 m)

MAP_FRAME = "map"
BASE_FRAME = "base_link"
SCAN_FRAME = "rplidar_link"   # SIM: scan header frame is not in TF; this frame is (verified with tf2_echo)


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class BeepRetryNode(Node):
    def __init__(self):
        super().__init__("beep_retry_node")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST)
        sensor = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                            history=HistoryPolicy.KEEP_LAST)

        self.create_subscription(Path, PLAN_TOPIC, self.on_plan, 10)
        self.create_subscription(GoalStatusArray, f"{NAV_ACTION}/_action/status",
                                 self.on_status, latched)
        self.create_subscription(String, PERSON_TOPIC, self.on_person, 10)
        self.create_subscription(LaserScan, SCAN_TOPIC, self.on_scan, sensor)
        self.create_subscription(OccupancyGrid, MAP_TOPIC, self.on_map, latched)
        self.audio_pub = self.create_publisher(AudioNoteVector, AUDIO_TOPIC, 10)

        self.nav_client = ActionClient(self, NavigateToPose, NAV_ACTION)
        self.cmd_pub = self.create_publisher(TwistStamped, CMD_VEL_TOPIC, 10)
        self.create_subscription(Odometry, ODOM_TOPIC, self.on_odom, sensor)
        self.odom_xy = None
        self.rear_min = None          # nearest LiDAR return in the rear sector (latest scan)
        self.reverse_timer = None
        self.reverse_start_xy = None
        self.reverse_start_t = 0.0

        self.goal = None              # goal pose = end of latest plan (PoseStamped)
        self.current_id = None        # goal id currently being tracked
        self.own_ids = set()          # goal ids sent by this node (retries)
        self.expect_own = False       # a retry was just sent; its status may arrive before the accept callback
        self.retries = 0
        self.backed_up = False
        self.handled_ids = set()      # goal ids already acted on
        self.retry_timer = None
        self.busy = False             # waiting or backing up

        self.last_person_block = 0.0
        self.last_lidar_block = 0.0
        self.grid = None
        self.mount_yaw = None

        self.get_logger().info(
            f"beep_retry_node: retries {MAX_RETRIES}, wait {RETRY_WAIT_S:.0f} s, "
            f"person {PERSON_BLOCK_M} m / {PERSON_MEMORY_S:.0f} s, "
            f"lidar unmapped {LIDAR_BLOCK_M} m +/-{FRONT_HALF_DEG:.0f} deg / {LIDAR_MEMORY_S:.0f} s, "
            f"backup {BACKUP_DIST_M} m once per goal")

    def now_s(self):
        # SIM: node clock (sim time), same clock as create_timer
        return self.get_clock().now().nanoseconds * 1e-9

    def stop_cmd(self):
        t = TwistStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = BASE_FRAME
        return t

    # ------------------------------------------------------------ inputs
    def on_plan(self, msg):
        if not msg.poses:
            return
        end = msg.poses[-1]
        g = PoseStamped()
        g.header.frame_id = msg.header.frame_id or MAP_FRAME
        g.pose = end.pose
        if self.goal is None or math.hypot(g.pose.position.x - self.goal.pose.position.x,
                                           g.pose.position.y - self.goal.pose.position.y) > 0.05:
            self.get_logger().info(f"goal from plan end: ({g.pose.position.x:.2f}, {g.pose.position.y:.2f})")
        self.goal = g

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
                self.get_logger().info(
                    f"LiDAR mount yaw ({BASE_FRAME}->{SCAN_FRAME}): {math.degrees(self.mount_yaw):+.1f} deg")
            except Exception:
                return
        try:
            tr = self.tf_buffer.lookup_transform(MAP_FRAME, SCAN_FRAME, rclpy.time.Time())
        except Exception:
            return
        lx, ly = tr.transform.translation.x, tr.transform.translation.y
        lyaw = yaw_of(tr.transform.rotation)
        rear = None
        a = msg.angle_min
        for r in msg.ranges:
            if 0.15 < r < 10.0 and not math.isinf(r) and not math.isnan(r):
                b = (math.degrees(a + self.mount_yaw) + 180.0) % 360.0 - 180.0
                if abs(b) >= 180.0 - REAR_HALF_DEG and (rear is None or r < rear):
                    rear = r
            a += msg.angle_increment
        self.rear_min = rear
        a = msg.angle_min
        for r in msg.ranges:
            if 0.15 < r <= LIDAR_BLOCK_M and not math.isinf(r) and not math.isnan(r):
                bearing = math.degrees(a + self.mount_yaw)
                bearing = (bearing + 180.0) % 360.0 - 180.0
                if abs(bearing) <= FRONT_HALF_DEG:
                    x = lx + r * math.cos(lyaw + a)
                    y = ly + r * math.sin(lyaw + a)
                    if not self.on_map_cell_occupied(x, y):
                        self.last_lidar_block = self.now_s()
                        return
            a += msg.angle_increment

    # ------------------------------------------------------------ decision
    def on_status(self, msg):
        if not msg.status_list:
            return
        newest = max(msg.status_list,
                     key=lambda s: (s.goal_info.stamp.sec, s.goal_info.stamp.nanosec))
        gid = bytes(newest.goal_info.goal_id.uuid)

        # new goal from outside this node -> reset
        if gid != self.current_id and newest.status in (GoalStatus.STATUS_ACCEPTED,
                                                        GoalStatus.STATUS_EXECUTING):
            self.current_id = gid
            if self.expect_own:
                self.expect_own = False
                self.own_ids.add(gid)
            elif gid not in self.own_ids:
                self.goal = None
                self.retries = 0
                self.backed_up = False
                self.busy = False
                self.cancel_timer()
                if self.reverse_timer is not None:
                    self.reverse_timer.cancel()
                    self.destroy_timer(self.reverse_timer)
                    self.reverse_timer = None
                    self.cmd_pub.publish(self.stop_cmd())
                    self.get_logger().info("reverse cancelled by new goal")
                self.get_logger().info("new user goal - retry counter reset")
            return
        if self.goal is None:
            if newest.status == GoalStatus.STATUS_ABORTED and gid not in self.handled_ids:
                self.handled_ids.add(gid)
                self.get_logger().warn("goal ABORTED but no plan was received - cannot retry")
            return
        if newest.status != GoalStatus.STATUS_ABORTED or gid in self.handled_ids:
            if newest.status == GoalStatus.STATUS_SUCCEEDED and gid not in self.handled_ids:
                self.handled_ids.add(gid)
                self.get_logger().info("goal SUCCEEDED")
            return
        self.handled_ids.add(gid)
        if self.busy:
            return

        now = self.now_s()
        person = now - self.last_person_block <= PERSON_MEMORY_S
        lidar = now - self.last_lidar_block <= LIDAR_MEMORY_S
        blocked = person or lidar

        if self.retries >= MAX_RETRIES:
            self.get_logger().warn(f"goal ABORTED - {MAX_RETRIES} retries used, giving up")
            self.beep_give_up()
            return
        self.retries += 1

        if blocked:
            why = " + ".join(w for w, on in (("person", person), ("unmapped lidar", lidar)) if on)
            self.get_logger().info(
                f"goal ABORTED - BLOCKED ({why}) -> beep, wait {RETRY_WAIT_S:.0f} s, retry {self.retries}/{MAX_RETRIES}")
            self.beep_blocked()
            self.schedule_retry(RETRY_WAIT_S)
        elif not self.backed_up:
            self.get_logger().info(
                f"goal ABORTED - WALL TRAP -> backup {BACKUP_DIST_M} m, retry {self.retries}/{MAX_RETRIES}")
            self.backed_up = True
            self.do_backup()
        else:
            self.get_logger().info(
                f"goal ABORTED - WALL TRAP (already backed up) -> retry {self.retries}/{MAX_RETRIES}")
            self.schedule_retry(RETRY_WAIT_S)

    # ------------------------------------------------------------ actions
    def cancel_timer(self):
        if self.retry_timer is not None:
            self.retry_timer.cancel()
            self.destroy_timer(self.retry_timer)
            self.retry_timer = None

    def schedule_retry(self, delay):
        self.busy = True
        self.cancel_timer()
        self.retry_timer = self.create_timer(delay, self.send_retry)

    def send_retry(self):
        self.cancel_timer()
        self.busy = False
        if self.goal is None:
            return
        if not self.nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error(f"{NAV_ACTION} not available - retry skipped")
            return
        g = NavigateToPose.Goal()
        g.pose = self.goal
        g.pose.header.stamp = self.get_clock().now().to_msg()
        self.expect_own = True
        fut = self.nav_client.send_goal_async(g)
        fut.add_done_callback(self.on_retry_accepted)
        self.get_logger().info(
            f"retry goal sent to ({g.pose.pose.position.x:.2f}, {g.pose.pose.position.y:.2f})")

    def on_retry_accepted(self, fut):
        handle = fut.result()
        if handle is None or not handle.accepted:
            self.expect_own = False
            self.get_logger().warn("retry goal rejected")
            return
        gid = bytes(handle.goal_id.uuid)
        self.own_ids.add(gid)
        self.current_id = gid

    def on_odom(self, msg):
        p = msg.pose.pose.position
        self.odom_xy = (p.x, p.y)

    def do_backup(self):
        self.busy = True
        if self.odom_xy is None or self.rear_min is None:
            self.get_logger().warn("no odom or scan yet - retrying without reverse")
            self.schedule_retry(RETRY_WAIT_S)
            return
        if self.rear_min < REAR_STOP_RANGE_M:
            self.get_logger().warn(
                f"reverse skipped: obstacle behind at {self.rear_min:.2f} m - retrying without reverse")
            self.schedule_retry(RETRY_WAIT_S)
            return
        self.reverse_start_xy = self.odom_xy
        self.reverse_start_t = self.now_s()
        self.reverse_timer = self.create_timer(0.05, self.reverse_step)
        self.get_logger().info(
            f"reversing {BACKUP_DIST_M} m at {BACKUP_SPEED} m/s (rear clear {self.rear_min:.2f} m)")

    def reverse_step(self):
        moved = math.hypot(self.odom_xy[0] - self.reverse_start_xy[0],
                           self.odom_xy[1] - self.reverse_start_xy[1])
        reason = None
        if moved >= BACKUP_DIST_M:
            reason = f"done, {moved:.2f} m"
        elif self.rear_min is not None and self.rear_min < REAR_STOP_RANGE_M:
            reason = f"obstacle behind at {self.rear_min:.2f} m after {moved:.2f} m"
        elif self.now_s() - self.reverse_start_t > BACKUP_TIMEOUT_S:
            reason = f"timeout after {moved:.2f} m"
        if reason is None:
            t = self.stop_cmd()
            t.twist.linear.x = -BACKUP_SPEED
            self.cmd_pub.publish(t)
            return
        self.cmd_pub.publish(self.stop_cmd())
        self.reverse_timer.cancel()
        self.destroy_timer(self.reverse_timer)
        self.reverse_timer = None
        self.get_logger().info(f"reverse stopped: {reason} - retrying in 1 s")
        self.schedule_retry(1.0)

    # ------------------------------------------------------------ audio
    def play(self, notes):
        msg = AudioNoteVector()
        msg.append = False
        for freq, dur in notes:
            n = AudioNote()
            n.frequency = freq
            n.max_runtime = DurationMsg(sec=int(dur), nanosec=int((dur % 1) * 1e9))
            msg.notes.append(n)
        self.audio_pub.publish(msg)

    def beep_blocked(self):
        self.play([(880, 0.2), (1100, 0.2), (880, 0.2)])        # short high triple beep

    def beep_give_up(self):
        self.play([(440, 0.5), (330, 0.5), (220, 0.8)])        # falling tones


def main():
    rclpy.init()
    node = BeepRetryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
