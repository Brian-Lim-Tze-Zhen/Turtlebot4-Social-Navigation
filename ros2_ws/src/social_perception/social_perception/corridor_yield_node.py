#!/usr/bin/env python3
"""
corridor_yield_node.py

Head-on corridor yield behaviour:
  NAVIGATING  — robot is driving toward its goal normally
  YIELDING    — person approaching head-on within YIELD_DIST; robot stops and beeps
  RESUMING    — person has passed (distance > CLEAR_DIST); robot re-sends goal

Subscribes:
  /predicted_person_positions  (std_msgs/String)  — KF output
  /goal_pose                   (geometry_msgs/PoseStamped) — captured from RViz

Controls Nav2 via /navigate_to_pose action (cancel to stop, re-send to resume).
Beeps via /cmd_audio (irobot_create_msgs/msg/AudioNoteVector).
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose

try:
    from irobot_create_msgs.msg import AudioNoteVector, AudioNote
    from builtin_interfaces.msg import Duration
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

# ── tuneable thresholds ────────────────────────────────────────────────────────
YIELD_DIST  = 1.8   # m — stop when person this close and approaching
CLEAR_DIST  = 2.5   # m — resume when person this far away
CHECK_HZ    = 5.0   # state machine tick rate
# ──────────────────────────────────────────────────────────────────────────────

STATE_NAVIGATING = "NAVIGATING"
STATE_YIELDING   = "YIELDING"
STATE_RESUMING   = "RESUMING"


class CorridorYieldNode(Node):

    def __init__(self):
        super().__init__("corridor_yield_node")

        self.declare_parameter("yield_dist", YIELD_DIST)
        self.declare_parameter("clear_dist", CLEAR_DIST)
        self.yield_dist = self.get_parameter("yield_dist").value
        self.clear_dist = self.get_parameter("clear_dist").value

        # last known robot position (from /goal_pose frame; good enough for dist)
        self.robot_x = 0.0
        self.robot_y = 0.0

        # last person state from KF
        self.person_x  = None
        self.person_y  = None
        self.person_vx = None
        self.person_vy = None

        # current Nav2 goal (captured from /goal_pose)
        self.current_goal: PoseStamped | None = None
        self._nav_goal_handle = None

        self.state = STATE_NAVIGATING

        # Nav2 action client
        self._nav_client = ActionClient(self, NavigateToPose, "/navigate_to_pose")

        # Beep publisher
        if HAS_AUDIO:
            self._audio_pub = self.create_publisher(
                AudioNoteVector, "/cmd_audio", 10)
        else:
            self._audio_pub = None
            self.get_logger().warn("irobot_create_msgs not found — beep disabled")

        sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )

        self.create_subscription(
            String, "/predicted_person_positions",
            self._person_cb, sensor_qos)

        # Capture goal set in RViz so we can re-send it after yielding
        self.create_subscription(
            PoseStamped, "/goal_pose",
            self._goal_cb, 10)

        # Also subscribe to /robot_pose if available; fallback to last known
        # position being 0,0 until we get a better source.
        try:
            from nav_msgs.msg import Odometry
            self.create_subscription(Odometry, "/odom", self._odom_cb, sensor_qos)
        except Exception:
            pass

        self.create_timer(1.0 / CHECK_HZ, self._tick)

        self.get_logger().info(
            f"Corridor yield node ready | "
            f"yield_dist={self.yield_dist}m  clear_dist={self.clear_dist}m")

    # ── subscribers ───────────────────────────────────────────────────────────

    def _odom_cb(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

    def _goal_cb(self, msg: PoseStamped):
        self.current_goal = msg
        self.get_logger().info(
            f"Goal captured: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")
        if self.state == STATE_NAVIGATING:
            self._send_nav_goal(msg)

    def _person_cb(self, msg: String):
        parts = msg.data.split(",")
        if len(parts) < 6:
            return
        try:
            self.person_x  = float(parts[2])
            self.person_y  = float(parts[3])
            self.person_vx = float(parts[4])
            self.person_vy = float(parts[5])
        except ValueError:
            pass

    # ── state machine tick ────────────────────────────────────────────────────

    def _tick(self):
        if self.person_x is None:
            return

        dist = math.hypot(self.person_x - self.robot_x,
                          self.person_y - self.robot_y)

        # Vector from person to robot
        dx = self.robot_x - self.person_x
        dy = self.robot_y - self.person_y

        # Is the person moving toward the robot?
        # dot(person_velocity, robot_direction) > 0 means approaching
        approaching = (self.person_vx * dx + self.person_vy * dy) > 0.05

        if self.state == STATE_NAVIGATING:
            if dist < self.yield_dist and approaching:
                self.get_logger().warn(
                    f"[YIELD] Person {dist:.2f}m away and approaching — stopping")
                self._cancel_nav_goal()
                self._beep()
                self.state = STATE_YIELDING

        elif self.state == STATE_YIELDING:
            if dist > self.clear_dist and not approaching:
                self.get_logger().info(
                    f"[RESUME] Person cleared ({dist:.2f}m) — resuming goal")
                self.state = STATE_RESUMING
                if self.current_goal is not None:
                    self._send_nav_goal(self.current_goal)
                    self.state = STATE_NAVIGATING

        self.get_logger().debug(
            f"[{self.state}] person dist={dist:.2f}m approaching={approaching}")

    # ── nav2 helpers ──────────────────────────────────────────────────────────

    def _send_nav_goal(self, pose: PoseStamped):
        if not self._nav_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn("NavigateToPose server not available")
            return
        goal = NavigateToPose.Goal()
        goal.pose = pose
        future = self._nav_client.send_goal_async(goal)
        future.add_done_callback(self._goal_response_cb)
        self.get_logger().info("Nav2 goal sent")

    def _goal_response_cb(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn("Nav2 goal rejected")
            return
        self._nav_goal_handle = handle

    def _cancel_nav_goal(self):
        if self._nav_goal_handle is not None:
            self._nav_goal_handle.cancel_goal_async()
            self._nav_goal_handle = None

    # ── beep ──────────────────────────────────────────────────────────────────

    def _beep(self):
        if self._audio_pub is None or not HAS_AUDIO:
            self.get_logger().info("[BEEP] (audio not available)")
            return
        msg = AudioNoteVector()
        # Two short beeps: 880 Hz for 0.2s, pause, 880 Hz for 0.2s
        for _ in range(2):
            note = AudioNote()
            note.frequency = 880
            note.max_runtime = Duration(sec=0, nanosec=200_000_000)
            msg.notes.append(note)
            pause = AudioNote()
            pause.frequency = 0
            pause.max_runtime = Duration(sec=0, nanosec=100_000_000)
            msg.notes.append(pause)
        msg.append = False
        self._audio_pub.publish(msg)
        self.get_logger().info("[BEEP] Published to /cmd_audio")


def main(args=None):
    rclpy.init(args=args)
    node = CorridorYieldNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
