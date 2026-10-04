#!/usr/bin/env python3
"""
person_stop_beep_node_sim.py - SIMULATION (Jazzy, gz sim).

Beeps while the collision monitor holds the robot still for a person.

The stopping itself is done by Nav2's collision monitor: the config
social_nav2_headon_F_hwreq_block_stopbeep_sim.yaml has a "PersonStop" zone in
front of the robot that reads /predicted_person_cloud only. While a person is
inside it the monitor outputs zero velocity; when the zone is empty again the
controller's commands pass through and the robot drives on. No goal is
cancelled or resent.

This node only watches /collision_monitor_state:
  STOP by polygon "PersonStop"  -> beep now, and again every beep_period_s
  anything else                 -> silent; logs how long the stop lasted

Differs from beep_retry_node_sim.py, which acts after a goal has ABORTED.

Publishes:
  /cmd_audio          irobot_create_msgs/AudioNoteVector  (the beep)
  /person_stop_event  std_msgs/String  "stop,<sim t>" / "resume,<sim t>,<duration s>"

Run:
  python3 person_stop_beep_node_sim.py --ros-args -p use_sim_time:=true
"""
import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from nav2_msgs.msg import CollisionMonitorState
from irobot_create_msgs.msg import AudioNoteVector, AudioNote
from builtin_interfaces.msg import Duration as DurationMsg

STATE_TOPIC = "/collision_monitor_state"
AUDIO_TOPIC = "/cmd_audio"
EVENT_TOPIC = "/person_stop_event"


class PersonStopBeepNode(Node):
    def __init__(self):
        super().__init__("person_stop_beep_node")
        self.declare_parameter("polygon_name", "PersonStop")
        self.declare_parameter("beep_period_s", 2.0)
        # The monitor only publishes its state when it CHANGES, and a stop can
        # flicker off for one cycle when the cloud drops a few points. A stop
        # that comes back within this time counts as the same stop.
        self.declare_parameter("rejoin_s", 0.5)
        self.polygon_name = self.get_parameter("polygon_name").value
        self.beep_period = float(self.get_parameter("beep_period_s").value)
        self.rejoin_s = float(self.get_parameter("rejoin_s").value)

        self.create_subscription(CollisionMonitorState, STATE_TOPIC, self.on_state, 10)
        self.audio_pub = self.create_publisher(AudioNoteVector, AUDIO_TOPIC, 10)
        self.event_pub = self.create_publisher(String, EVENT_TOPIC, 10)

        self.stopped = False          # monitor currently reports the person stop
        self.stop_start = None        # sim time the current stop began
        self.released_at = None       # sim time the stop was last released
        self.last_beep = None
        self.create_timer(0.1, self.tick)
        self.get_logger().info(
            f"person_stop_beep_node: polygon '{self.polygon_name}', "
            f"beep every {self.beep_period:.1f} s while stopped")

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_state(self, msg):
        self.stopped = (msg.action_type == CollisionMonitorState.STOP
                        and msg.polygon_name == self.polygon_name)

    def tick(self):
        now = self.now_s()
        if self.stopped:
            self.released_at = None
            if self.stop_start is None:
                self.stop_start = now
                self.last_beep = None
                self.event_pub.publish(String(data=f"stop,{now:.2f}"))
                self.get_logger().info("Person too near - robot stopped, beeping.")
            if self.last_beep is None or now - self.last_beep >= self.beep_period:
                self.beep()
                self.last_beep = now
        elif self.stop_start is not None:
            if self.released_at is None:
                self.released_at = now
            elif now - self.released_at >= self.rejoin_s:
                dur = self.released_at - self.stop_start
                self.event_pub.publish(
                    String(data=f"resume,{self.released_at:.2f},{dur:.2f}"))
                self.get_logger().info(f"Person clear after {dur:.1f} s - continuing.")
                self.stop_start = None
                self.released_at = None

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
    node = PersonStopBeepNode()
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
