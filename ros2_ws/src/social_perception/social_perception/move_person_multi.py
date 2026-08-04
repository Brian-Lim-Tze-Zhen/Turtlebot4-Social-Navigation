#!/usr/bin/env python3
#
# move_person_multi.py
#
# THESIS ADDITION (sequential multi-human scenario)
#
# Two people, two different encounter geometries, staggered in space so
# the robot meets them one after the other rather than simultaneously:
#
#   person_1 : HEAD-ON.  Walks along the x-axis at y=0, toward the
#              robot. Endpoints (3,0) <-> (6,0), same as the
#              single-person head-on scenario.
#
#   person_2 : CROSSING. Walks along the y-axis at fixed x=6.5,
#              perpendicular to the robot's path.
#
# The robot navigates (0,0) -> (8,0) as in every other scenario, so it
# encounters person_1 first (around x~3) and person_2 second (x~6.5).
#
# WHY SEQUENTIAL RATHER THAN SIMULTANEOUS
# ---------------------------------------
# Simultaneous conflicts are the more impressive demonstration, but if
# the robot fails there are three candidate causes that cannot be
# separated from the recorded data: multi-track ID confusion in the
# perception pipeline, two overlapping prediction ellipses saturating
# the corridor, or MPPI having no valid trajectory left given the 1.0 m
# inflation radius. Sequential encounters stay individually
# interpretable while STILL exercising the multi-track paths - both
# people are tracked, predicted, and injected into the costmap
# throughout the run; only the conflict moments are staggered.
#
# TIMING IS THE HARD PART
# -----------------------
# The single-person crossing probe (bags/crossing_full_probe) failed
# because the robot and the person were never at the intersection at
# the same moment: the recorded closest approach came from an unrelated
# later instant, after the person had reversed. With two people there
# are two such timings to get right.
#
# PERSON_2_START_DELAY below exists for exactly this reason. It must be
# calibrated against measured robot arrival times (see the x-position
# timing probe in eval_scripts) - the default here is a STARTING GUESS,
# not a measured value. Run a probe and check both encounters actually
# occur before recording any evaluation trials.
#
# Both people bounce between their endpoints continuously. This is more
# realistic than stopping after one pass, but note the consequence
# observed in the crossing probe: a person who reverses can approach the
# robot again AFTER their real encounter is over, which can set
# min_distance from a moment where no avoidance was possible or
# required. The per-person encounter windows added to
# analyse_avoidance.py should isolate this correctly - verify on the
# probe runs that they do.

import math
import subprocess
import threading

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseArray, Pose


# =====================================================================
# TIMING CALIBRATION
#
# Seconds after node start before person_2 begins moving. Person_1
# starts immediately.
#
# Set this so person_2 is near y=0 (crossing the robot's path) at the
# moment the robot reaches x=6.5. Measure the robot's arrival time at
# x=6.5 from a probe bag, subtract the time person_2 needs to walk from
# their start y to y=0 (distance / speed), and use the difference.
#
# CALIBRATED from bags/crossing_full_probe robot arrival times
# (x=6 at t=26.21s, x=7 at t=30.10s -> x=6.5 at t~28.2s).
# person_2 walks 2.0 m from y=-2 to y=0 at 0.2 m/s = 10.0 s.
# 28.2 - 10.0 = 18.2, rounded to 18.0.
# =====================================================================
PERSON_2_START_DELAY = 18.0


class PersonMover:
    """Holds moving state for one Gazebo person model and steps itself
    toward its current target waypoint. Supports an endpoint pause and
    an optional start delay."""

    def __init__(self, model_name, point_a, point_b, speed=0.2,
                 pause_duration=1.5, start_delay=0.0):
        self.model_name = model_name
        self.point_a = point_a
        self.point_b = point_b
        self.speed = speed
        self.pause_duration = pause_duration

        self.start_delay = start_delay
        self.elapsed = 0.0

        self.pause_timer = 0.0

        self.current_x = point_a[0]
        self.current_y = point_a[1]
        self.current_z = point_a[2]

        self.target = point_b

    def step(self, dt):
        """Returns (x, y, z, yaw) if a new pose should be sent this
        tick, or None if the person is delayed, paused, or has just
        switched target. Position remains valid in all cases."""

        self.elapsed += dt

        if self.elapsed < self.start_delay:
            return None

        if self.pause_timer > 0.0:
            self.pause_timer = max(0.0, self.pause_timer - dt)
            return None

        target_x, target_y, target_z = self.target

        dx = target_x - self.current_x
        dy = target_y - self.current_y

        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 0.05:
            self.pause_timer = self.pause_duration
            self.target = self.point_a if self.target == self.point_b else self.point_b
            return None

        step = min(self.speed * dt, dist)

        ux = dx / dist
        uy = dy / dist

        self.current_x += ux * step
        self.current_y += uy * step

        # +pi/2 offset corrects the person_standing model's default mesh
        # orientation. Verified for x-axis motion; verify visually for
        # person_2's y-axis motion before recording.
        yaw = math.atan2(uy, ux) + math.pi / 2.0
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))

        return self.current_x, self.current_y, self.current_z, yaw


class MovePeopleMulti(Node):
    def __init__(self):
        super().__init__("move_people_multi")

        self.world_name = "two_human"

        # 0.5 s to match move_person_gazebo.py's ground-truth resolution.
        # At 1.0 s, /person_ground_truth published only ~47 samples over a
        # 47 s run, so min_distance was interpolated across 1 s gaps during
        # which the crossing person moves 0.2 m and the robot 0.25 m -
        # too coarse to trust a 0.387 m reported minimum.
        self.update_dt = 0.5

        self.people = [
            # HEAD-ON: walks toward the robot along y=0.
            PersonMover(
                "person_1",
                # Pulled back from (3,0)<->(6,0): at 0.2 m/s person_1
                # would otherwise be near x~5.8 when the robot reaches
                # x=3 (t=14.2s), meeting around x~5 at t~20s - too close
                # to person_2's crossing at x=6.5, t~28s, and person_1
                # would still be near x=6 at that moment. These
                # endpoints put the head-on meeting near x~3 at t~14s,
                # leaving a ~14 s gap between the two encounters.
                point_a=(2.0, 0.0, 0.0),
                point_b=(4.0, 0.0, 0.0),
                speed=0.2,
                pause_duration=1.5,
                start_delay=0.0,
            ),
            # CROSSING: walks perpendicular across the robot's path.
            PersonMover(
                "person_2",
                point_a=(6.5, -2.0, 0.0),
                point_b=(6.5, 2.0, 0.0),
                speed=0.2,
                pause_duration=1.5,
                start_delay=PERSON_2_START_DELAY,
            ),
        ]

        self.last_time = self.get_clock().now()
        self.timer = self.create_timer(self.update_dt, self.timer_callback)

        # Ground truth for offline evaluation only - never feeds
        # navigation. Pose order matches self.people, so pose index 0 is
        # person_1 (head-on) and index 1 is person_2 (crossing). This
        # ordering is what analyse_avoidance.py's person_tracks_all()
        # keys on, so DO NOT reorder this list without updating any
        # per-person result tables.
        self.ground_truth_pub = self.create_publisher(
            PoseArray,
            "/person_ground_truth",
            10
        )
        self.frame_id = "map"

        self.get_logger().info("Multi-human sequential scenario")
        self.get_logger().info(f"World: {self.world_name}")
        for i, p in enumerate(self.people):
            self.get_logger().info(
                f"  [{i}] {p.model_name}: {p.point_a} <-> {p.point_b} "
                f"delay={p.start_delay:.1f}s"
            )

    def timer_callback(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds / 1e9
        self.last_time = now

        # Clamp ceiling must stay comfortably above update_dt, or normal
        # ticks get throttled and the effective speed silently drops.
        # See the effective-speed halving bug in move_person_gazebo2.py.
        dt = max(0.01, min(dt, self.update_dt * 3.0))

        for person in self.people:
            result = person.step(dt)

            if result is None:
                continue

            x, y, z, yaw = result

            self.get_logger().info(
                f"{person.model_name} pos=({x:.2f},{y:.2f}) "
                f"target={person.target}"
            )

            threading.Thread(
                target=self.set_model_pose,
                args=(person.model_name, x, y, z, yaw),
                daemon=True,
            ).start()

        # Published every tick for ALL people regardless of whether any
        # individual person moved this tick - their position is still
        # valid while delayed or paused.
        self.publish_ground_truth()

    def publish_ground_truth(self):
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        for person in self.people:
            pose = Pose()
            pose.position.x = person.current_x
            pose.position.y = person.current_y
            pose.position.z = person.current_z
            msg.poses.append(pose)

        self.ground_truth_pub.publish(msg)

    def set_model_pose(self, model_name, x, y, z, yaw):
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)

        service_name = f"/world/{self.world_name}/set_pose"

        req = (
            f"name: '{model_name}', "
            f"position: {{x: {x}, y: {y}, z: {z}}}, "
            f"orientation: {{x: 0, y: 0, z: {qz}, w: {qw}}}"
        )

        cmd = [
            "gz", "service",
            "-s", service_name,
            "--reqtype", "gz.msgs.Pose",
            "--reptype", "gz.msgs.Boolean",
            "--timeout", "3000",
            "--req", req
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=3.0
            )
            if result.returncode != 0:
                self.get_logger().warn(
                    f"set_pose failed for {model_name}: {result.stderr}"
                )
        except subprocess.TimeoutExpired:
            self.get_logger().warn(f"set_pose service timeout for {model_name}")

    def destroy_node(self):
        self.get_logger().info("Stopping move_people_multi node")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MovePeopleMulti()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
