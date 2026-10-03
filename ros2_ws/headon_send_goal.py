#!/usr/bin/env python3
"""headon_send_goal.py <x> <y> <timeout_s> <accepted_flag_file>

NavigateToPose client for run_headon_F_trial.sh. Unlike
`ros2 action send_goal` it retries the goal handshake: under trial load
bt_navigator was seen to log "Failed to send goal response (timeout)",
the CLI never heard back, and the robot sat still for the whole trial
while the person walked through it.

Touches <accepted_flag_file> as soon as the goal is accepted, so the
trial script can start the person at that moment. Prints the final
status (SUCCEEDED / ABORTED / CANCELED / TIMEOUT / NOT_ACCEPTED) and
exits 0 only on SUCCEEDED. The timeout is in wall seconds.
"""
import sys
import time

import rclpy
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose

ACCEPT_TRIES = 4
ACCEPT_WAIT_S = 5.0
# An abort this soon after acceptance is the stack failing to start the
# controller (seen: bt_navigator "Timed out while waiting for action server
# to acknowledge goal request for follow_path" under load), not an outcome
# of the encounter. Reported separately so the trial can be rerun.
EARLY_ABORT_S = 10.0
STATUS = {GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
          GoalStatus.STATUS_ABORTED: "ABORTED",
          GoalStatus.STATUS_CANCELED: "CANCELED"}


def spin_until(node, future, timeout_s):
    end = time.monotonic() + timeout_s
    while rclpy.ok() and not future.done() and time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    return future.done()


def main():
    x, y, timeout_s, flag = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    rclpy.init()
    node = rclpy.create_node("headon_send_goal")
    client = ActionClient(node, NavigateToPose, "/navigate_to_pose")
    if not client.wait_for_server(timeout_sec=30.0):
        print("NOT_ACCEPTED (no /navigate_to_pose server)", flush=True)
        sys.exit(1)

    goal = NavigateToPose.Goal()
    goal.pose.header.frame_id = "map"
    goal.pose.pose.position.x = x
    goal.pose.pose.position.y = y
    goal.pose.pose.orientation.w = 1.0

    handle = None
    for attempt in range(1, ACCEPT_TRIES + 1):
        goal.pose.header.stamp = node.get_clock().now().to_msg()
        fut = client.send_goal_async(goal)
        if spin_until(node, fut, ACCEPT_WAIT_S) and fut.result() is not None \
                and fut.result().accepted:
            handle = fut.result()
            break
        print(f"goal not acknowledged (attempt {attempt}/{ACCEPT_TRIES}), retrying", flush=True)
    if handle is None:
        print("NOT_ACCEPTED", flush=True)
        sys.exit(1)

    open(flag, "w").close()
    print(f"Goal accepted: ({x}, {y})", flush=True)

    t_accept = time.monotonic()
    res = handle.get_result_async()
    if not spin_until(node, res, timeout_s):
        handle.cancel_goal_async()
        print("TIMEOUT", flush=True)
        sys.exit(1)
    status = STATUS.get(res.result().status, f"STATUS_{res.result().status}")
    if status == "ABORTED" and time.monotonic() - t_accept < EARLY_ABORT_S:
        status = "EARLY_ABORT"
    print(f"Goal finished with status: {status}", flush=True)
    sys.exit(0 if status == "SUCCEEDED" else 1)


if __name__ == "__main__":
    main()
