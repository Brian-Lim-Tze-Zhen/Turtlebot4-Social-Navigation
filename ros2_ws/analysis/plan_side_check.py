#!/usr/bin/env python3
"""plan_side_check.py - on which side of the walker does each published global plan pass?
For every /plan message published while the walker is still ahead of the robot (robot x < walker x - GAP), the plan is read at the x of the walker
at that time (ground truth, /person_ground_truth): side = sign(plan y - walker y). With the robot keeping right it passes on the walker's -y side,
so a plan on the +y side is on the 'wrong' side. Tolerance TOL for 'centre' (the plan runs through the walker).
Usage: python3 plan_side_check.py <bag dir> [<bag dir> ...]"""
import sys, os
import rosbag2_py
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseArray
GAP, TOL = 2.0, 0.15
def run(bag):
    uri = bag + "/data" if os.path.exists(bag + "/data/metadata.yaml") else bag
    r = rosbag2_py.SequentialReader(); r.open(rosbag2_py.StorageOptions(uri=uri, storage_id=""), rosbag2_py.ConverterOptions("", ""))
    walker = None; left = right = centre = 0
    while r.has_next():
        tp, d, t = r.read_next()
        if tp == "/person_ground_truth":
            m = deserialize_message(d, PoseArray)
            if m.poses: walker = (m.poses[0].position.x, m.poses[0].position.y)
        elif tp == "/plan" and walker is not None:
            m = deserialize_message(d, Path); pts = [(q.pose.position.x, q.pose.position.y) for q in m.poses]
            if not pts or walker[0] - pts[0][0] < GAP: continue
            near = [q for q in pts if abs(q[0] - walker[0]) < 0.3]
            if not near: continue
            dy = sum(q[1] for q in near) / len(near) - walker[1]
            if dy > TOL: left += 1
            elif dy < -TOL: right += 1
            else: centre += 1
    return left, right, centre
tot = [0, 0, 0]
for b in sys.argv[1:]:
    res = run(b); tot = [a + c for a, c in zip(tot, res)]
    print(f"{os.path.basename(b)}: wrong side (+y) {res[0]}, right (-y) {res[1]}, centre {res[2]}")
n = sum(tot)
print(f"TOTAL plans {n}: wrong side {tot[0]} ({100*tot[0]/max(n,1):.1f} %), right {tot[1]}, centre {tot[2]}")
