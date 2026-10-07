#!/usr/bin/env python3
"""plan_bend_check.py - how far does the global plan bend to the side? For each bag: the largest |y| of any /plan point with x between -1 and +3 over all
plans published while the walker is 2 m or more ahead of the robot, and the largest lateral offset of the plan from the walker's line (y = 0) in that region.
Usage: python3 plan_bend_check.py <bag dir> ..."""
import sys, os
import rosbag2_py
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseArray
for bag in sys.argv[1:]:
    uri = bag + "/data" if os.path.exists(bag + "/data/metadata.yaml") else bag
    r = rosbag2_py.SequentialReader(); r.open(rosbag2_py.StorageOptions(uri=uri, storage_id=""), rosbag2_py.ConverterOptions("", ""))
    walker = None; best = 0.0; n = 0; per = []
    while r.has_next():
        tp, d, t = r.read_next()
        if tp == "/person_ground_truth":
            m = deserialize_message(d, PoseArray)
            if m.poses: walker = m.poses[0].position.x
        elif tp == "/plan" and walker is not None:
            m = deserialize_message(d, Path); pts = [(q.pose.position.x, q.pose.position.y) for q in m.poses]
            if not pts or walker - pts[0][0] < 2.0: continue
            ys = [abs(y) for x, y in pts if -1.0 <= x <= 3.0]
            if ys: n += 1; best = max(best, max(ys)); per.append(max(ys))
    print(f"{os.path.basename(bag)}: plans {n}, largest bend {best:.2f} m")
