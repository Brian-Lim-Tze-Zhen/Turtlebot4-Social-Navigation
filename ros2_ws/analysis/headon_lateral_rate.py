#!/usr/bin/env python3
"""Read-only: how fast does the robot build lateral offset in the head-on trials?"""
import sys, math, glob, os, statistics
sys.path.insert(0, "/ws/analysis")
import analyse_avoidance as aa
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
def gt_path(d):
    r = SequentialReader(); r.open(StorageOptions(uri=d, storage_id="mcap"), ConverterOptions("", ""))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    if "/sim_ground_truth_pose" not in types: return None
    C = get_message(types["/sim_ground_truth_pose"]); out = []
    while r.has_next():
        tp, data, t = r.read_next()
        if tp == "/sim_ground_truth_pose":
            m = deserialize_message(data, C); out.append((aa.stamp_to_sec(m.header.stamp), m.pose.pose.position.x, m.pose.pose.position.y))
    return out
def rate(path, lo=0.1, hi=0.7):
    # lateral offset from the straight line y = y at the start of forward motion (trials run along x)
    y0 = statistics.median(p[2] for p in path[:50])
    t_lo = next((p[0] for p in path if abs(p[2] - y0) >= lo), None)
    t_hi = next((p[0] for p in path if abs(p[2] - y0) >= hi), None)
    mx = max(abs(p[2] - y0) for p in path)
    # peak lateral speed over 1 s windows
    peak = 0.0; j = 0
    for i in range(len(path)):
        while j < len(path) and path[j][0] - path[i][0] < 1.0: j += 1
        if j < len(path): peak = max(peak, abs(path[j][2] - path[i][2]) / (path[j][0] - path[i][0]))
    return (hi - lo) / (t_hi - t_lo) if t_lo and t_hi and t_hi > t_lo else None, peak, mx
for label, pat, gt in (("B: cloud, open world", "/ws/bags/b_critweight20_t0*/data", False), ("E: cloud and critic, open world", "/ws/bags/e_critweight20_socialcritic_on_t0*/data", False), ("lane rules, corridor", "/ws/bags/headon_hwreq_z5_corr_trial*/data", True)):
    R = []
    for d in sorted(glob.glob(pat)):
        if gt: path = gt_path(d)
        else:
            data = aa.read_bag(d); path = aa.robot_path_in_map(data["/odom"], aa.extract_map_to_odom(data["/tf"] + data["/tf_static"]))
        r = rate(path); R.append(r)
        print(f"  {os.path.basename(os.path.dirname(d))}: mean lateral rate from 0.1 to 0.7 m offset {r[0]:.2f} m/s; peak over 1 s {r[1]:.2f} m/s; largest offset {r[2]:.2f} m" if r[0] else f"  {os.path.basename(os.path.dirname(d))}: offset of 0.7 m not reached (largest {r[2]:.2f} m)", flush=True)
    v = [r[0] for r in R if r[0]]; pk = [r[1] for r in R]
    print(f"{label}: mean lateral rate {statistics.mean(v):.2f} +/- {statistics.pstdev(v):.2f} m/s (n={len(v)}), peak {statistics.mean(pk):.2f} m/s")
