#!/usr/bin/env python3
"""analyse_headon_F.py - head-on trials recorded by run_headon_F_trial.sh.

Usage:
    python3 analyse_headon_F.py <bag_dir> [<bag_dir> ...]     (globs OK)

Per trial it prints a report and writes <bag>/trajectory.csv; with more
than one bag it also prints mean +/- sample SD over the trials that
reached the goal.

Definitions match analyse_F_narrow.py / aggregate_F_trials.py so the
head-on numbers are comparable with the wide and narrow conversation
cases:
  robot position = /sim_ground_truth_pose (Gazebo world frame == map frame
                   with maps/corridor_headon_aligned.yaml); fallback /amcl_pose
  velocities     = /odom twist (its pose drifts, so it is not used)
  run start      = first |vx| > MOVE_V
  goal           = first sample within GOAL_TOL of GOAL
  centre         = robot centre to true person centre
  surface        = centre - PERSON_R - ROBOT_R
The difference is the person: here it MOVES, so its position comes from
/person_ground_truth, interpolated onto the robot timeline, not from the
SDF.

Topics without a header (the CSV String topics) are stamped with the bag
receive time, which is wall time; they are converted to sim time through
/clock.
"""
import argparse
import csv
import glob
import math
import os
import statistics
import sys
from bisect import bisect_right

import yaml

try:
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError as e:
    sys.exit(f"Missing ROS 2 Python packages ({e}).\n"
             "Run this inside the container with /opt/ros/jazzy sourced.")

GOAL = (8.0, 0.0)
GOAL_TOL = 0.30
ROBOT_R = 0.189
PERSON_R = 0.25
STALL_V = 0.02          # |vx| below this = not translating
SPIN_W = 0.1            # |wz| above this while not translating = spin in place
MOVE_V = 0.05
COMMIT_DEV = 0.20       # lateral deviation that counts as "committed to a side"
FLIP_W = 0.1            # wz sign flips are counted between samples above this
KF_SPEED = 0.8          # KF speed lag is measured to this speed
PERSON_MOVE_M = 0.05    # person displacement that counts as "started walking"


def stamp_s(st):
    return st.sec + st.nanosec * 1e-9


def interp(series, t, n):
    """Linear interpolation of columns 1..n of a time-sorted list of tuples."""
    ts = [s[0] for s in series]
    i = bisect_right(ts, t)
    if i == 0:
        return series[0][1:n + 1]
    if i == len(series):
        return series[-1][1:n + 1]
    a, b = series[i - 1], series[i]
    k = (t - a[0]) / (b[0] - a[0]) if b[0] > a[0] else 0.0
    return tuple(a[j] + k * (b[j] - a[j]) for j in range(1, n + 1))


def find_bag(path):
    """run_headon_F_trial.sh records into <bag>/data; older bags sit in <bag>."""
    for p in (os.path.join(path, "data"), path):
        if os.path.exists(os.path.join(p, "metadata.yaml")):
            return p
    return None


def load(uri):
    meta = yaml.safe_load(open(os.path.join(uri, "metadata.yaml")))
    sid = meta["rosbag2_bagfile_information"]["storage_identifier"]
    r = SequentialReader()
    r.open(StorageOptions(uri=uri, storage_id=sid), ConverterOptions("", ""))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    want = ["/odom", "/sim_ground_truth_pose", "/amcl_pose", "/person_ground_truth",
            "/clock", "/person_positions_fused", "/predicted_person_positions", "/plan"]
    optional = ["/person2_ground_truth"]   # second walker (PERSON2_Y trials)
    have = [t for t in want + optional if t in types]
    r.set_filter(StorageFilter(topics=have))
    mt = {t: get_message(types[t]) for t in have}

    d = {"odom": [], "gt": [], "amcl": [], "person": [], "person2": [], "clock": [],
         "fused": [], "kf": [], "plans": 0,
         "missing": [t for t in want if t not in types]}
    while r.has_next():
        topic, raw, t_ns = r.read_next()
        msg = deserialize_message(raw, mt[topic])
        if topic == "/odom":
            tw = msg.twist.twist
            d["odom"].append((stamp_s(msg.header.stamp), tw.linear.x, tw.angular.z))
        elif topic == "/sim_ground_truth_pose":
            p = msg.pose.pose.position
            d["gt"].append((stamp_s(msg.header.stamp), p.x, p.y))
        elif topic == "/amcl_pose":
            p = msg.pose.pose.position
            d["amcl"].append((stamp_s(msg.header.stamp), p.x, p.y))
        elif topic == "/person_ground_truth":
            if msg.poses:
                p = msg.poses[0].position
                d["person"].append((stamp_s(msg.header.stamp), p.x, p.y))
        elif topic == "/person2_ground_truth":
            if msg.poses:
                p = msg.poses[0].position
                d["person2"].append((stamp_s(msg.header.stamp), p.x, p.y))
        elif topic == "/clock":
            d["clock"].append((t_ns * 1e-9, stamp_s(msg.clock)))
        elif topic == "/person_positions_fused":
            f = msg.data.split(",")
            try:
                d["fused"].append((t_ns * 1e-9, float(f[2]), float(f[3])))
            except (IndexError, ValueError):
                pass
        elif topic == "/predicted_person_positions":
            f = msg.data.split(",")
            try:
                d["kf"].append((t_ns * 1e-9, math.hypot(float(f[4]), float(f[5]))))
            except (IndexError, ValueError):
                pass
        elif topic == "/plan":
            d["plans"] += 1
    return d


def analyse(path):
    uri = find_bag(path)
    if uri is None:
        return None, "no metadata.yaml in the bag folder or its data/ subfolder"
    d = load(uri)
    if d["gt"]:
        ref, ref_name = d["gt"], "/sim_ground_truth_pose (exact)"
    elif d["amcl"]:
        ref, ref_name = d["amcl"], "/amcl_pose (fallback, localisation error included)"
    else:
        return None, "no /sim_ground_truth_pose or /amcl_pose"
    if not d["person"]:
        return None, "no /person_ground_truth"
    if not d["odom"]:
        return None, "no /odom"

    # Robot timeline: every reference sample, with odom twist and the true
    # person position interpolated onto it. All three are sim-time stamped.
    R = []
    for t, x, y in ref:
        vx, wz = interp(d["odom"], t, 2)
        px, py = interp(d["person"], t, 2)
        R.append((t, x, y, vx, wz, px, py, math.hypot(x - px, y - py)))

    t0 = next((r[0] for r in R if abs(r[3]) > MOVE_V), R[0][0])
    tg = next((r[0] for r in R
               if math.hypot(r[1] - GOAL[0], r[2] - GOAL[1]) < GOAL_TOL), None)
    t_end = tg if tg is not None else R[-1][0]
    win = [r for r in R if t0 <= r[0] <= t_end]
    if len(win) < 2:
        return None, "robot never moved"

    def dur(cond):
        return sum(b[0] - a[0] for a, b in zip(win, win[1:]) if cond(a))

    stopped = dur(lambda a: abs(a[3]) < STALL_V and abs(a[4]) < SPIN_W)
    spin = dur(lambda a: abs(a[3]) < STALL_V and abs(a[4]) >= SPIN_W)
    reverse = dur(lambda a: a[3] < -STALL_V)

    signs = [1 if r[4] > 0 else -1 for r in win if abs(r[4]) > FLIP_W]
    flips = sum(1 for a, b in zip(signs, signs[1:]) if a != b)

    closest = min(win, key=lambda r: r[7])
    # Second walker, if the trial had one: closest approach and pass side.
    p2 = None
    if d["person2"]:
        best = None
        for r in win:
            qx, qy = interp(d["person2"], r[0], 2)
            dist = math.hypot(r[1] - qx, r[2] - qy)
            if best is None or dist < best[0]:
                best = (dist, r[0], r[1], r[2], qy)
        p2 = {"min_centre_m": best[0], "t_min_s": best[1] - t0, "x_at_min": best[2],
              "side": "left (+y)" if best[3] > best[4] else "right (-y)",
              "person_y": best[4], "person1_y": closest[6]}
    y0 = win[0][2]
    max_dev = max(abs(r[2] - y0) for r in win)
    commit = next((r for r in win if abs(r[2] - y0) > COMMIT_DEV), None)
    path_len = sum(math.hypot(b[1] - a[1], b[2] - a[2]) for a, b in zip(win, win[1:]))

    # Person: when it started walking, and how fast it actually walked.
    p0 = d["person"][0]
    t_walk = next((p[0] for p in d["person"]
                   if math.hypot(p[1] - p0[1], p[2] - p0[2]) > PERSON_MOVE_M), None)
    person_speed = None
    if t_walk is not None:
        moving = [p for p in d["person"] if p[0] >= t_walk]
        seg = [(b[0] - a[0], math.hypot(b[1] - a[1], b[2] - a[2]))
               for a, b in zip(moving, moving[1:])]
        seg = [s for s in seg if s[1] > 1e-4]
        if seg:
            person_speed = sum(s[1] for s in seg) / sum(s[0] for s in seg)

    # Wall-stamped topics -> sim time through /clock.
    def to_sim(t_wall):
        return interp(d["clock"], t_wall, 1)[0] if d["clock"] else None

    def true_gap(t_sim):
        x, y = interp(ref, t_sim, 2)
        px, py = interp(d["person"], t_sim, 2)
        return math.hypot(x - px, y - py)

    first_det_range = det_to_kf = None
    if d["clock"] and t_walk is not None:
        fused = [to_sim(f[0]) for f in d["fused"]]
        fused = [t for t in fused if t >= t_walk]
        if fused:
            first_det_range = true_gap(fused[0])
            kf = [(to_sim(k[0]), k[1]) for k in d["kf"]]
            t_fast = next((t for t, s in kf if t >= fused[0] and s >= KF_SPEED), None)
            if t_fast is not None:
                det_to_kf = t_fast - fused[0]

    amcl_err = None
    if d["gt"] and d["amcl"]:
        errs = [math.hypot(a[1] - gx, a[2] - gy)
                for a in d["amcl"] if t0 <= a[0] <= t_end
                for gx, gy in [interp(d["gt"], a[0], 2)]]
        if errs:
            amcl_err = max(errs)

    rtf = None
    if len(d["clock"]) > 1:
        c = [k for k in d["clock"] if t0 <= k[1] <= t_end]
        if len(c) > 1 and c[-1][0] > c[0][0]:
            rtf = (c[-1][1] - c[0][1]) / (c[-1][0] - c[0][0])

    with open(os.path.join(path, "trajectory.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "x_ref", "y_ref", "x_person", "y_person", "vx", "wz", "d_person_1"])
        for r in R:
            w.writerow([f"{r[0]:.3f}", f"{r[1]:.4f}", f"{r[2]:.4f}", f"{r[5]:.4f}",
                        f"{r[6]:.4f}", f"{r[3]:.4f}", f"{r[4]:.4f}", f"{r[7]:.4f}"])

    return {
        "ref": ref_name,
        "missing": d["missing"],
        "reached": tg is not None,
        "time_s": t_end - t0,
        "path_m": path_len,
        "mean_vx": path_len / (t_end - t0) if t_end > t0 else 0.0,
        "min_centre_m": closest[7],
        "min_surface_m": closest[7] - PERSON_R - ROBOT_R,
        "t_min_s": closest[0] - t0,
        "x_at_min": closest[1],
        "side": "left (+y)" if closest[2] > closest[6] else "right (-y)",
        "p2": p2,
        "max_dev_m": max_dev,
        "commit_gap_m": commit[7] if commit else None,
        "stopped_s": stopped,
        "spin_s": spin,
        "reverse_s": reverse,
        "wz_flips": flips,
        "gap_at_start_m": true_gap(t0),
        "walk_minus_start_s": (t_walk - t0) if t_walk is not None else None,
        "person_speed": person_speed,
        "first_det_range_m": first_det_range,
        "det_to_kf_s": det_to_kf,
        "amcl_err_m": amcl_err,
        "plans": d["plans"],
        "rtf": rtf,
    }, None


def fmt(v, spec, unit=""):
    return "n/a" if v is None else f"{v:{spec}}{unit}"


def report(name, m):
    print(f"\n=== {name} ===")
    print(f"  reference pose              {m['ref']}")
    if m["missing"]:
        print(f"  topics missing from bag     {', '.join(m['missing'])}")
    print(f"  goal reached                {'yes' if m['reached'] else 'NO'}")
    print(f"  time to goal                {m['time_s']:.1f} s"
          f"{'' if m['reached'] else '  (to end of bag)'}")
    print(f"  path length / mean speed    {m['path_m']:.2f} m / {m['mean_vx']:.2f} m/s")
    print(f"  min centre distance         {m['min_centre_m']:.3f} m"
          f"  (t = {m['t_min_s']:.1f} s, robot x = {m['x_at_min']:.2f})")
    print(f"  min surface clearance       {m['min_surface_m']:+.3f} m")
    print(f"  robot passed on its         {m['side']}")
    if m.get("p2"):
        q = m["p2"]
        print(f"  person 2 min centre dist    {q['min_centre_m']:.3f} m"
              f"  (t = {q['t_min_s']:.1f} s, robot x = {q['x_at_min']:.2f})")
        print(f"  person 2 passed on its      {q['side']}")
        split = (m["side"] != q["side"])
        print(f"  walkers at y                {q['person1_y']:+.2f} / {q['person_y']:+.2f}"
              f"   robot went {'BETWEEN them' if split else 'around both'}")
    print(f"  max lateral deviation       {m['max_dev_m']:.2f} m")
    print(f"  gap when committed (>{COMMIT_DEV} m) {fmt(m['commit_gap_m'], '.2f', ' m')}")
    print(f"  stopped / spin / reverse    {m['stopped_s']:.1f} / {m['spin_s']:.1f} / "
          f"{m['reverse_s']:.1f} s")
    print(f"  wz sign flips               {m['wz_flips']}")
    print(f"  gap at robot start          {m['gap_at_start_m']:.2f} m")
    print(f"  person start - robot start  {fmt(m['walk_minus_start_s'], '+.1f', ' s')}")
    print(f"  person speed (GT)           {fmt(m['person_speed'], '.2f', ' m/s')}")
    print(f"  first detection range       {fmt(m['first_det_range_m'], '.2f', ' m')}")
    print(f"  detection -> KF {KF_SPEED} m/s      {fmt(m['det_to_kf_s'], '.2f', ' s')}")
    print(f"  max AMCL error              {fmt(m['amcl_err_m'], '.3f', ' m')}")
    print(f"  plans published             {m['plans']}")
    print(f"  real-time factor            {fmt(m['rtf'], '.2f')}")


def main():
    ap = argparse.ArgumentParser(description="Head-on trial metrics (condition F).")
    ap.add_argument("bags", nargs="+", help="bag directories (globs OK)")
    args = ap.parse_args()

    paths = []
    for b in args.bags:
        paths.extend(sorted(p for p in glob.glob(b) if os.path.isdir(p)) or [b])

    results = []
    for p in paths:
        m, err = analyse(p.rstrip("/"))
        if err:
            print(f"\n=== {os.path.basename(p.rstrip('/'))} ===\n  SKIPPED: {err}")
            continue
        report(os.path.basename(p.rstrip("/")), m)
        results.append(m)

    if len(results) < 2:
        return
    ok = [m for m in results if m["reached"]]
    print(f"\n=== Aggregate: goal reached in {len(ok)} / {len(results)} trials ===")
    if len(ok) < 2:
        return
    print(f"  {'metric':<22} {'mean':>8} {'sd':>8} {'min':>8} {'max':>8}   n")
    for key in ("time_s", "min_centre_m", "min_surface_m", "max_dev_m", "commit_gap_m",
                "stopped_s", "spin_s", "reverse_s", "wz_flips", "mean_vx",
                "first_det_range_m", "det_to_kf_s", "amcl_err_m", "person_speed", "rtf"):
        v = [m[key] for m in ok if m[key] is not None]
        if len(v) < 2:
            continue
        print(f"  {key:<22} {statistics.mean(v):8.3f} {statistics.stdev(v):8.3f} "
              f"{min(v):8.3f} {max(v):8.3f}   {len(v)}")
    sides = [m["side"] for m in ok]
    print(f"  pass side: " + ", ".join(f"{s} x{sides.count(s)}" for s in sorted(set(sides))))


if __name__ == "__main__":
    main()
