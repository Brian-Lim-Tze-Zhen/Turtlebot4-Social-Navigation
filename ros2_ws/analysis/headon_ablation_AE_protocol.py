#!/usr/bin/env python3
"""Read-only: start protocol of the A-E head-on ablation trials, from the bags."""
import sys, math, glob, os, statistics
sys.path.insert(0, "/ws/analysis")
import analyse_avoidance as aa
rows = []
for d in sorted(glob.glob("/ws/bags/[a-e]_*_t0*/data")):
    name = os.path.basename(os.path.dirname(d))
    if "_1.2_" in name: continue
    data = aa.read_bag(d)
    m2o = aa.extract_map_to_odom(data["/tf"] + data["/tf_static"])
    robot = aa.robot_path_in_map(data["/odom"], m2o)
    per = aa.person_track(data["/person_ground_truth"])
    if not robot or not per: continue
    # first times of motion
    def first_move(track, thr):
        x0, y0 = track[0][1], track[0][2]
        for t, x, y in track:
            if math.hypot(x - x0, y - y0) > thr: return t
        return None
    tr = first_move(robot, 0.05); tp = first_move(per, 0.05)
    # robot speed when the person starts
    def speed_at(track, t):
        best = min(range(1, len(track)), key=lambda i: abs(track[i][0] - t)); a, b = track[max(0, best - 10)], track[min(len(track) - 1, best + 10)]
        return math.hypot(b[1] - a[1], b[2] - a[2]) / (b[0] - a[0]) if b[0] > a[0] else 0.0
    def at(track, t):
        i = min(range(len(track)), key=lambda i: abs(track[i][0] - t)); return track[i]
    rp = at(robot, tp) if tp else None
    rows.append((name, robot[0][1:], robot[-1][1:], per[0][1:], per[-1][1:], (tp - tr) if tr and tp else None, speed_at(robot, tp) if tp else None, rp[1:] if rp else None,
                 math.hypot(rp[1] - at(per, tp)[1], rp[2] - at(per, tp)[2]) if rp else None))
    print(f"{name}: robot start ({robot[0][1]:.2f},{robot[0][2]:.2f}) end ({robot[-1][1]:.2f},{robot[-1][2]:.2f}); person ({per[0][1]:.2f},{per[0][2]:.2f}) -> ({per[-1][1]:.2f},{per[-1][2]:.2f}); person starts {rows[-1][5]:+.1f} s after the robot; robot then at ({rp[1]:.2f},{rp[2]:.2f}) moving {rows[-1][6]:.2f} m/s; gap {rows[-1][8]:.2f} m", flush=True)
f = lambda v: f"{statistics.mean(v):.2f} +/- {statistics.pstdev(v):.2f} (min {min(v):.2f}, max {max(v):.2f})"
print("--- all", len(rows), "trials")
print("robot start x:", f([r[1][0] for r in rows]), " y:", f([r[1][1] for r in rows]))
print("robot end   x:", f([r[2][0] for r in rows]), " y:", f([r[2][1] for r in rows]))
print("person start x:", f([r[3][0] for r in rows]), " end x:", f([r[4][0] for r in rows]))
print("person starts after robot (s):", f([r[5] for r in rows]))
print("robot speed when the person starts (m/s):", f([r[6] for r in rows]))
print("robot x when the person starts:", f([r[7][0] for r in rows]), " gap to person (m):", f([r[8] for r in rows]))
