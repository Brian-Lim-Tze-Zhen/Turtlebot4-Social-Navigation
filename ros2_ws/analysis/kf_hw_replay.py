#!/usr/bin/env python3
"""kf_hw_replay.py - replay the human Kalman filter on HARDWARE bags, offline.

The filter class (HumanTrackKF in camera_lidar/human_kf_predictor_lidar.py) is
the same file on the robot and in the sim, so it is imported and fed the
/person_positions_fused messages a real run recorded, at their recorded times,
with the robot's rotation gate from /turtlebot4/odom. Only Q and R are swapped
per setting. The hardware bags are opened read-only; nothing is written there.

Two steps (the hardware repo is not mounted in the sim container):

  1. on the host:
       python3 kf_hw_replay.py extract <hardware bags dir> <out dir> [name filter]
     writes <out dir>/<bag>.csv with rows "fused,<t>,<data...>" and "wz,<t>,<rad/s>"

  2. where numpy is available (the container):
       python3 kf_hw_replay.py replay <out dir>

There is no ground truth on the robot, so the measures are self-consistency:
  pred err h   predicted position at t+h against the measured position of the
               same track at t+h (walking samples only)
  onset lag    for a track that walks, time from its first measurement until
               the published speed reaches 0.8 m/s
  flips        heading changes > 60 deg between consecutive outputs while walking
  still speed  published speed for a person who is standing (should be ~0)
  NIS          normalised innovation squared, y' S^-1 y; ~2 for a consistent
               2-D filter, > 2 = the filter trusts itself too much
"""
import bisect
import collections
import glob
import math
import os
import sqlite3
import struct
import sys

ODOM_TOPIC = "/turtlebot4/odom"
FUSED_TOPIC = "/person_positions_fused"
ROT_GATE = 0.8          # rad/s, the node's rot_gate_threshold default
HORIZONS = (1.0, 2.0, 3.0)
WALK_SPEED = 0.6        # m/s measured speed above this = walking sample
STILL_SPEED = 0.15      # m/s measured speed below this = standing sample
SPEED_WIN = 1.5         # s; window for the measured speed
# The camera-ray position jitters (a standing person can read 0.3-1.2 m/s
# between two messages), so "walking" needs a steady displacement: net travel
# over the window, and that travel mostly in one direction.
WALK_STRAIGHT = 0.7     # net displacement / summed step lengths
MATCH_TOL = 0.15        # s; a measurement this close to t+h counts as "at t+h"


# ------------------------------------------------------------------ extract
def cdr_string(buf, off):
    off = (off + 3) & ~3
    n = struct.unpack_from("<I", buf, off)[0]
    return buf[off + 4:off + 4 + n - 1].decode("utf-8", "replace"), off + 4 + n


def odom_wz(buf):
    """Angular z of nav_msgs/Odometry from its CDR bytes."""
    off = 4 + 8                                   # encapsulation + stamp
    _, off = cdr_string(buf, off)                 # header.frame_id
    _, off = cdr_string(buf, off)                 # child_frame_id
    off = 4 + ((off - 4 + 7) & ~7)                # doubles align to 8 after the 4-byte header
    off += 8 * (7 + 36)                           # pose + covariance
    return struct.unpack_from("<6d", buf, off)[5]


def extract(bags_dir, out_dir, name_filter=""):
    os.makedirs(out_dir, exist_ok=True)
    for db in sorted(glob.glob(os.path.join(bags_dir, "*", "**", "*.db3"), recursive=True)):
        # One CSV per bag: folders such as perception/ and trials/ hold several.
        parts = os.path.relpath(os.path.dirname(db), bags_dir).split(os.sep)
        if parts[-1] == "bag":
            parts = parts[:-1]
        if parts[0] in ("replays", "replay_logs"):
            continue                      # re-processed output, not a robot run
        name = "__".join(parts)
        if name_filter and name_filter not in name:
            continue
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        topics = dict(con.execute("select name, id from topics"))
        if FUSED_TOPIC not in topics:
            con.close()
            continue
        rows = []
        for t, data in con.execute(
                "select timestamp, data from messages where topic_id=?", (topics[FUSED_TOPIC],)):
            s, _ = cdr_string(bytes(data), 4)
            rows.append((t * 1e-9, "fused", s))
        if ODOM_TOPIC in topics:
            for t, data in con.execute(
                    "select timestamp, data from messages where topic_id=?", (topics[ODOM_TOPIC],)):
                rows.append((t * 1e-9, "wz", f"{odom_wz(bytes(data)):.4f}"))
        con.close()
        if not any(r[1] == "fused" for r in rows):
            continue
        rows.sort()
        with open(os.path.join(out_dir, name + ".csv"), "w") as f:
            for t, kind, s in rows:
                f.write(f"{kind},{t:.6f},{s}\n")
        print(f"{name}: {sum(1 for r in rows if r[1] == 'fused')} fused msgs")


# ------------------------------------------------------------------- replay
def load(path):
    ev = []
    for line in open(path):
        kind, t, rest = line.rstrip("\n").split(",", 2)
        ev.append((float(t), kind, rest))
    ev.sort()
    return ev


def replay_bag(ev, KF, np, q_pos, q_vel, r):
    """Returns per-track lists of output samples and the NIS values."""
    tracks, out, nis = {}, collections.defaultdict(list), []
    wz = 0.0
    for t, kind, rest in ev:
        if kind == "wz":
            wz = float(rest)
            continue
        p = rest.split(",")
        try:
            tid, mx, my = int(float(p[0])), float(p[2]), float(p[3])
        except (ValueError, IndexError):
            continue
        source = p[11] if len(p) > 11 else "camera_confirmed"
        if tid not in tracks:
            k = KF(mx, my, t)
            k.Q = np.diag([q_pos, q_pos, q_vel, q_vel])
            k.R = np.eye(2) * r
            tracks[tid] = k
            out[tid].append((t, mx, my, mx, my, 0.0, 0.0, source))
            continue
        k = tracks[tid]
        # Innovation as the filter itself will compute it (same F, Q, R).
        dt_raw = t - k.last_time
        if dt_raw <= 1.5:
            dt = dt_raw if 0.0 < dt_raw <= 1.0 else 0.1
            F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
            xp = F @ k.x
            Pp = F @ k.P @ F.T + k.Q
            lidar = source in ("lidar_only", "lidar_hold")
            S = k.H @ Pp @ k.H.T + (k.R * k.lidar_only_noise_scale if lidar else k.R)
            y = np.array([[mx], [my]]) - k.H @ xp
            if k.update_count >= 5:
                nis.append(float((y.T @ np.linalg.inv(S) @ y)[0, 0]))
        k.update(mx, my, t, freeze_velocity=abs(wz) > ROT_GATE,
                 is_lidar_only=source in ("lidar_only", "lidar_hold"))
        x, yy, vx, vy, _, _, _, _ = k.predict_future(1.0)
        out[tid].append((t, mx, my, x, yy, vx, vy, source))
    return out, nis


def measured_speed(samples, i):
    """Finite-difference speed of the MEASUREMENTS around sample i."""
    t = samples[i][0]
    lo = i
    while lo > 0 and t - samples[lo - 1][0] <= SPEED_WIN / 2:
        lo -= 1
    hi = i
    while hi < len(samples) - 1 and samples[hi + 1][0] - t <= SPEED_WIN / 2:
        hi += 1
    dt = samples[hi][0] - samples[lo][0]
    if dt < SPEED_WIN / 2:
        return None, 0.0, 0.0
    net = math.hypot(samples[hi][1] - samples[lo][1], samples[hi][2] - samples[lo][2])
    steps = sum(math.hypot(b[1] - a[1], b[2] - a[2])
                for a, b in zip(samples[lo:hi], samples[lo + 1:hi + 1]))
    cx = sum(q[1] for q in samples[lo:hi + 1]) / (hi - lo + 1)
    cy = sum(q[2] for q in samples[lo:hi + 1]) / (hi - lo + 1)
    return net / dt, (net / steps if steps > 1e-6 else 0.0), math.hypot(
        samples[i][1] - cx, samples[i][2] - cy)


def score(out, nis):
    err = {h: [] for h in HORIZONS}
    err0 = {h: [] for h in HORIZONS}     # same samples, prediction = "stays where it is"
    still_v, lags, flips, walk_n, jitter = [], [], 0, 0, []
    for samples in out.values():
        ts = [s[0] for s in samples]
        ms = [measured_speed(samples, i) for i in range(len(samples))]
        speeds = [m[0] if (m[0] is None or m[0] < STILL_SPEED or m[1] >= WALK_STRAIGHT) else None
                  for m in ms]
        prev_heading = None
        for i, (t, mx, my, x, y, vx, vy, _src) in enumerate(samples):
            sp = speeds[i]
            if sp is None:
                continue
            v = math.hypot(vx, vy)
            if sp < STILL_SPEED:
                still_v.append(v)
                jitter.append(ms[i][2])
                prev_heading = None
            elif sp > WALK_SPEED:
                walk_n += 1
                for h in HORIZONS:
                    j = bisect.bisect_left(ts, t + h)
                    best = min((c for c in (j - 1, j) if 0 <= c < len(ts)),
                               key=lambda c: abs(ts[c] - (t + h)), default=None)
                    if best is not None and abs(ts[best] - (t + h)) <= MATCH_TOL:
                        err[h].append(math.hypot(x + vx * h - samples[best][1],
                                                 y + vy * h - samples[best][2]))
                        err0[h].append(math.hypot(x - samples[best][1], y - samples[best][2]))
                if v > 0.2:
                    heading = math.atan2(vy, vx)
                    if prev_heading is not None:
                        d = abs(math.atan2(math.sin(heading - prev_heading),
                                           math.cos(heading - prev_heading)))
                        if d > math.radians(60):
                            flips += 1
                    prev_heading = heading
        # Onset lag: a track that walks at some point; from its first
        # measurement to the first published speed >= 0.8 m/s.
        walk = [i for i, s in enumerate(speeds) if s is not None and s > 0.8]
        if len(walk) >= 5:
            t_first_walk = samples[walk[0]][0]
            hit = next((s[0] for s in samples[walk[0]:] if math.hypot(s[5], s[6]) >= 0.8), None)
            if hit is not None:
                lags.append(hit - t_first_walk)
    return err, still_v, lags, flips, walk_n, nis, err0, jitter


def med(v):
    v = sorted(v)
    return v[len(v) // 2] if v else float("nan")


def replay(data_dir):
    import numpy as np
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(
        here, "..", "src", "social_perception", "social_perception", "camera_lidar"))
    from human_kf_predictor_lidar import HumanTrackKF

    settings = [  # (label, q_pos, q_vel, R)
        ("before 2 Oct", 0.05, 0.05, 0.10),
        ("CURRENT", 0.02, 0.15, 0.10),
        ("q_vel 0.08", 0.02, 0.08, 0.10),
        ("q_vel 0.30", 0.02, 0.30, 0.10),
        ("q_pos 0.01 q_vel 0.30", 0.01, 0.30, 0.10),
        ("q_pos 0.005", 0.005, 0.15, 0.10),
        ("R 0.05", 0.02, 0.15, 0.05),
        ("R 0.03", 0.02, 0.15, 0.03),
        ("R 0.03 q_pos 0.005", 0.005, 0.15, 0.03),
        ("R 0.20", 0.02, 0.15, 0.20),
    ]
    allbags = {os.path.basename(p)[:-4]: load(p)
               for p in sorted(glob.glob(os.path.join(data_dir, "*.csv")))}
    groups = [
        ("Head-on runs on the robot, 25 Sep - 2 Oct (a person walks at the robot)",
         lambda n: n.startswith(("headon_real_F", "headon_F_", "headon_cloud_off", "headon_prediction"))),
        ("Standing people (conversation, narrow corridor, stationary protocols)",
         lambda n: n.startswith(("conversation", "narrow_corridor", "F_narrow_pass", "F_wide_around",
                                 "proto_A_stationary", "proto_F_robotmoving_personstill",
                                 "proto_H_robotmove_still", "proto_H2_", "proto_H3_robotmove"))),
        ("All bags", lambda n: True),
    ]
    for title, pick in groups:
        bags = {n: ev for n, ev in allbags.items() if pick(n)}
        print(f"\n=== {title}: {len(bags)} bags ===")
        print(f"{'setting':22s} {'q_pos':>6s} {'q_vel':>5s} {'R':>5s} | "
              f"{'err 1s':>6s} {'err 2s':>6s} {'err 3s':>6s} | {'lag':>5s} {'(n)':>5s} | "
              f"{'flips':>5s} | {'still v':>7s} {'p95':>6s} | {'NIS':>5s}")
        for label, qp, qv, r in settings:
            err = {h: [] for h in HORIZONS}
            err0 = {h: [] for h in HORIZONS}
            still, lags, flips, walk_n, nis, jit = [], [], 0, 0, [], []
            for ev in bags.values():
                e, sv, lg, fl, wn, ni, e0, jt = score(*replay_bag(ev, HumanTrackKF, np, qp, qv, r))
                for h in HORIZONS:
                    err[h] += e[h]
                    err0[h] += e0[h]
                still += sv
                lags += lg
                flips += fl
                walk_n += wn
                nis += ni
                jit += jt
            st = sorted(still)
            p95 = st[int(0.95 * (len(st) - 1))] if st else float("nan")
            print(f"{label:22s} {qp:6.3f} {qv:5.2f} {r:5.2f} | "
                  f"{med(err[1.0]):6.3f} {med(err[2.0]):6.3f} {med(err[3.0]):6.3f} | "
                  f"{med(lags):5.2f} {len(lags):5d} | {flips:5d} | "
                  f"{(sum(still) / len(still) if still else float('nan')):7.3f} {p95:6.3f} | "
                  f"{(sum(nis) / len(nis) if nis else float('nan')):5.2f}")
        print(f"  walking samples {walk_n} (with a measurement at t+1/2/3 s: "
              f"{len(err[1.0])}/{len(err[2.0])}/{len(err[3.0])}), standing samples {len(still)}")
        print(f"  if the prediction were 'stays where it is': "
              f"{med(err0[1.0]):.3f} / {med(err0[2.0]):.3f} / {med(err0[3.0]):.3f} m at 1 / 2 / 3 s")
        if jit:
            js = sorted(jit)
            print(f"  measurement scatter of a standing person about its 1.5 s mean: "
                  f"median {js[len(js) // 2]:.3f} m, p95 {js[int(0.95 * (len(js) - 1))]:.3f} m")
    print("\nerr = median distance between predicted and later measured position, walking samples;")
    print("lag = median s from the start of walking to a published speed of 0.8 m/s; still v in m/s.")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "extract":
        extract(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "")
    elif len(sys.argv) == 3 and sys.argv[1] == "replay":
        replay(sys.argv[2])
    else:
        sys.exit(__doc__)
