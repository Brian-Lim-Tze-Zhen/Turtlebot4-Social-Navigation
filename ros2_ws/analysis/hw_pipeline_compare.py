#!/usr/bin/env python3
"""hw_pipeline_compare.py <replay dir> - replayed perception output against the
output the robot recorded at the time (same bag, same clock).

<replay dir>/in/<bag>/*.db3 is the hardware bag (original outputs),
<replay dir>/out/<bag>.csv the replay (hw_pipeline_replay.sh).
"""
import bisect
import collections
import glob
import math
import os
import sqlite3
import statistics
import struct
import sys

TOPICS = ["/person_positions_map", "/camera_ray_clusters", "/person_positions_fused"]


def original(bag_dir):
    out = collections.defaultdict(list)
    for db in glob.glob(os.path.join(bag_dir, "*.db3")):
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        tp = dict(con.execute("select name, id from topics"))
        t0, t1 = con.execute("select min(timestamp), max(timestamp) from messages").fetchone()
        for t in TOPICS:
            if t in tp:
                for ts, d in con.execute("select timestamp, data from messages where topic_id=?", (tp[t],)):
                    n = struct.unpack_from("<I", d, 4)[0]
                    out[t].append((ts * 1e-9, bytes(d[8:8 + n - 1]).decode("utf-8", "replace")))
        con.close()
    return out, (t1 - t0) * 1e-9


def replayed(path):
    out = collections.defaultdict(list)
    for line in open(path):
        topic, t, rx, ry, ryaw, data = line.rstrip("\n").split(",", 5)
        out[topic].append((float(t), data, (float(rx), float(ry), float(ryaw))))
    return out


def xy(topic, data):
    p = data.split(",")
    try:
        return int(float(p[0])), float(p[2]), float(p[3])
    except (ValueError, IndexError):
        return None


def main(root):
    for csvp in sorted(glob.glob(os.path.join(root, "out", "*.csv"))):
        name = os.path.basename(csvp)[:-4]
        orig, dur = original(os.path.join(root, "in", name))
        rep = replayed(csvp)
        print(f"\n=== {name}  ({dur:.0f} s) ===")
        print(f"  {'topic':26s} {'recorded':>9s} {'replay':>7s}")
        for t in TOPICS:
            print(f"  {t:26s} {len(orig[t]):9d} {len(rep[t]):7d}")
        # The person position the pipeline hands on: clusters if recorded, else fused.
        for t in ("/camera_ray_clusters", "/person_positions_fused"):
            R = [(ts, xy(t, d), pose) for ts, d, pose in rep[t] if xy(t, d)]
            if not R:
                continue
            rng = [math.hypot(p[1] - pose[0], p[2] - pose[1]) for _, p, pose in R if not math.isnan(pose[0])]
            ids = sorted({p[0] for _, p, _ in R})
            gaps = [b[0] - a[0] for a, b in zip(R, R[1:])]
            print(f"  replay {t}: ids {ids[:8]}{'...' if len(ids) > 8 else ''}, "
                  f"longest gap {max(gaps) if gaps else 0:.2f} s")
            if rng:
                print(f"    range from the robot: first {rng[0]:.2f} m, median {statistics.median(rng):.2f} m, "
                      f"min {min(rng):.2f} m, max {max(rng):.2f} m, "
                      f"std {statistics.pstdev(rng):.3f} m")
            O = [(ts, xy(t, d)) for ts, d in orig[t] if xy(t, d)]
            if O:
                ots = [o[0] for o in O]
                diffs = []
                for ts, p, _ in R:
                    j = bisect.bisect_left(ots, ts)
                    c = min((k for k in (j - 1, j) if 0 <= k < len(O)), key=lambda k: abs(ots[k] - ts))
                    if abs(ots[c] - ts) <= 0.25:
                        diffs.append(math.hypot(p[1] - O[c][1][1], p[2] - O[c][1][2]))
                if diffs:
                    ds = sorted(diffs)
                    print(f"    against the recorded position at the same time ({len(diffs)} pairs): "
                          f"median {ds[len(ds) // 2]:.3f} m, p90 {ds[int(0.9 * (len(ds) - 1))]:.3f} m, max {ds[-1]:.3f} m")
                print(f"    first message: recorded t+{O[0][0] - min(o[0] for v in orig.values() for o in v):.1f} s"
                      if False else f"    first message at: recorded {O[0][0] % 1000:.1f}, replay {R[0][0] % 1000:.1f} (s, same clock)")
            break
        conf = [float(d.split(",")[1]) for _, d, _ in rep["/person_positions_map"] if len(d.split(",")) > 1]
        if conf:
            print(f"  replay YOLO confidence: median {statistics.median(conf):.2f}, min {min(conf):.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
