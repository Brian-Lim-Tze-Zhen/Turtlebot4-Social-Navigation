#!/usr/bin/env python3
"""Read-only check of the Kalman predictor in the A-E head-on ablation bags (Section 4.3.1).

For each bag:
  - speed of the walker as published by the predictor (smoothed velocity, fields 4 and 5 of
    /predicted_person_positions), against the true speed from /person_ground_truth
  - lag of the published position: the time shift that minimises the mean distance between the
    published position at time t and the true position at time t - lag (sweep 0 to 1.2 s)
  - message rate of the predictor output and the time from the last prediction to the end of
    the walker's path
All times are simulation times: every bag receive time is converted with the pairs of receive
time and header stamp of the /odom messages of the same bag (linear interpolation).

Usage: headon_ablation_kf_check.py <bag_dir> [<bag_dir> ...]   (bag_dir contains data/)
"""
import sys, os, math, bisect, statistics
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

WANT = ("/predicted_person_positions", "/person_positions_map", "/person_ground_truth", "/odom")


def read(bag):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag, storage_id=""), rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    r.set_filter(rosbag2_py.StorageFilter(topics=[w for w in WANT if w in types]))
    out = {w: [] for w in WANT}
    while r.has_next():
        tp, data, t = r.read_next()
        out[tp].append((t * 1e-9, deserialize_message(data, get_message(types[tp]))))
    return out


def interp(gt, t):
    ts = [g[0] for g in gt]
    i = bisect.bisect_left(ts, t)
    if i <= 0 or i >= len(gt):
        return None
    (t0, x0, y0), (t1, x1, y1) = gt[i - 1], gt[i]
    if t1 - t0 <= 0 or t1 - t0 > 1.5:
        return None
    a = (t - t0) / (t1 - t0)
    return x0 + a * (x1 - x0), y0 + a * (y1 - y0)


def analyse(bagdir):
    name = os.path.basename(bagdir.rstrip("/"))
    d = read(os.path.join(bagdir, "data"))
    od = d["/odom"]
    if len(od) < 10:
        return None
    ow = [t for t, _ in od]
    osim = [m.header.stamp.sec + m.header.stamp.nanosec * 1e-9 for _, m in od]
    rtf = (osim[-1] - osim[0]) / (ow[-1] - ow[0])

    def sim(t):
        i = min(max(bisect.bisect_left(ow, t), 1), len(ow) - 1)
        a = (t - ow[i - 1]) / (ow[i] - ow[i - 1]) if ow[i] > ow[i - 1] else 0.0
        return osim[i - 1] + a * (osim[i] - osim[i - 1])

    gt = [(sim(t), m.poses[0].position.x, m.poses[0].position.y) for t, m in d["/person_ground_truth"] if m.poses]
    # walking phase of the walker: from the first to the last change of position
    mv = [i for i in range(1, len(gt)) if math.hypot(gt[i][1] - gt[i - 1][1], gt[i][2] - gt[i - 1][2]) > 0.02]
    if not mv:
        return None
    w0, w1 = gt[mv[0] - 1][0], gt[mv[-1]][0]
    walk = [g for g in gt if w0 <= g[0] <= w1]
    true_speed = math.hypot(walk[-1][1] - walk[0][1], walk[-1][2] - walk[0][2]) / (walk[-1][0] - walk[0][0])
    pred = []
    for t, m in d["/predicted_person_positions"]:
        for line in m.data.strip().split("\n"):
            f = line.split(",")
            try:
                pred.append((sim(t), f[0], float(f[2]), float(f[3]), math.hypot(float(f[4]), float(f[5]))))
            except (ValueError, IndexError):
                pass
    pred = [p for p in pred if w0 <= p[0] <= w1]
    res = {"name": name, "rtf": rtf, "true_speed": true_speed, "n_pred": len(pred),
           "n_det": sum(1 for t, _ in d["/person_positions_map"] if w0 <= sim(t) <= w1)}
    if len(pred) < 8:
        return res
    span = pred[-1][0] - pred[0][0]
    res["span_s"] = span
    res["rate_hz"] = (len(pred) - 1) / span if span > 0 else float("nan")
    speeds = [p[4] for p in pred]
    res["speed_max"] = max(speeds)
    # plateau: predictions more than 1 s (simulation time) after the first one
    plat = [p[4] for p in pred if p[0] - pred[0][0] >= 1.0]
    if plat:
        res["speed_median"] = statistics.median(plat)
        res["n_plat"] = len(plat)
    best = None
    for k in range(0, 25):
        lag = 0.05 * k
        errs = []
        for p in pred:
            g = interp(gt, p[0] - lag)
            if g:
                errs.append(math.hypot(p[2] - g[0], p[3] - g[1]))
        if len(errs) >= 0.8 * len(pred):
            e = sum(errs) / len(errs)
            if k == 0:
                res["err_lag0"] = e
            if best is None or e < best[1]:
                best = (lag, e)
    if best:
        res["lag_s"], res["err_at_lag"] = best
    return res


def ms(v):
    return f"{statistics.mean(v):.2f} +/- {statistics.stdev(v):.2f}" if len(v) > 1 else (f"{v[0]:.2f}" if v else "-")


def main():
    rows = [r for r in (analyse(b) for b in sys.argv[1:]) if r]
    for r in rows:
        if "lag_s" not in r:
            print(f"{r['name']}: real-time factor {r['rtf']:.2f}; true speed {r['true_speed']:.2f} m/s; "
                  f"{r['n_det']} detections, {r['n_pred']} predictions while the walker moved (too few to analyse)")
            continue
        print(f"{r['name']}: real-time factor {r['rtf']:.2f}; true speed {r['true_speed']:.2f} m/s; "
              f"{r['n_det']} detections, {r['n_pred']} predictions over {r['span_s']:.1f} s ({r['rate_hz']:.1f} Hz); "
              f"published speed 1 s after the first prediction: median {r.get('speed_median', float('nan')):.2f} m/s "
              f"(n={r.get('n_plat', 0)}), max {r['speed_max']:.2f} m/s; "
              f"position lag {r['lag_s']:.2f} s (mean error {r['err_lag0']:.2f} m without shift, {r['err_at_lag']:.2f} m at that lag)")
    ok = [r for r in rows if "lag_s" in r and "speed_median" in r]
    print(f"--- {len(ok)} of {len(rows)} trials with at least 8 predictions")
    if ok:
        print("true speed of the walker (m/s):", ms([r["true_speed"] for r in ok]))
        print("published speed, median per trial (m/s):", ms([r["speed_median"] for r in ok]),
              f"(min {min(r['speed_median'] for r in ok):.2f}, max {max(r['speed_median'] for r in ok):.2f})")
        print("position lag (s):", ms([r["lag_s"] for r in ok]),
              f"(min {min(r['lag_s'] for r in ok):.2f}, max {max(r['lag_s'] for r in ok):.2f})")
        print("mean position error without shift (m):", ms([r["err_lag0"] for r in ok]),
              "; at the best lag (m):", ms([r["err_at_lag"] for r in ok]))
        print("predictor output rate (Hz):", ms([r["rate_hz"] for r in ok]))
        print("time the walker was tracked (s):", ms([r["span_s"] for r in ok]))
        print("real-time factor:", ms([r["rtf"] for r in rows]))


if __name__ == "__main__":
    main()
