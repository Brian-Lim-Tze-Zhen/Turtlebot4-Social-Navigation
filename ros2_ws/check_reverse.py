#!/usr/bin/env python3
"""
check_reverse.py

Reports whether the robot reversed during a run, and if so when and how
far it travelled backwards.

WHY THIS MATTERS
----------------
Reversing is geometrically effective in a head-on encounter - backing
away opens the gap immediately, whereas a differential base steering
around the person has to arc, which takes time. So MPPI will use it if
vx_min allows it, and min_distance improves as a result.

It is also not what a person does. Pedestrians yield sideways; they do
not walk backwards to let someone past. A clearance figure obtained by
reversing is worth less in a social-navigation result than the same
figure obtained with forward motion only, so the two need to be
distinguished rather than reported together.

Reads /odom, so it measures what the robot ACTUALLY did, not what the
controller asked for.

USAGE
    python3 check_reverse.py --bag bags/fusion_ta_try_01/data
    python3 check_reverse.py --bag bags/*/data          # compare runs
"""

import argparse
import glob
import sys

try:
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError as e:
    sys.exit(
        f"Missing ROS 2 Python packages ({e}).\n"
        "Run inside the container with /opt/ros/jazzy sourced."
    )


# Odometry noise sits well under this even when stationary, so anything
# past it is a commanded reversal rather than drift.
REVERSE_THRESHOLD = -0.02   # m/s


def stamp_to_sec(h):
    return h.sec + h.nanosec * 1e-9


def read_odom(path):
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=path, storage_id="mcap"),
        ConverterOptions("", ""),
    )
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if "/odom" not in types:
        return []
    msg_cls = get_message(types["/odom"])

    out = []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic != "/odom":
            continue
        m = deserialize_message(data, msg_cls)
        out.append((stamp_to_sec(m.header.stamp), m.twist.twist.linear.x))
    out.sort(key=lambda r: r[0])
    return out


def analyse(path):
    odom = read_odom(path)
    if not odom:
        return None

    t0 = odom[0][0]
    reversing = [(t - t0, v) for t, v in odom if v < REVERSE_THRESHOLD]

    if not reversing:
        return {
            "n": len(odom), "frac": 0.0, "peak": 0.0,
            "dist": 0.0, "first": None, "last": None,
        }

    # Integrate the reverse segments to get distance actually travelled
    # backwards. Sample spacing varies, so use the gap to the next
    # sample rather than assuming a fixed rate.
    dist = 0.0
    for i, (t, v) in enumerate(odom[:-1]):
        if v < REVERSE_THRESHOLD:
            dist += abs(v) * (odom[i + 1][0] - t)

    return {
        "n": len(odom),
        "frac": len(reversing) / len(odom),
        "peak": min(v for _, v in reversing),
        "dist": dist,
        "first": reversing[0][0],
        "last": reversing[-1][0],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", nargs="+", required=True)
    args = ap.parse_args()

    paths = []
    for p in args.bag:
        paths.extend(sorted(glob.glob(p)) or [p])

    print(f"{'trial':<32} {'rev %':>7} {'peak':>8} {'back m':>8} "
          f"{'from':>7} {'to':>7}")
    print("-" * 76)

    for p in paths:
        name = p.rstrip("/").split("/")[-2] if p.rstrip("/").endswith("data") \
            else p.rstrip("/").split("/")[-1]
        try:
            r = analyse(p)
        except Exception as e:
            print(f"{name:<32} FAILED: {e}")
            continue
        if r is None:
            print(f"{name:<32} no /odom in bag")
            continue
        if r["frac"] == 0.0:
            print(f"{name:<32} {'0.0':>7} {'—':>8} {'—':>8} {'—':>7} {'—':>7}")
        else:
            print(f"{name:<32} {100 * r['frac']:>6.1f}% {r['peak']:>8.3f} "
                  f"{r['dist']:>8.3f} {r['first']:>7.1f} {r['last']:>7.1f}")

    print()
    print("rev %  fraction of odom samples with linear.x < -0.02 m/s")
    print("peak   most negative velocity reached (m/s)")
    print("back m distance actually travelled backwards")
    print("from/to  seconds from bag start, first and last reverse sample")


if __name__ == "__main__":
    main()
