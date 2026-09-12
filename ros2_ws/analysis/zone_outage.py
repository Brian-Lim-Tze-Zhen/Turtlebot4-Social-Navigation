"""Report /social_group_cloud outages per trial.

The zone can vanish mid-run when the queue hold-open goes dormant
(camera unable to re-confirm the queue for QUEUE_HOLD_TIMEOUT). Reported
alongside the navigation metrics rather than used to discard trials, but
it must be reported: a run where the zone was absent for a third of the
approach is not evidence about the zone.
"""
import glob
import os
import sys

import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2

for d in sorted(sys.argv[1:]):
    hits = glob.glob(os.path.join(d, "**", "metadata.yaml"), recursive=True)
    if not hits:
        print(f"{os.path.basename(d)}: no metadata.yaml")
        continue
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=os.path.dirname(hits[0]), storage_id=""),
           rosbag2_py.ConverterOptions("", ""))
    t0 = None
    t = None
    prev = None
    marks = []
    n = 0
    while r.has_next():
        tp, data, t = r.read_next()
        if t0 is None:
            t0 = t
        if tp != "/social_group_cloud":
            continue
        w = deserialize_message(data, PointCloud2).width
        el = (t - t0) / 1e9
        n += 1
        if prev is not None and (w == 0) != (prev == 0):
            marks.append((el, w == 0))
        prev = w
    spans = []
    for i in range(len(marks) - 1):
        if marks[i][1] and not marks[i + 1][1]:
            spans.append((marks[i][0], marks[i + 1][0] - marks[i][0]))
    total = sum(s[1] for s in spans)
    dur = (t - t0) / 1e9 if t0 else 0.0
    desc = ", ".join(f"{s:.1f}s+{d2:.1f}s" for s, d2 in spans) or "none"
    pct = 100 * total / dur if dur else 0.0
    print(f"{os.path.basename(d):<32} n={n:<4} outage={total:5.1f}s "
          f"({pct:4.1f}%)  [{desc}]")
