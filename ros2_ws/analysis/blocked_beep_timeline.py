#!/usr/bin/env python3
"""blocked_beep_timeline.py <bag/data> - robot motion, beeps and block events over time.

Time is sim seconds from the first robot motion in the bag (the goal start). One robot line every 2 s;
every /cmd_audio (beep) and /blocked_person_event message is listed.
"""
import math
import sys

from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message

r = SequentialReader()
r.open(StorageOptions(uri=sys.argv[1], storage_id='mcap'), ConverterOptions('', ''))
ty = {t.name: t.type for t in r.get_all_topics_and_types()}
st, t0, last, p1x, first = 0.0, None, -9.0, None, None
while r.has_next():
    tp, d, _ = r.read_next()
    if tp == '/clock':
        m = deserialize_message(d, get_message(ty[tp]))
        st = m.clock.sec + m.clock.nanosec * 1e-9
    elif tp == '/person_ground_truth':
        m = deserialize_message(d, get_message(ty[tp]))
        p1x = m.poses[0].position.x
    elif tp == '/sim_ground_truth_pose':
        m = deserialize_message(d, get_message(ty[tp]))
        p, q = m.pose.pose.position, m.pose.pose.orientation
        if first is None:
            first = (p.x, p.y)
        # The ground-truth message carries no twist; start = first 5 cm of travel.
        if t0 is None and math.hypot(p.x - first[0], p.y - first[1]) > 0.05:
            t0 = st
        if t0 is not None and st - last >= 2.0:
            last = st
            yaw = math.degrees(math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))
            px = 'n/a' if p1x is None else f'{p1x:5.2f}'
            print(f'  t={st - t0:5.1f}  robot ({p.x:5.2f},{p.y:5.2f}) yaw {yaw:6.1f}  person_1 x {px}')
    elif tp in ('/cmd_audio', '/blocked_person_event') and t0 is not None:
        m = deserialize_message(d, get_message(ty[tp]))
        what = m.data if tp == '/blocked_person_event' else 'BEEP'
        print(f'  t={st - t0:5.1f}  ** {what}')
