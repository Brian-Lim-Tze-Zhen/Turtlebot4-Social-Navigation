#!/usr/bin/env python3
"""queue_run_summary.py <bag> [goal_x goal_y] - one queue trial (world queue_test).

Robot path against Gazebo ground truth, distance to each queue member (true
poses from <bag>/world_used.sdf), where the robot crossed the queue's line,
what /social_groups reported, and the zone mask along the line.
"""
import collections
import math
import os
import sys
import xml.etree.ElementTree as ET

from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message

ROBOT_R, PERSON_R, GOAL_TOL = 0.189, 0.25, 0.30
bag = sys.argv[1].rstrip('/')
goal = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) >= 4 else (6.0, 0.0)

people = []
for m in ET.parse(os.path.join(bag, 'world_used.sdf')).getroot().iter('model'):
    if m.get('name', '').startswith('person'):
        v = [float(x) for x in m.find('pose').text.split()]
        people.append((m.get('name'), v[0], v[1]))
people.sort(key=lambda p: -p[2])

r = SequentialReader()
r.open(StorageOptions(uri=bag, storage_id='mcap'), ConverterOptions('', ''))
ty = {t.name: t.type for t in r.get_all_topics_and_types()}
st, path, groups, zone = 0.0, [], collections.Counter(), None
t_first_group = {}
while r.has_next():
    tp, d, _ = r.read_next()
    if tp == '/clock':
        m = deserialize_message(d, get_message(ty[tp]))
        st = m.clock.sec + m.clock.nanosec * 1e-9
    elif tp == '/sim_ground_truth_pose':
        m = deserialize_message(d, get_message(ty[tp]))
        path.append((st, m.pose.pose.position.x, m.pose.pose.position.y))
    elif tp == '/social_groups':
        f = deserialize_message(d, get_message(ty[tp])).data.split(',')
        key = (f[1], len(f[9].split('|')))
        groups[key] += 1
        t_first_group.setdefault(key, st)
    elif tp == '/social_zone_map':
        m = deserialize_message(d, get_message(ty[tp]))
        if zone is None or sum(1 for v in m.data if v > 0) >= zone[1]:
            zone = (m, sum(1 for v in m.data if v > 0))

x0, y0 = path[0][1], path[0][2]
t0 = next((t for t, x, y in path if math.hypot(x - x0, y - y0) > 0.05), path[0][0])
tg = next((t for t, x, y in path if math.hypot(x - goal[0], y - goal[1]) < GOAL_TOL), None)
run = [p for p in path if p[0] >= t0 and (tg is None or p[0] <= tg)]
still = sum(b[0] - a[0] for a, b in zip(run, run[1:])
            if math.hypot(b[1] - a[1], b[2] - a[2]) / max(b[0] - a[0], 1e-6) < 0.02)
print(f'bag: {bag}   goal ({goal[0]}, {goal[1]})')
print(f'  goal reached: {"yes" if tg else "NO"}   time {((tg or run[-1][0]) - t0):.1f} s   '
      f'standing still {still:.1f} s   max |y| {max(abs(p[2]) for p in run):.2f} m')
qx = sum(p[1] for p in people) / len(people)
cross = next(((a, b) for a, b in zip(run, run[1:]) if (a[1] - qx) * (b[1] - qx) <= 0), None)
if cross:
    yc = cross[0][2]
    ys = [p[2] for p in people]
    where = ('beyond the head (+y end)' if yc > max(ys) else
             'beyond the tail (-y end)' if yc < min(ys) else 'THROUGH THE QUEUE')
    print(f'  crossed the queue line (x = {qx:.2f}) at y = {yc:+.2f}: {where}')
for name, px, py in people:
    dmin = min(math.hypot(x - px, y - py) for _, x, y in run)
    print(f'  {name} ({px:.2f},{py:+.2f}): min centre {dmin:.3f} m, surface {dmin - ROBOT_R - PERSON_R:+.3f} m')
print('  /social_groups (type, members): messages, first seen after start')
for k, v in groups.most_common():
    print(f'    {k[0]:13s} {k[1]} members: {v:4d}   t = {t_first_group[k] - t0:+.1f} s')
if zone:
    m = zone[0]
    i = m.info

    def at(x, y):
        return m.data[int((y - i.origin.position.y) / i.resolution) * i.width
                      + int((x - i.origin.position.x) / i.resolution)]
    ys = [max(p[2] for p in people) + 1.2 - 0.3 * k for k in range(18)]
    print('  zone mask along the queue line (largest mask), y: cost')
    print('   ', '  '.join(f'{y:+.1f}:{at(qx, y)}' for y in ys))
