#!/usr/bin/env python3
"""check_person_stop_zone.py <bag/data> - did the person cloud enter the PersonStop zone?

For every /predicted_person_cloud message: transform the points into the
robot frame with the Gazebo ground-truth pose and count those inside the
zone rectangle of social_nav2_headon_F_hwreq_block_stopbeep_sim.yaml.
Also lists /collision_monitor_state, /person_stop_event and /cmd_audio.
"""
import bisect
import math
import struct
import sys

from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message

X0, X1, HALF_W, MIN_POINTS = 0.19, 1.0, 0.30, 3

r = SequentialReader()
r.open(StorageOptions(uri=sys.argv[1], storage_id='mcap'), ConverterOptions('', ''))
types = {t.name: t.type for t in r.get_all_topics_and_types()}
want = ['/predicted_person_cloud', '/sim_ground_truth_pose', '/collision_monitor_state',
        '/person_stop_event', '/cmd_audio', '/clock']
gt, clouds, sim_t = [], [], 0.0
while r.has_next():
    topic, data, _ = r.read_next()
    if topic not in want:
        continue
    m = deserialize_message(data, get_message(types[topic]))
    if topic == '/clock':
        sim_t = m.clock.sec + m.clock.nanosec * 1e-9
    elif topic == '/sim_ground_truth_pose':
        p, q = m.pose.pose.position, m.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        gt.append((sim_t, p.x, p.y, yaw))
    elif topic == '/predicted_person_cloud':
        n = m.width * m.height
        pts = [struct.unpack_from('fff', m.data, i * m.point_step) for i in range(n)]
        clouds.append((sim_t, pts))
    elif topic == '/collision_monitor_state':
        print(f'  state      t={sim_t:8.2f}  action_type={m.action_type} polygon="{m.polygon_name}"')
    elif topic == '/person_stop_event':
        print(f'  event      t={sim_t:8.2f}  {m.data}')
    elif topic == '/cmd_audio':
        print(f'  beep       t={sim_t:8.2f}')

ts = [g[0] for g in gt]
inside_t, best = [], (0, None)
for t, pts in clouds:
    i = min(max(bisect.bisect_left(ts, t), 0), len(gt) - 1)
    _, rx, ry, yaw = gt[i]
    c, s = math.cos(yaw), math.sin(yaw)
    n_in = 0
    for x, y, _z in pts:
        bx = c * (x - rx) + s * (y - ry)
        by = -s * (x - rx) + c * (y - ry)
        if X0 <= bx <= X1 and abs(by) <= HALF_W:
            n_in += 1
    if n_in >= MIN_POINTS:
        inside_t.append(t)
    if n_in > best[0]:
        best = (n_in, t)
print(f'cloud msgs {len(clouds)}, non-empty {sum(1 for _, p in clouds if p)}')
print(f'max points inside the zone: {best[0]} at t={best[1]}')
if inside_t:
    print(f'zone occupied (>= {MIN_POINTS} pts) in {len(inside_t)} cloud msgs, '
          f't = {inside_t[0]:.2f} .. {inside_t[-1]:.2f}')
else:
    print('zone never occupied')
