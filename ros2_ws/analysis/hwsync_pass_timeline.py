#!/usr/bin/env python3
"""hwsync_pass_timeline.py <bag_dir> - pass timeline of one head-on trial (8 Oct 2026).

Puts the critic / predictor log events (wall time) on the sim timeline via /clock
and prints, every 0.25 s around the closest approach: robot y, walker x, centre
distance, robot wz, and whether a person message / person cloud was received.
Used to separate the two hw-sync ports (side_ref_goal, noghost predictor)."""
import csv, os, re, sys
from bisect import bisect_left
from rclpy.serialization import deserialize_message
from rosgraph_msgs.msg import Clock
import rosbag2_py

bag = sys.argv[1].rstrip('/'); name = os.path.basename(bag)
ws = os.path.dirname(os.path.dirname(os.path.abspath(bag)))
r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=os.path.join(bag, 'data')), rosbag2_py.ConverterOptions('', ''))
clock, pers, cloud = [], [], []
while r.has_next():
    topic, data, t_ns = r.read_next()
    if topic == '/clock':
        c = deserialize_message(data, Clock).clock
        clock.append((t_ns * 1e-9, c.sec + c.nanosec * 1e-9))
    elif topic == '/predicted_person_positions':
        pers.append(t_ns * 1e-9)
    elif topic == '/predicted_person_cloud':
        cloud.append(t_ns * 1e-9)
clock.sort(); cw = [c[0] for c in clock]
def w2s(w):
    i = min(max(bisect_left(cw, w), 0), len(cw) - 1)
    return clock[i][1]
rows = [list(map(float, x.values())) for x in csv.DictReader(open(os.path.join(bag, 'trajectory.csv')))]
# columns: t,x_ref,y_ref,x_person,y_person,vx,wz,d_person_1
best = min(rows, key=lambda x: x[7])
ev = []
pat = re.compile(r'\[(\d+\.\d+)\] \[[a-z_]+\]: (.*)')
for f, keys in (('nav2.log', ('lane rule now follows', 'pass side fixed', 'switch', 'released', 'dropped lane')),
                ('perception.log', ('NOGHOST: dropped', 'Pruned stale'))):
    p = os.path.join(ws, 'logs', name, f)
    for line in open(p, errors='ignore'):
        m = pat.search(line)
        if m and any(k in m.group(2) for k in keys) and ('SocialCritic' in m.group(2) or 'NOGHOST' in m.group(2) or 'Pruned' in m.group(2)):
            ev.append((w2s(float(m.group(1))), m.group(2)))
ps = sorted(w2s(w) for w in pers); cs = sorted(w2s(w) for w in cloud)
def near(lst, t, h=0.125):
    i = bisect_left(lst, t - h); return i < len(lst) and lst[i] <= t + h
print(f'{name}: closest {best[7]:.3f} m at sim t {best[0]:.2f}')
for t, e in sorted(ev):
    print(f'  event {t - best[0]:+6.2f} s  {e}')
print('   dt     robot_y  walker_x  dist   wz    person_msg  cloud_msg')
t = best[0] - 4.0
while t <= best[0] + 1.5:
    x = min(rows, key=lambda q: abs(q[0] - t))
    print(f'  {t - best[0]:+5.2f}  {x[2]:+7.3f}  {x[3]:+7.2f}  {x[7]:5.2f}  {x[6]:+5.2f}   {"yes" if near(ps, t) else " - "}         {"yes" if near(cs, t) else " - "}')
    t += 0.25
