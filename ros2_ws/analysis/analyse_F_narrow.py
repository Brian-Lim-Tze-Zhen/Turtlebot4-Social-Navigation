#!/usr/bin/env python3
"""analyse_F_narrow.py - summarise one ablation-F narrow-corridor run.

Usage:
    python3 analyse_F_narrow.py <bag_dir>

Reads: /odom, /tf (map->odom), /social_groups, /social_zone_map, /clock,
       /cmd_vel, /plan  (anything missing is skipped and reported).
True person poses come from <bag_dir>/world_used.sdf (provenance copy);
falls back to (3.0, +/-0.75) with a warning.

Robot POSITION reference, in priority order:
  1. /sim_ground_truth_pose  (Gazebo world pose; world frame == map frame
     because the map is generated from the SDF)       -> exact
  2. /amcl_pose              (fallback for bags without ground truth)
/odom is used ONLY for velocities (vx, wz): its pose drifts badly in this
setup (measured 24 Sep: map->odom = (-0.52, -3.37, -146 deg) after
recoveries), so it must never be used for position.
If both 1 and 2 are present, AMCL error vs ground truth is reported.

Writes <bag_dir>/trajectory.csv.
"""
import bisect
import csv
import math
import os
import sys
import xml.etree.ElementTree as ET

import yaml
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageFilter, StorageOptions
from rosidl_runtime_py.utilities import get_message

# ---- parameters (keep in sync with the thesis setup) ----
SPAWN = (-1.0, 0.0, 0.0)        # x, y, yaw of the robot spawn (launch args)
GOAL = (6.0, 0.0)
GOAL_TOL = 0.30                 # m; "reached" radius for this analysis
ROBOT_R = 0.189                 # m; TB4 footprint radius (critic header)
PERSON_R = 0.25                 # m; SDF person collision radius
NARROW_THR = 0.39               # buffer < this = narrow (detector/zone/critic)
NARROW_SD = 0.60                # narrow_social_distance (F YAML)
SOCIAL_SD = 0.94                # social_distance (F YAML)
STALL_V = 0.02                  # m/s; |vx| below this counts as stopped
SPIN_W = 0.40                   # rad/s; |wz| above this with vx~0 = spinning
GAP_X = (2.0, 3.6)              # m (map x); the gap-passage window
MOVE_V = 0.05                   # m/s; first |vx| above this = run start


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def stamp_s(s):
    return s.sec + s.nanosec * 1e-9


def persons_from_world(bag):
    path = os.path.join(bag, 'world_used.sdf')
    out = []
    if os.path.exists(path):
        root = ET.parse(path).getroot()
        for m in root.iter('model'):
            if m.get('name', '').startswith('person'):
                pose = m.find('pose')
                if pose is not None:
                    v = [float(x) for x in pose.text.split()]
                    out.append((m.get('name'), v[0], v[1]))
    if not out:
        print('WARN: world_used.sdf missing/unparsed - using (3.0, +/-0.75)')
        out = [('person_1', 3.0, -0.75), ('person_2', 3.0, 0.75)]
    return sorted(out, key=lambda p: p[2])      # sorted by y


def read_bag(bag):
    meta = yaml.safe_load(open(os.path.join(bag, 'metadata.yaml')))
    sid = meta['rosbag2_bagfile_information']['storage_identifier']
    r = SequentialReader()
    r.open(StorageOptions(uri=bag, storage_id=sid), ConverterOptions('', ''))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    want = ['/odom', '/sim_ground_truth_pose', '/amcl_pose', '/social_groups',
            '/social_zone_map', '/clock', '/cmd_vel', '/plan']
    have = [t for t in want if t in types]
    missing = [t for t in want if t not in types]
    r.set_filter(StorageFilter(topics=have))
    mt = {t: get_message(types[t]) for t in have}

    d = {'odom': [], 'gt': [], 'amcl': [], 'groups': [], 'zone_last': None,
         'clock': [], 'cmd': [], 'plans': 0, 'plans_through_gap': 0,
         'plan_goals': [], 'missing': missing}
    while r.has_next():
        topic, raw, t_ns = r.read_next()
        msg = deserialize_message(raw, mt[topic])
        if topic == '/odom':
            p, tw = msg.pose.pose, msg.twist.twist
            d['odom'].append((stamp_s(msg.header.stamp), p.position.x, p.position.y,
                              yaw_of(p.orientation), tw.linear.x, tw.angular.z))
        elif topic == '/sim_ground_truth_pose':
            p = msg.pose.pose.position
            d['gt'].append((stamp_s(msg.header.stamp), p.x, p.y))
        elif topic == '/amcl_pose':
            p = msg.pose.pose.position
            d['amcl'].append((stamp_s(msg.header.stamp), p.x, p.y))
        elif topic == '/social_groups':
            f = msg.data.split(',')
            if len(f) >= 11:
                try:
                    mem = [tuple(float(v) for v in m.split(';')) for m in f[9].split('|')]
                    d['groups'].append((t_ns * 1e-9, float(f[10]), sorted(mem, key=lambda m: m[1])))
                except ValueError:
                    pass
        elif topic == '/social_zone_map':
            d['zone_last'] = msg
        elif topic == '/clock':
            d['clock'].append((t_ns * 1e-9, stamp_s(msg.clock)))
        elif topic == '/cmd_vel':
            tw = msg.twist if hasattr(msg, 'twist') else msg
            d['cmd'].append((t_ns * 1e-9, tw.linear.x, tw.angular.z))
        elif topic == '/plan':
            d['plans'] += 1
            if msg.poses:
                g = msg.poses[-1].pose.position
                goal = (round(g.x, 1), round(g.y, 1))
                if not d['plan_goals'] or d['plan_goals'][-1][1] != goal:
                    d['plan_goals'].append((stamp_s(msg.header.stamp), goal))
            if any(2.8 <= ps.pose.position.x <= 3.2 and abs(ps.pose.position.y) < 0.5
                   for ps in msg.poses):
                d['plans_through_gap'] += 1
    return d


def zone_value(grid, x, y):
    if grid is None:
        return None
    i = grid.info
    gx = int((x - i.origin.position.x) / i.resolution)
    gy = int((y - i.origin.position.y) / i.resolution)
    if 0 <= gx < i.width and 0 <= gy < i.height:
        return grid.data[gy * i.width + gx]
    return None


def analyse(d, persons):
    if d['gt']:
        ref, ref_name = d['gt'], '/sim_ground_truth_pose (exact)'
    elif d['amcl']:
        ref, ref_name = d['amcl'], '/amcl_pose (fallback, ~0.1 m error, sparse)'
    else:
        print('No /sim_ground_truth_pose or /amcl_pose - cannot analyse positions.')
        return []
    ref = sorted(ref)
    odom = sorted(d['odom'])
    ot = [o[0] for o in odom]

    def vel_at(t):
        if not odom:
            return 0.0, 0.0
        k = min(len(odom) - 1, max(0, bisect.bisect_right(ot, t) - 1))
        return odom[k][4], odom[k][5]

    rows = []
    for (t, x, y) in ref:
        vx, wz = vel_at(t)
        dists = [math.hypot(x - px, y - py) for (_, px, py) in persons]
        rows.append((t, x, y, None, None, vx, wz, dists))
    print(f'position reference: {ref_name}, {len(rows)} samples')

    # run window
    t0 = next((o[0] for o in odom if abs(o[4]) > MOVE_V), rows[0][0])
    t_goal = next((r[0] for r in rows
                   if math.hypot(r[1] - GOAL[0], r[2] - GOAL[1]) < GOAL_TOL), None)
    t_end = t_goal if t_goal is not None else rows[-1][0]
    win = [r for r in rows if t0 <= r[0] <= t_end]

    print('=' * 64)
    print('RUN')
    print(f'  goal reached (reference pose, tol {GOAL_TOL} m): '
          f'{"YES" if t_goal is not None else "NO"}')
    print(f'  time start->{"goal" if t_goal else "end"}: {t_end - t0:.1f} s (sim)')
    last = rows[-1]
    print(f'  final pose: ({last[1]:.2f}, {last[2]:.2f})')
    print(f'  max x reached: {max(r[1] for r in rows):.2f}  '
          f'(pair axis x = {sum(p[1] for p in persons) / len(persons):.2f})')

    # stall / spin (dense /odom velocities inside the run window)
    stall = spin = 0.0
    ow = [(o[0], None, None, None, None, o[4], o[5]) for o in odom if t0 <= o[0] <= t_end]
    for a, b in zip(ow, ow[1:]):
        dt = b[0] - a[0]
        if abs(a[5]) < STALL_V and abs(a[6]) < 0.1:
            stall += dt
        if abs(a[5]) < 0.05 and abs(a[6]) > SPIN_W:
            spin += dt
    print(f'  stopped time (|vx|<{STALL_V}, |wz|<0.1): {stall:.1f} s')
    print(f'  spin-in-place time (|wz|>{SPIN_W}): {spin:.1f} s')
    if ow:
        print(f'  min vx (reversing if <0): {min(r[5] for r in ow):.2f} m/s')
    if d['plan_goals']:
        print('  plan goals seen (sim t, goal): ' +
              ', '.join(f'{t:.0f}:{g}' for t, g in d['plan_goals']))

    # gap passage
    gap = [r for r in win if GAP_X[0] <= r[1] <= GAP_X[1]]
    if len(gap) > 1:
        print(f'  gap window x {GAP_X}: {gap[-1][0] - gap[0][0]:.1f} s, '
              f'vx mean {sum(r[5] for r in gap) / len(gap):.2f} / '
              f'min {min(r[5] for r in gap):.2f} m/s, '
              f'|y| max {max(abs(r[2]) for r in gap):.2f} m')
    else:
        print('  gap window: robot never entered it')

    print('=' * 64)
    print(f'CLEARANCE (reference pose, true person poses from SDF)')
    for i, (name, px, py) in enumerate(persons):
        best = min(rows, key=lambda r: r[7][i])
        dc = best[7][i]
        print(f'  {name} ({px:.2f},{py:.2f}): min centre {dc:.3f} m, '
              f'surface {dc - PERSON_R - ROBOT_R:+.3f} m at t={best[0]:.1f} '
              f'robot ({best[1]:.2f},{best[2]:.2f})')
    print(f'  reference: narrow_social_distance {NARROW_SD} m, '
          f'social_distance {SOCIAL_SD} m, contact at {PERSON_R + ROBOT_R:.3f} m')

    print('=' * 64)
    print('LOCALISATION (AMCL vs ground truth)')
    if d['gt'] and d['amcl']:
        gt = sorted(d['gt'])
        gtt = [g[0] for g in gt]
        errs = []
        for (t, x, y) in d['amcl']:
            k = min(len(gt) - 1, max(0, bisect.bisect_right(gtt, t) - 1))
            errs.append((math.hypot(x - gt[k][1], y - gt[k][2]),
                         x - gt[k][1], y - gt[k][2], t, gt[k][1]))
        e = [v[0] for v in errs]
        worst = max(errs)
        print(f'  AMCL error: mean {sum(e) / len(e):.3f} m, max {worst[0]:.3f} m '
              f'({len(e)} samples)')
        print(f'  worst sample: t={worst[3]:.1f}, robot true x={worst[4]:.2f}, '
              f'error along corridor (x) {worst[1]:+.3f} m, across (y) {worst[2]:+.3f} m')
        in_gap = [v for v in errs if GAP_X[0] <= v[4] <= GAP_X[1]]
        if in_gap:
            print(f'  max error while in gap window: {max(v[0] for v in in_gap):.3f} m, '
                  f'max |y| error there {max(abs(v[2]) for v in in_gap):.3f} m')
        if in_gap and max(v[0] for v in in_gap) > 0.3:
            print('  FLAG: AMCL error > 0.3 m INSIDE the gap window - passage may be affected')
    else:
        print('  needs both /sim_ground_truth_pose and /amcl_pose - skipped')
    print('=' * 64)
    print('GROUP DETECTION (/social_groups)')
    g = d['groups']
    if g:
        bufs = sorted(x[1] for x in g)
        n_nar = sum(1 for b in bufs if b < NARROW_THR)
        print(f'  msgs {len(g)}, narrow {n_nar}/{len(g)} '
              f'({100.0 * n_nar / len(g):.0f}%), buffer min/med/max '
              f'{bufs[0]:.3f}/{bufs[len(bufs) // 2]:.3f}/{bufs[-1]:.3f}')
        for k in range(2):
            mx = sum(x[2][k][0] for x in g) / len(g)
            my = sum(x[2][k][1] for x in g) / len(g)
            name, px, py = persons[k]
            print(f'  detected member (lower-y #{k}) mean ({mx:.3f},{my:.3f}) vs '
                  f'{name} ({px:.2f},{py:.2f}): error {math.hypot(mx - px, my - py):.3f} m')
        print(f'  first group msg at bag t={g[0][0]:.1f}, last at {g[-1][0]:.1f}')
    else:
        print('  no /social_groups messages')

    zl = d['zone_last']
    if zl is not None:
        mx = sum(p[1] for p in persons) / len(persons)
        my = sum(p[2] for p in persons) / len(persons)
        print(f'  last zone mask at true midpoint ({mx:.2f},{my:.2f}): '
              f'{zone_value(zl, mx, my)}  (narrow expects 35, wide 90)')

    print('=' * 64)
    print('PLANNER / SIM')
    print(f'  /plan msgs {d["plans"]}, through the gap {d["plans_through_gap"]}')
    c = d['clock']
    if len(c) > 1 and c[-1][0] > c[0][0]:
        print(f'  RTF (sim/wall over bag): '
              f'{(c[-1][1] - c[0][1]) / (c[-1][0] - c[0][0]):.2f}')
    if d['missing']:
        print(f'  topics not in bag: {", ".join(d["missing"])}')
    print('=' * 64)
    return rows


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    bag = sys.argv[1].rstrip('/')
    print(f'bag: {bag}')
    persons = persons_from_world(bag)
    d = read_bag(bag)
    rows = analyse(d, persons)
    if rows:
        out = os.path.join(bag, 'trajectory.csv')
        with open(out, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['t', 'x_ref', 'y_ref', 'unused1', 'unused2', 'vx', 'wz']
                       + [f'd_{p[0]}' for p in persons])
            for r in rows:
                w.writerow([f'{r[0]:.3f}', f'{r[1]:.3f}', f'{r[2]:.3f}',
                            '' if r[3] is None else f'{r[3]:.3f}',
                            '' if r[4] is None else f'{r[4]:.3f}',
                            f'{r[5]:.3f}', f'{r[6]:.3f}'] + [f'{x:.3f}' for x in r[7]])
        print(f'wrote {out}')


if __name__ == '__main__':
    main()
