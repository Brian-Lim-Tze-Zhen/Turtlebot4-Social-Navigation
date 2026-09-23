#!/usr/bin/env python3
"""
analyze_bag_tracks.py

Reads /lidar_person_clusters from a recorded bag and reports:
- When each track ID first appeared (confirmed)
- When each track ID disappeared (dropped)
- How long each track lived
- Whether it was dropped while still actively moving (suspicious = bug candidate)

Usage:
  python3 analyze_bag_tracks.py /tmp/test_bag
"""

import sys
import os
import sqlite3
import json
import struct

BAG_DIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/test_bag"

# Find the .db3 file
db3 = next((os.path.join(BAG_DIR, f) for f in os.listdir(BAG_DIR) if f.endswith(".db3")), None)
if not db3:
    print("No .db3 file found in", BAG_DIR)
    sys.exit(1)

con = sqlite3.connect(db3)

# Get topic id for /lidar_person_clusters
row = con.execute(
    "SELECT id FROM topics WHERE name='/lidar_person_clusters'"
).fetchone()
if not row:
    print("Topic /lidar_person_clusters not found in bag")
    print("Available topics:")
    for r in con.execute("SELECT name FROM topics"):
        print(" ", r[0])
    sys.exit(1)

topic_id = row[0]

# Fetch all messages (timestamp + serialized data)
rows = con.execute(
    "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp",
    (topic_id,)
).fetchall()

print(f"Found {len(rows)} messages on /lidar_person_clusters")
print()

# Parse std_msgs/String — CDR encoding: 4-byte header + 4-byte length + string
def parse_string_msg(data):
    try:
        # CDR: 4 bytes header, 4 bytes string length, then string
        str_len = struct.unpack_from('<I', data, 4)[0]
        return data[8:8+str_len-1].decode('utf-8')  # -1 to strip null terminator
    except Exception:
        return None

# Track state per id
track_first_seen = {}   # id -> timestamp (ns)
track_last_seen  = {}   # id -> timestamp (ns)
track_positions  = {}   # id -> list of (x, y)

prev_ids = set()

for ts, data in rows:
    text = parse_string_msg(data)
    if text is None:
        continue

    # Format: "id,x,y,vx,vy[;id,x,y,vx,vy...]" or similar
    # Check actual format
    current_ids = set()
    for entry in text.strip().split(';'):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(',')
        if len(parts) < 3:
            continue
        try:
            tid = int(parts[0])
            x   = float(parts[1])
            y   = float(parts[2])
        except ValueError:
            continue
        current_ids.add(tid)
        if tid not in track_first_seen:
            track_first_seen[tid] = ts
        track_last_seen[tid] = ts
        track_positions.setdefault(tid, []).append((x, y))

    # Detect disappearances
    disappeared = prev_ids - current_ids
    for tid in disappeared:
        if tid in track_last_seen:
            lifetime_s = (track_last_seen[tid] - track_first_seen[tid]) / 1e9
            positions  = track_positions.get(tid, [])
            # Compute total travel
            travel = 0.0
            for i in range(1, len(positions)):
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                travel += (dx**2 + dy**2)**0.5
            last_x, last_y = positions[-1] if positions else (0, 0)
            ts_s = ts / 1e9
            print(f"[DROP] id:{tid:2d}  lifetime={lifetime_s:.1f}s  "
                  f"travel={travel:.2f}m  "
                  f"last_pos=({last_x:.2f},{last_y:.2f})  "
                  f"at t={ts_s:.1f}s")

    prev_ids = current_ids

# Final summary
print()
print("=" * 60)
print("TRACK SUMMARY")
print("=" * 60)
for tid in sorted(track_first_seen):
    t0 = track_first_seen[tid]
    t1 = track_last_seen[tid]
    lifetime = (t1 - t0) / 1e9
    positions = track_positions.get(tid, [])
    travel = 0.0
    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i-1][0]
        dy = positions[i][1] - positions[i-1][1]
        travel += (dx**2 + dy**2)**0.5
    print(f"  id:{tid:2d}  lifetime={lifetime:.1f}s  travel={travel:.2f}m  "
          f"observations={len(positions)}")

print()
# Print first message raw to understand format
if rows:
    sample = parse_string_msg(rows[0][1])
    print(f"Sample message format: '{sample[:200] if sample else 'parse failed'}'")
