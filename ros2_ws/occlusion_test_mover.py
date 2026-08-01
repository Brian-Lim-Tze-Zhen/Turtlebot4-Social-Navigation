#!/usr/bin/env python3
"""
occlusion_test_mover.py

THESIS TEST HARNESS - manufactures a camera-only occlusion event to
validate lidar-anchored re-ID.

Walks person_1 laterally out of the camera's horizontal FOV, holds it
there long enough for ByteTrack to drop the track_id (and for
human_kf_predictor's coast_timeout=1.5s to prune the KF), then walks it
back. Expected: camera assigns a NEW track_id on return, while the
lidar track stays continuous throughout.

-----------------------------------------------------------------------
YAW CONVENTION (fixed after an earlier bug in this script)
-----------------------------------------------------------------------
person_standing's mesh needs a +pi/2 offset, so:

    pose_yaw = heading + pi/2

(same convention documented in conversation_test.sdf and used by
move_person_gazebo.py / move_person_gazebo2.py). Yaw is recomputed
per-step from the direction of travel, so the person faces where it's
walking rather than moonwalking on the return leg.

An earlier version hardcoded pose_yaw = pi/2, rotating person_1 90deg
from its original orientation. That mattered far more than expected:
the simulated RPLidar raycasts the VISUAL MESH (standing.dae, a
two-legged figure), NOT the 0.25m collision cylinder in the SDF. At the
lidar's 0.193m scan height that's ankle level, so orientation decides
whether the legs occlude into ONE cluster or resolve into TWO:

  - side-on to the robot  -> legs aligned along the line of sight,
                             rear leg occluded -> ONE cluster
  - front/back-on         -> legs side by side across the line of
                             sight -> TWO clusters (~0.25m apart)

Confirmed empirically: the buggy pi/2 rotation produced two tracks
0.255m apart for person_1 while unrotated person_2 produced one.

Both legs of this walk (-Y out, +Y back) are side-on to a robot sitting
near the origin, so the one-cluster profile stays stable for the whole
run. That's deliberate: a profile that changes mid-walk causes 1<->2
cluster splits/merges, which spawn and orphan lidar tracks and corrupt
the very continuity measurement this test exists to make.

-----------------------------------------------------------------------
TIMING vs. REAL-TIME FACTOR
-----------------------------------------------------------------------
This script sleeps in WALL-CLOCK time, but scans arrive on SIM time.
Measured RTF on this machine oscillates ~0.08-1.0 (Gazebo GUI
rendering is the dominant cause - see notes/yolo-cpu-thread-contention.md).

At an instantaneous RTF of 0.08, one ~0.1s sim-time scan interval spans
~1.25s wall-clock. The step rate below is therefore chosen so that even
in that worst case the person moves well under leg_detector_node's
MAX_ASSOCIATION_DIST (0.5m) between consecutive scans:

    0.05m / 0.5s  = 0.1 m/s  ->  ~0.125m per scan at RTF 0.08

Do NOT "fix" association breakage by widening MAX_ASSOCIATION_DIST -
that gate is a real design parameter, and loosening it to make a test
pass would mask genuine tracking failures. Slow the mover instead, or
run Gazebo headless for a stable RTF near 1.0.
"""

import math
import subprocess
import time

WORLD_NAME = "conversation_test"
MODEL_NAME = "person_1"

START_XY = (3.0, -0.5)     # original position, in camera FOV
OCCLUDED_XY = (3.0, -4.0)  # bearing ~-53deg from robot: outside the
                            # OAK-D preview's horizontal FOV, but range
                            # ~5m keeps it well inside lidar coverage
                            # (and under leg_detector's 8m MAX_VALID_RANGE)

# Original SDF yaw for person_1 (<pose>3 -0.5 0 0 0 3.14159</pose>),
# i.e. heading=+pi/2 (facing person_2) + pi/2 mesh offset.
START_YAW = math.pi

STEP_DIST = 0.05    # m per step; << MAX_ASSOCIATION_DIST (0.5m)
STEP_DELAY = 0.5    # s between steps -> ~0.1 m/s (see TIMING note)
HOLD_SECONDS = 12.0 # a 5s hold already sufficed to force a new
                     # track_id in the first run; 12s adds margin for
                     # RTF sag, since ByteTrack/coast_timeout age out
                     # on sim time while this sleep is wall-clock

Z = 0.0


def set_pose(x, y, z, yaw):
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)

    req = (
        f"name: '{MODEL_NAME}', "
        f"position: {{x: {x}, y: {y}, z: {z}}}, "
        f"orientation: {{x: 0, y: 0, z: {qz}, w: {qw}}}"
    )

    cmd = [
        "gz", "service",
        "-s", f"/world/{WORLD_NAME}/set_pose",
        "--reqtype", "gz.msgs.Pose",
        "--reptype", "gz.msgs.Boolean",
        "--timeout", "3000",
        "--req", req,
    ]

    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5.0
    )
    if result.returncode != 0:
        print(f"  WARN set_pose failed: {result.stderr.strip()}")


def travel_yaw(dx, dy):
    """pose_yaw = heading + pi/2, normalized to [-pi, pi]."""
    heading = math.atan2(dy, dx)
    yaw = heading + math.pi / 2.0
    return math.atan2(math.sin(yaw), math.cos(yaw))


def walk(from_xy, to_xy, label):
    fx, fy = from_xy
    tx, ty = to_xy

    dx, dy = tx - fx, ty - fy
    total = math.hypot(dx, dy)
    n_steps = max(1, int(total / STEP_DIST))

    yaw = travel_yaw(dx, dy)

    print(f"\n[{label}] {from_xy} -> {to_xy}")
    print(f"  {total:.2f}m, {n_steps} steps, pose_yaw={yaw:.3f} rad "
          f"({math.degrees(yaw):.1f} deg)")

    for i in range(1, n_steps + 1):
        frac = i / n_steps
        x = fx + dx * frac
        y = fy + dy * frac
        set_pose(x, y, Z, yaw)
        if i % 10 == 0 or i == n_steps:
            print(f"  step {i}/{n_steps}: ({x:.2f}, {y:.2f})")
        time.sleep(STEP_DELAY)


def main():
    print("=" * 62)
    print("OCCLUSION TEST - camera-only occlusion, lidar stays continuous")
    print("=" * 62)

    print(f"\nResetting {MODEL_NAME} to {START_XY}, yaw={START_YAW:.3f}...")
    set_pose(START_XY[0], START_XY[1], Z, START_YAW)
    time.sleep(2.0)

    walk(START_XY, OCCLUDED_XY, "WALK OUT OF FOV")

    print(f"\n[HOLD] out of camera FOV for {HOLD_SECONDS}s (wall-clock)...")
    print("       -> expect camera track_id to be DROPPED")
    print("       -> expect lidar_id to stay CONTINUOUS")
    time.sleep(HOLD_SECONDS)

    walk(OCCLUDED_XY, START_XY, "WALK BACK INTO FOV")

    print("\nRestoring original facing (toward person_2)...")
    set_pose(START_XY[0], START_XY[1], Z, START_YAW)

    print("\n" + "=" * 62)
    print("DONE. Now compare:")
    print("  - leg_detector log: did one lidar_id persist throughout?")
    print("  - human_kf_predictor log: did a NEW track_id appear on return?")
    print("=" * 62)


if __name__ == "__main__":
    main()
