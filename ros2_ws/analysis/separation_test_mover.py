#!/usr/bin/env python3
"""
separation_test_mover.py

THESIS TEST HARNESS - exercises the RELEASE path of the sticky
confirmed-pair flag in group_formation_detector.py.

WHAT IT TESTS
-------------
Once a pair is VLM-confirmed, self.confirmed_pairs holds it on geometry
alone. The only thing that revokes it is _clear_close_since(), called
when the distance or speed check fails. If that path never fires, a
phantom conversation zone could persist in the costmap long after two
people have parted - so it needs to be seen working, not assumed.

WHY A WALK AND NOT A TELEPORT
-----------------------------
Two earlier attempts to test this were invalid, both for the same
reason: the pair's identity changed before the check could run.

  - Driving the robot away lost camera sight of one person, so the pair
    dropped out of active_ids and never reached the distance check at
    all. _clear_close_since() is only called from inside the pair loop.
  - Teleporting person_1 exceeded the association gates, killing the old
    tracks and allocating new stable ids (0,1 -> 5,7). The sticky entry
    is keyed frozenset({0,1}), so the new pair had no entry to clear.

The clear path can only be reached while BOTH members stay continuously
tracked under their ORIGINAL stable ids, and their separation grows past
CONV_MAX_DIST. That means walking, in steps small enough to stay inside
every association gate in the pipeline (leg_detector's 0.5m, the camera
jump filter's 0.8m, identity_fusion's 0.35m).

GEOMETRY
--------
person_2 is static at (3, +0.5). person_1 walks from (3, -0.5) to
(3, -2.0):

    separation 1.0m -> 2.5m, crossing CONV_MAX_DIST (1.8m) at y = -1.3

The endpoint is chosen so the crossing happens while person_1 is still
inside the camera's horizontal FOV. Walking further would eventually
push them out of frame, which would stop the pair being evaluated and
mask the very transition being tested - the same failure as attempt one.

USAGE
-----
The robot must be parked far enough back (~3.2m) to see BOTH people and
clear the TOP_CLIP_MARGIN_PX framing gate, or the pair will never
confirm in the first place and there will be nothing to release.

The script resets person_1 to the start position, then WAITS for you to
confirm the pair has been VLM-confirmed (watch for "VLM-confirmed - now
sticky" in the group detector log) before walking. Do not restart the
group detector after that point - confirmed_pairs is in-memory and a
restart wipes it.
"""

import math
import subprocess
import time

WORLD_NAME = "conversation_test"
MODEL_NAME = "person_1"

START_XY = (3.0, -0.5)
END_XY = (3.0, -2.0)

# Original SDF yaw: heading +Y (facing person_2) + pi/2 mesh offset.
# Held constant so person_1 backs away while still facing their partner -
# keeps the presented profile stable and avoids the bearing-dependent
# leg-cluster split confusing the lidar track mid-walk.
YAW = math.pi

STEP_DIST = 0.05    # m; well under every association gate in the chain
STEP_DELAY = 0.5    # s; ~0.1 m/s

Z = 0.0

CONV_MAX_DIST = 1.8
PERSON_2_Y = 0.5


def set_pose(x, y, yaw):
    qz, qw = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
    req = (
        f"name: '{MODEL_NAME}', position: {{x: {x}, y: {y}, z: {Z}}}, "
        f"orientation: {{x: 0, y: 0, z: {qz}, w: {qw}}}"
    )
    result = subprocess.run(
        ["gz", "service", "-s", f"/world/{WORLD_NAME}/set_pose",
         "--reqtype", "gz.msgs.Pose", "--reptype", "gz.msgs.Boolean",
         "--timeout", "3000", "--req", req],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5.0
    )
    if result.returncode != 0:
        print(f"  WARN set_pose failed: {result.stderr.strip()}")


def main():
    print("=" * 68)
    print("SEPARATION TEST - exercises the sticky-flag release path")
    print("=" * 68)

    print(f"\nResetting {MODEL_NAME} to {START_XY} ...")
    set_pose(START_XY[0], START_XY[1], YAW)

    print("\nNow waiting for the pair to be VLM-confirmed.")
    print("Watch the group detector log for:")
    print("    'VLM-confirmed - now sticky, held on geometry until it breaks'")
    print("\nIf it does not appear, the robot is probably too close")
    print("(framing gate) or cannot see both people. Reposition and retry.")
    print("\nDo NOT restart the group detector after confirmation -")
    print("confirmed_pairs is in-memory and a restart wipes it.")
    input("\nPress Enter once confirmed to start the walk... ")

    fy, ty = START_XY[1], END_XY[1]
    total = abs(ty - fy)
    n_steps = max(1, int(total / STEP_DIST))

    print(f"\nWalking y {fy:+.2f} -> {ty:+.2f}  ({total:.2f}m, {n_steps} steps)")
    print(f"Separation crosses CONV_MAX_DIST ({CONV_MAX_DIST}m) "
          f"at y = {PERSON_2_Y - CONV_MAX_DIST:+.2f}\n")

    crossed = False
    for i in range(1, n_steps + 1):
        y = fy + (ty - fy) * (i / n_steps)
        set_pose(START_XY[0], y, YAW)

        sep = abs(PERSON_2_Y - y)
        if not crossed and sep > CONV_MAX_DIST:
            crossed = True
            print(f"  step {i:2d}/{n_steps}: y={y:+.2f} sep={sep:.2f}m "
                  f"<-- CROSSED CONV_MAX_DIST, expect clear now")
        elif i % 5 == 0 or i == n_steps:
            print(f"  step {i:2d}/{n_steps}: y={y:+.2f} sep={sep:.2f}m")

        time.sleep(STEP_DELAY)

    print("\n" + "=" * 68)
    print("DONE. Check the group detector log for:")
    print("    'confirmation cleared - geometry broke'")
    print("and confirm /social_groups has stopped publishing conv_*.")
    print("=" * 68)


if __name__ == "__main__":
    main()
