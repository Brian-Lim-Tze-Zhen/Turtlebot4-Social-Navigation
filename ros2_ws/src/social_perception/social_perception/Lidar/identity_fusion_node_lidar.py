#!/usr/bin/env python3
"""
identity_fusion_node.py

THESIS ADDITION - camera/lidar identity fusion (re-ID across occlusion).

-----------------------------------------------------------------------
THE PROBLEM
-----------------------------------------------------------------------
Every downstream layer keys its state by ByteTrack's track_id. A ~2s+
camera occlusion makes ByteTrack assign a NEW track_id to the same
physical person, which resets:

  - human_kf_predictor.py : new HumanTrackKF, update_count=0,
                            vx_filt=None, covariance reset, velocity
                            suppressed for 5 updates
  - group_formation_detector.py : close_since dict emptied, so a
                            conversation pair established for 30s must
                            re-accumulate CONV_MIN_DURATION from zero
  - social_group_cloud_node.py (planned) : group_id string changes,
                            costmap zone ages out and rebuilds -> flicker

-----------------------------------------------------------------------
THE FIX
-----------------------------------------------------------------------
The 2D lidar does not share the camera's occlusion geometry. Measured
this session: through a full walk-out / 12s hold / walk-back cycle, the
camera assigned a new track_id on return while ONE lidar track
(leg_detector_node id:2, 659 msgs, y spanning -3.95 to -0.42) stayed
continuous and was never pruned.

So: bind camera track_ids to lidar tracks while both are visible. When
a NEW camera track_id appears near a lidar track that still carries a
binding from a previously-lost camera track, adopt that old identity
instead of allocating a fresh one.

This node is a TRANSPARENT SHIM. It republishes /person_positions_map
verbatim except field [0] (track_id), which is rewritten to a stable
identity. Point human_kf_predictor's existing `input_topic` parameter
at the output topic and nothing else in the pipeline needs to change -
the KF, the close_since timers, and any future group_id strings simply
never observe the switch.

-----------------------------------------------------------------------
WHY SPURIOUS LIDAR TRACKS ARE HARMLESS HERE (design note)
-----------------------------------------------------------------------
leg_detector_node produces spurious tracks from wall fragments: a
person standing in front of a wall splits its continuous return into
short segments, and at ~4.9m range a real leg yields only 2-3 points,
making legs and fragments genuinely indistinguishable by width (measured:
fragment 0.061m sits BETWEEN person clusters at 0.048m and 0.100m) and
by shape (a 2-point cluster is exactly collinear by definition, so
curvature is undefined - no shape test can exist).

Rather than trying to purify the detector, this node makes purity
unnecessary: a lidar track can only ever be adopted as an anchor if it
was CAMERA-CONFIRMED first, i.e. it appeared in stable_of_lidar because
a real camera detection was matched to it. A wall fragment is never
camera-confirmed, so it can never donate an identity no matter how long
it persists. The detector does not need to be clean - only continuous,
which was verified independently.

-----------------------------------------------------------------------
INPUT / OUTPUT
-----------------------------------------------------------------------
Subscribes:
  /person_positions_map    (String) from yolo_leg_detector_lidar.py
      track_id,conf,x1,y1,x2,y2,leg_kpts
      (pixel-space only - the camera node no longer computes depth or
      a map-frame position; LIDAR is the sole range/position source)
  /lidar_person_clusters   (String) from lidar_person_detector.py
      lidar_id,conf,map_x,map_y   (conf is a placeholder, always 1.00)

Publishes:
  /person_positions_fused  (String) -
      stable_id,conf,x,y,depth,u,v,x1,y1,x2,y2,source
      Field [0] is the stable id (not camera's raw track_id). Fields
      [2],[3] (x,y) are always the LIDAR track's position - the camera
      node has no position of its own to fall back on anymore. depth/u/v
      (fields [4]-[6]) are always "0.0","0","0" now (no longer computed);
      kept only so field count/indices stay stable for any downstream
      parser. x1,y1,x2,y2 (fields [7]-[10]) are the camera's pixel bbox
      when a camera detection drove this message.
      Field [11], "source", is either "camera_confirmed" (this cycle's
      camera detection matched a LIDAR track by bearing) or "lidar_only"
      (person is camera-occluded; position is LIDAR alone, coasting on
      a previously camera-confirmed binding).
      A camera detection with NO lidar match this cycle publishes
      nothing - there is no camera-only position to report anymore, so
      unlike the old depth-based version, an unmatched camera track is
      silent rather than emitting a stale/wrong position.
      human_kf_predictor.py should apply higher measurement/process
      noise on "lidar_only" updates.

      lidar_only messages are emitted from a timer, not from
      camera_callback - there is no camera detection to piggyback on
      while occluded, so this is the only path that keeps a person's
      track alive (and SocialCritic fed) through the occlusion window.

-----------------------------------------------------------------------
CALIBRATION STATUS - READ BEFORE TRUSTING THE ANGLE GATE
-----------------------------------------------------------------------
Camera/lidar matching is now bearing-based, not position-based: the
camera contributes a pixel bbox center converted to a bearing relative
to the robot's heading (assumes the OAK-D's optical axis is aligned
with base_link's forward axis - true for TurtleBot4's fixed mount), and
each lidar track's map-frame (x,y) is converted to its own bearing from
the robot via TF (map -> base_link). ANGLE_GATE_DEG bounds how far
apart those two bearings may be and still count as the same person.

Like the old distance gate, this is an initial estimate, not a measured
one. This node logs every camera-lidar pairing bearing it evaluates
(look for "pair-bearing" lines, in degrees) precisely so the gate can be
set from the observed distribution after one run instead of guessed.
Do that before relying on any result from this node. Two people
standing at similar bearings but different ranges will alias under
bearing-only matching in a way the old distance gate would not have -
worth watching for in multi-person scenes.

-----------------------------------------------------------------------
THESIS FIX (position-based re-ID fallback) - read before touching the
re-ID logic in camera_callback() or prune()
-----------------------------------------------------------------------
The id-based re-ID step (adopt stable_of_lidar[lid] when a new camera
track lands on a lidar id that already carries a binding) only works if
THIS CYCLE's lidar id is already a stable_of_lidar key. That assumption
breaks when lidar_person_detector's own clustering churns - confirmed
repeatedly this session that a person's leg cluster can split/merge
differently between scans, especially near furniture/walls, producing a
BRAND NEW raw lidar id for the same physical person. When that happens
at the same time as a camera track_id change (the common case: camera
loses someone right as they walk near an obstacle), NEITHER id matches
anything already known, and the id-based step silently allocates a new
stable identity - even though stable_of_lidar still directly relates
that old identity to a nearby physical location.

Fix: when a lidar->stable binding is dropped in prune() (from either
LIDAR_BINDING_TIMEOUT or the lidar track vanishing), remember its last
known (x, y) as an "orphan" for ORPHAN_REATTACH_WINDOW seconds. A new,
otherwise-unmatched camera+lidar detection landing within
ORPHAN_REATTACH_RADIUS of an orphan's last position reclaims that
identity instead of allocating a new one. This is a fallback checked
only when the id-based step above found nothing - it never overrides a
successful id-based match, and it still respects stable_id_in_use() so
it can't merge two people who are both genuinely still being tracked.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener


# =======================================================================
# Tunables
# =======================================================================

# THESIS CHANGE (bearing-based matching): the camera node no longer
# computes depth or a map-frame position (see yolo_leg_detector_lidar.py -
# depth/TF removed, LIDAR is now the sole range source). camera_callback
# therefore has nothing but a pixel bbox to offer, so matching switched
# from "camera (x,y) vs lidar (x,y) within CAMERA_LIDAR_GATE metres" to
# "camera bbox-center bearing vs lidar track bearing (both relative to
# the robot, via TF) within ANGLE_GATE degrees".
#
# Max angular separation to consider a camera detection and a lidar
# track the same person. PROVISIONAL like the old distance gate was -
# log lines below ("pair-bearing") let this be tuned from the observed
# distribution instead of guessed.
# THESIS FIX (sticky binding to furniture). 20 deg allowed ~0.7 m of
# lateral slack at 2 m, so a static cluster and the real person could both
# sit inside the gate; sticky_lidar then held the first pick indefinitely.
# Measured on headon_ray3: at 20 deg the identity sat on a table at
# (0.73..0.81, -0.33..-0.29) - 0.10 m of movement in 1.7 s - then
# fragmented into a second id. Swept on the same bag:
#   20 deg: 2 tracks, spread 0.10 m and 0.43 m, 29 camera_confirmed
#   12 deg: 1 track,  spread 2.61 m,            20 camera_confirmed
#    8 deg: 1 track,  spread 2.61 m,            16 camera_confirmed
# 12 keeps the continuous track with the most confirmations. NOT yet
# verified against tape-measure ground truth, and swept on ONE 7.5 s
# encounter - re-run the sweep on a second bag before treating as final.
ANGLE_GATE_DEG = 12.0

# THESIS FIX (out-of-FOV bindings). ANGLE_GATE_DEG bounds how far a lidar
# candidate may sit FROM THE CAMERA DETECTION, but nothing bounded where
# it sits in ABSOLUTE bearing. Calibrated intrinsics (fx 287.31, 11 Sep)
# give a true HFOV of 47.0 deg, i.e. +/-23.5 deg (gate = 27 deg incl.
# margin); a detection near the frame edge could
# therefore bind to a cluster the camera physically cannot see.
# On walk_full, stable id 0 sat at -42.7 deg with 5 camera_confirmed
# messages - 10.9 deg outside the FOV - and never moved, so neither
# MAX_COAST_SPEED nor the per-step coast check could catch it: a
# stationary wrong answer passes every continuity test.
# Applied to both match paths and to the coast, so no path can accept or
# hold a candidate outside the camera's view. Exposed as the ROS
# parameter fov_gate_deg for offline tuning; <= 0 disables the gate.
FOV_GATE_DEG = 27.0

# Calibrated OAK-D preview intrinsics / mounting (250x250 crop)
CAMERA_FX = 287.31
CAMERA_CX = 124.95
CAMERA_YAW_OFFSET_DEG = 2.25   # scan bearing test 11 Sep: mean residual +1.5 deg
CAMERA_X_OFFSET = 0.165        # m, forward of base_link

# Preview stream width in pixels (matches oakd_pro.yaml i_preview_size).
CAMERA_PREVIEW_WIDTH = 250

# Frames for the TF lookup used to convert lidar map-frame (x,y) into a
# bearing relative to the robot's current heading.
MAP_FRAME = "map"
BASE_FRAME = "base_link"

# How long a camera track_id -> stable_id mapping is retained after the
# camera stops reporting it. Only needs to outlive brief detection
# dropouts; the lidar binding is what survives real occlusions.
CAMERA_TRACK_TIMEOUT = 2.0    # s

# A stable id is considered "in use" if some camera track reported it
# this recently. Prevents two SIMULTANEOUSLY visible camera tracks from
# collapsing onto one stable id (which would merge two people).
CAMERA_ACTIVE_WINDOW = 1.0    # s

# How long a lidar->stable binding survives without camera confirmation.
# This is the real occlusion budget: it must exceed the longest camera
# gap to be bridged.
#
# MEASURED, not guessed: an earlier value of 30s was set by (wrongly)
# equating the blind window with the mover's 12s hold phase. The actual
# camera-blind window is far longer - the person exits the FOV partway
# through the 35s walk-out and only re-enters late in the 35s walk-back,
# giving ~60s. The binding expired 0.85s before the returning camera
# track appeared, so the re-ID had nothing left to adopt and allocated a
# fresh identity instead. Confirmed the lidar anchor itself was never
# pruned during that window.
#
# Note this timeout is NOT the safety property. The real guard against
# adopting a stale identity is the continuity check in prune(): a
# binding is dropped the moment its lidar track disappears, which is
# exactly what happens if the person genuinely leaves lidar range. That
# is the "long absence" case the design scope explicitly excludes. So
# this can be generous - it only bounds how long a CONTINUOUSLY TRACKED
# but camera-invisible person keeps their identity.
LIDAR_BINDING_TIMEOUT = 300.0  # s

# Drop a cached lidar track this long after its last message.
LIDAR_TRACK_TIMEOUT = 2.0     # s

# THESIS FIX (stale position published as camera_confirmed).
# lidar_tracks[lid] is only refreshed when the detector republishes that
# id. Between republishes the cached entry persists for up to
# LIDAR_TRACK_TIMEOUT (2.0 s), and every camera detection in that window
# re-publishes the SAME frozen coordinates tagged camera_confirmed - so
# the message looks fully corroborated while the position is seconds old.
# Measured on trial_horizon200: lidar 95 last appeared at t24.09 at
# (-1.15, 0.28); the fused output kept publishing that exact point until
# t26.9 - 2.8 s - while the bbox moved steadily across the frame, i.e.
# the camera was tracking the person the whole time. At walking pace
# that is ~1 m of position error, which is most of the available
# clearance in a corridor, so MPPI was planning around where the person
# HAD been. camera_confirmed does not imply the position is current.
# Publishing nothing is better than a confidently wrong position.
LIDAR_POSITION_MAX_AGE = 0.5   # s

# How often to emit "lidar_only" coasting updates for stable ids whose
# camera track has gone quiet but whose lidar binding is still alive.
# Independent of scan rate - this is a publish cadence, not a sensor rate.
LIDAR_ONLY_PUBLISH_PERIOD = 0.1   # s

# THESIS FIX (unvalidated coasting). Max speed a coasting position is
# allowed to imply, relative to the last camera-confirmed position for
# that identity. Set from the closing speed this pipeline was tuned for
# (1.46 m/s combined) rounded up - a coast position further away than a
# person could have walked in the elapsed time is not that person.
#
# MEASURED PROBLEM: on a 52 s hardware run the fused track spent 18% of
# the time frozen at exactly (-3.99, 0.32), in stretches up to 3.8 s,
# republished at 10 Hz. Downstream that is a stationary person: the KF
# integrates zero velocity, prediction equals current position, and the
# obstacle cloud stops following the real person entirely. Same failure
# in bag headon_t01, where the track froze at (-1.59, 2.02) for the
# rest of the recording.
#
# Raising this weakens the check; lowering it starts rejecting real
# fast walkers. It is a plausibility bound, not a tracking parameter.
MAX_COAST_SPEED = 1.5   # m/s

# THESIS FIX (phantom identities). How long an identity may keep
# coasting on lidar alone with no fresh camera confirmation before its
# binding is dropped entirely.
#
# CAMERA_ONLY_GRACE_PERIOD above catches an identity that NEVER got
# lidar corroboration. This is the mirror case, which nothing caught: an
# identity that got corroborated once, then stopped being camera-
# confirmed but kept a live lidar track, so LIDAR_BINDING_TIMEOUT (300 s)
# let it coast indefinitely. MAX_COAST_SPEED does not catch it either -
# a static object barely drifts, so every coast position looks
# "plausible".
#
# MEASURED: over a 760 s run the camera-confirmation rate cleanly
# separated real people from static false positives -
#   real:  id5 81%, id7 79%, id11 70%, id12 100%  (0.4-0.7 m/s)
#   false: id0  6%, id1  2%, id8  3%, id9  13%    (0.1-0.3 m/s)
# id9 held a person identity for 86 s while moving 1.48 m total.
#
# COUPLING - read before changing: this trades against occlusion
# tolerance, which is the whole reason lidar-only coasting exists. Too
# short and a genuinely occluded person is dropped mid-encounter; too
# long and phantoms persist. 5 s is roughly 2x CAMERA_ONLY_GRACE_PERIOD
# and far longer than any camera gap seen in the runs above, while being
# ~17x shorter than the 86 s phantom. A dropped identity can still be
# recovered by find_orphan_to_reattach(); a phantom cannot be undone.
CAMERA_CORROBORATION_TIMEOUT = 5.0   # s

# THESIS FIX (moving-track exemption, real robot 9 Sep). With the robot
# DRIVING, a person passing beside it leaves the camera FOV for most of
# the encounter while the lidar track stays alive. Measured: fused went
# silent for 19.8 s with lidar clusters continuous the whole time - the
# log was 100+ lines of 'coast rejected ... outside camera FOV', then
# 'no camera corroboration for 5.6s' dropped the binding. Both gates
# exist to kill STATIC phantoms (walls, door frames). A track that has
# displaced >= MOVING_MIN_DISP_M over the last MOVING_WINDOW_S is not a
# wall, so for such tracks the FOV gate is skipped and the corroboration
# cutoff is extended. Static tracks keep every existing guard unchanged.
MOVING_WINDOW_S = 1.0                  # s; displacement measured over at least this long
MOVING_MIN_DISP_M = 0.5                # m; >= this over the window = moving (cluster jitter hops are transient, ~0.7 m max single-scan)
MOVING_GRACE_S = 3.0                   # s; track counts as moving for this long after last qualifying displacement
MOVING_CORROBORATION_TIMEOUT = 15.0    # s; replaces CAMERA_CORROBORATION_TIMEOUT for moving tracks

# THESIS FIX (static cluster wins the bearing argmin). Bearing alone
# cannot separate a person from a wall feature behind them: both sit
# within a few degrees, so nearest_lidar()'s argmin picks whichever is
# marginally closer in angle and sticky_lidar() then locks it in for the
# whole encounter.
#
# MEASURED on bag headon2, over the 76 camera detections that had at
# least one candidate inside the 20 deg gate:
#   current (bearing only) : picks a moving track 34%, a static one 66%
#   with a 2:1 range gate  : picks a moving track 89%, a static one  8%
# Ratios of 1.5 reject everything 46% of the time; 3.0 lets the statics
# back in (39%). 2.0 is a genuine optimum, not a slope.
#
# The estimate comes from bbox height via the pinhole relation
#   range ~= PERSON_HEIGHT_M * fy / bbox_height_px
# It is deliberately used as a RATIO gate, not an absolute match: the
# bbox is routinely clipped at the top of the 250 px frame (y1=0), so
# the height underestimates the person and the range comes out short by
# up to ~1.5 m. A ratio test tolerates that bias while still rejecting a
# 7.4 m wall when the person is at 2.4 m.
#
# Set to 0 to disable the gate entirely and fall back to bearing-only.
PERSON_HEIGHT_M = 1.75
# CALIBRATED 11 Sep 2026: fx=fy=287.31, cx=124.95 on the 250x250 preview
# -> HFOV 47.0 deg. Supersedes camera_info (fx=201.3, HFOV 63.7 deg).
# Validated against raw /scan leg bearings, 1.4-3.2 m, +/-14 deg:
# residual -0.1 +/- 0.9 deg (max 1.5 deg) with yaw offset 2.25 deg.
CAMERA_FY = 287.31
# THESIS ADDITION: a bbox whose top edge is within this many pixels
# of y=0 is treated as clipped, so its height - and any range derived
# from it - is unusable. Set <0 to disable the check.
# THESIS ADDITION: use leg-keypoint bearing when keypoints are
# present, falling back to the bbox centre otherwise. See
# camera_callback for the rationale.
USE_KEYPOINT_BEARING = True

BBOX_CLIP_TOP_PX = 2.0

RANGE_RATIO_GATE = 2.0

# THESIS FIX (identity teleporting on the camera-confirmed path). The
# camera path re-runs the match every cycle and publishes whatever lidar
# track wins, with no check that the result is continuous with where the
# SAME identity was a moment ago. MAX_COAST_SPEED guards only
# publish_lidar_only(); nothing guarded this path at all.
#
# MEASURED on the headon3 run: 37 of 283 transitions within a stable id
# (13%) implied speeds above 1.5 m/s, and the worst were
# camera_confirmed -> camera_confirmed, not coasting:
#   16.3 m/s (1.29 m in 0.08 s), 15.4, 14.2, 9.6 ...
# Downstream that is one person rendered as an obstacle blob jumping
# metres between frames, and a KF fed impossible velocities.
#
# Why it happens: with a bearing-only camera, two clusters at similar
# bearings are near-indistinguishable, so the argmin flips between them
# frame to frame. A head-on approach is the worst case - the person
# holds an almost constant bearing while walking straight at the robot,
# so bearing carries least information exactly when it is needed most.
#
# Position continuity is the constraint that does not depend on bearing:
# a person cannot cross 1.3 m in 80 ms whatever the angles say. Same
# value as MAX_COAST_SPEED - both encode "a person cannot move faster
# than this", and the measured jumps (5-16 m/s) sit far clear of real
# walking speed (<1 m/s), so the threshold is not delicate.
MAX_IDENTITY_SPEED = 1.5   # m/s
IDENTITY_JITTER_M = 0.7   # m; fixed centroid-hop allowance, see camera_callback

# THESIS FIX (gate-then-argmin discards range information). The range
# ratio was a FILTER: everything inside 2:1 was treated as equally
# valid, and bearing alone then decided. So a cluster at 1.95x the
# estimated range could beat one at 1.36x purely on a 3.5 deg better
# bearing - which is exactly what happened at t+24 of the headon4 run,
# publishing a cluster 6.93 m away when the bbox said 3.56 m.
#
# Scoring both channels jointly fixes it:
#   cost = bearing_err_deg / ANGLE_GATE_DEG
#          + RANGE_COST_WEIGHT * |ln(track_range / bbox_range)|
# Both terms are dimensionless; the log makes the range term symmetric,
# so 2x too far costs the same as 2x too near.
#
# MEASURED on headon4, 122 camera detections, scored by distance from
# the camera-predicted person position:
#   bearing argmin (original)      median 0.84 m  p75 3.37  57% <1.5 m
#   2:1 gate then bearing argmin   median 0.49 m  p75 2.06  67% <1.5 m
#   joint cost, weight 1.0         median 0.33 m  p75 0.78  92% <1.5 m
#   joint cost, weight 2.0         median 0.33 m  p75 0.70 100% <1.5 m
#   oracle (nearest to prediction) median 0.33 m  p75 0.70 100% <1.5 m
# At weight 2.0 the rule matches the oracle on every detection. Note
# what that means: the range term dominates and bearing only breaks
# ties, which inverts the original bearing-first design.
#
# The ratio gate above still applies first - this only reorders what
# survives it. Set to 0 to fall back to bearing-argmin scoring.
RANGE_COST_WEIGHT = 2.0

# THESIS FIX (position-based re-ID fallback) - see module docstring.
# Generous radius: covers plausible walking distance during the gap
# between a binding being dropped and a fresh detection landing nearby,
# not a tight "same spot" check.
# THESIS ADDITION: allow a moving lidar track to reclaim an orphaned
# identity without camera corroboration. Set False to restore the
# camera-only re-ID path exactly as it was.
REID_ON_LIDAR_MOTION = False

# THESIS TUNE (measured, hallway_7m_06). The first version of the
# lidar re-ID guard reused CAMERA_ONLY_MOVE_THRESHOLD (0.08 m) and
# FAILED: a static cluster at (-0.890, 9.793) claimed an identity and
# republished that exact point for 2.3 s. A furniture centroid jitters
# more than 8 cm between scans as edge points come and go, so a
# per-scan displacement test cannot separate it from a person.
# Cumulative span since first-seen does. From the detector's own ids
# in that run:
#     person:    5.35 m, 7.60 m, 8.52 m
#     furniture: 0.28 m, 0.54 m
# 1.5 m sits in the empty middle of that gap. Raise if furniture still
# leaks; lower only with a bag that shows a real person rejected.
REID_MIN_SPAN_M = 1.5   # m, cumulative since the lidar id first appeared

# THESIS ADDITION (lidar-only person, real robot 9 Sep). The camera is
# pitched at the floor and confirms a person only at ~2-3 m; at a
# 1.5 m/s closing speed that leaves ~1.5 s - not enough to step aside
# at 0.31 m/s, so MPPI brakes/reverses instead. Lidar already tracks
# the person from 6-7 m. Allow an UNBOUND lidar track to be allocated
# a stable identity without the camera when it (a) has travelled
# REID_MIN_SPAN_M since first seen (furniture cannot fake this), (b)
# moves at walking speed over the last LIDAR_ONLY_WINDOW_S, and (c)
# is heading toward the robot within LIDAR_ONLY_APPROACH_DEG. Camera
# confirmation adopts the same stable id when it arrives, so nothing
# downstream sees an identity switch.
LIDAR_ONLY_PERSON_ENABLE = True
LIDAR_ONLY_WINDOW_S = 1.5        # s; velocity + straightness measured over at least this long
# THESIS FIX (rotation phantoms, 9 Sep): 85 lidar-only allocations in one
# run, nearly all on fixed objects that 'walked' +-1 m in sync with the
# robot's turns (TF lag at ~0.9 rad/s moves static clusters in map).
# Speed + approach angle cannot separate that from a person; these two can.
LIDAR_ONLY_MAX_YAW_RATE = 0.8    # rad/s (was 0.3; robot exceeds 0.3 for ~28% of a run and hardest during detours, so the gate suppressed lidar-only tracking exactly when it mattered - walk 2 lost the person 1.65 s at closest approach and scored 0.20 m vs 0.55 m. Offline sweep on fable6: held events 108 -> 14, phantom allocations 1 -> 2)
LIDAR_ONLY_MIN_STRAIGHT = 0.7    # net displacement / path length over the window
LIDAR_ONLY_MIN_NET_M = 0.8       # m; net displacement over the window
LIDAR_ONLY_MIN_SPEED = 0.4       # m/s
LIDAR_ONLY_MAX_SPEED = 2.0       # m/s
LIDAR_ONLY_APPROACH_DEG = 70.0   # deg; angle between velocity and vector to robot
LIDAR_ONLY_DUP_RADIUS_M = 1.0    # m; no new lidar-only identity this close to an already-bound lidar track
LIDAR_ONLY_MIN_SPAN_M = 0.5      # m; detector already requires 1.5 m to confirm - re-requiring REID_MIN_SPAN_M here cost ~1.5 s and put first sight at ~3 m

ORPHAN_REATTACH_RADIUS = 1.5   # m
ORPHAN_REATTACH_WINDOW = 8.0   # s; how long an orphan stays reclaimable

# THESIS FIX (cluttered-room clustering instability): in a cluttered
# room, lidar_person_detector.py's own width/motion filters still let
# 15-20+ short-lived clutter fragments (chair/table legs, cable bundles)
# pass as "candidates" on every scan - confirmed this session: dozens of
# raw lidar ids appearing and vanishing within 0.1-0.6s each, in the
# same window a real, high-confidence camera detection was present. A
# genuine camera-confirmed person can end up bearing-matched to one of
# these flickering fragments purely by bad timing, allocating a BRAND
# NEW stable identity anchored on a fragment instead of adopting the
# person's real, already-established lidar anchor - the new identity
# then drifts based on the fragment's noisy centroid rather than
# following the actual person.
#
# A real person's lidar cluster is continuously alive well before any
# camera match happens; a flickering fragment almost never survives
# this long. Require a lidar track to have existed continuously for at
# least MIN_LIDAR_ANCHOR_AGE before it's eligible to anchor a BRAND NEW
# identity. This does not affect sticky_lidar() or the re-ID paths -
# those only ever reuse an anchor that already proved itself by
# donating an identity in the past, which this gate does not touch.
MIN_LIDAR_ANCHOR_AGE = 1.0    # s

# THESIS FIX (camera-only false positive - static visual feature
# mistaken for a person): confirmed this session that YOLO-pose can
# repeatedly, confidently detect a static visual feature (a door frame's
# vertical jambs, spanning near-full frame height) as a person WITH all
# 4 leg keypoints, at high confidence (0.76-0.86), continuously
# re-triggering "camera_confirmed" - with NO lidar ever corroborating
# it, since there are no actual legs there to cluster. A real, close
# person tall enough to fill the frame would almost certainly also
# register on lidar; total absence of lidar corroboration is itself
# the signal something is wrong.
#
# A camera-only identity (no lidar anchor at creation) is given
# CAMERA_ONLY_GRACE_PERIOD seconds to get corroborated by SOME lidar
# match. If it never does, it's dropped as an untrusted visual-only
# detection rather than being trusted indefinitely.
CAMERA_ONLY_GRACE_PERIOD = 4.0   # s

# THESIS FIX (camera-only false positive gate, hardened): the plain
# "any lidar match clears the flag" version above was too weak - a
# static false positive (e.g. the door) can sit right next to
# unrelated, ALSO-spurious flickering lidar clutter, so it kept getting
# "corroborated" by coincidence before the grace period expired.
# Confirmed this session: the lidar cluster "confirming" a phantom near
# a door was itself independently flagged by lidar_person_detector.py
# as a non-moving/static candidate at the same time.
#
# Require the corroborating lidar track to show REAL displacement from
# its own first-seen position, not just exist - same movement concept
# lidar_person_detector.py already uses for its own static-object
# rejection (that file's static_move_threshold). A genuinely stationary
# fragment (real or spurious) can never satisfy this, however long it
# persists; a real person's lidar cluster accumulates this much
# displacement within a stride or two.
CAMERA_ONLY_MOVE_THRESHOLD = 0.08   # m

# THESIS FIX (respawn loop, bearing-based) - see module docstring. A
# BROKEN first attempt keyed the blacklist by raw camera track_id -
# ByteTrack recycles those integers, and confirmed this session that a
# genuinely different, real walking person got assigned the exact same
# id (2) only ~25.6s after the door's id:2 was blacklisted, well inside
# a 30s cooldown, silently discarding every one of their real
# detections. track_id carries no information about WHERE a detection
# is; bearing does. Blacklist the DIRECTION a drop happened at instead,
# with a tight angular gate so an unrelated real person elsewhere is
# never affected even while the exact same integer id is reused.
CAMERA_TRACK_BLACKLIST_COOLDOWN = 15.0   # s; shortened now that this is
                                          # direction-scoped rather than
                                          # id-scoped - a real person
                                          # briefly crossing the exact
                                          # blacklisted bearing during
                                          # this window is the residual
                                          # tradeoff, judged less bad
                                          # than the respawn loop.
BEARING_BLACKLIST_GATE_DEG = 6.0   # deg; tight - only re-blocks
                                    # detections at essentially the
                                    # SAME viewing direction as the drop


class IdentityFusionNode(Node):
    def __init__(self):
        super().__init__("identity_fusion_node")

        self.declare_parameter("camera_topic", "/person_positions_map")
        self.declare_parameter("lidar_topic", "/lidar_person_clusters")
        self.declare_parameter("output_topic", "/person_positions_fused")
        self.declare_parameter("angle_gate_deg", ANGLE_GATE_DEG)
        self.declare_parameter("fov_gate_deg", FOV_GATE_DEG)
        # Exposed for sweeping: 0.3 s removed all frozen plateaus on
        # trial_horizon200 but cut published messages 59 -> 12 (~1.3 Hz),
        # thin for a critic with track_timeout 0.5 s.
        self.declare_parameter("lidar_position_max_age", LIDAR_POSITION_MAX_AGE)
        # Exposed for the bbox-bias test: every bbox in headon_ray3 is
        # clipped at the top (y1=0), so bbox height underestimates the
        # person and the derived range OVERestimates their distance.
        # Scoring range jointly with bearing then favours a more distant
        # static cluster over the real person. Set 0 to score on bearing
        # alone and isolate the range term.
        self.declare_parameter("range_cost_weight", RANGE_COST_WEIGHT)
        self.declare_parameter("range_ratio_gate", RANGE_RATIO_GATE)
        self.declare_parameter("camera_preview_width", CAMERA_PREVIEW_WIDTH)
        self.declare_parameter("camera_fx", CAMERA_FX)
        self.declare_parameter("camera_cx", CAMERA_CX)
        self.declare_parameter("camera_yaw_offset_deg", CAMERA_YAW_OFFSET_DEG)
        self.declare_parameter("camera_x_offset", CAMERA_X_OFFSET)

        self.camera_topic = self.get_parameter("camera_topic").value
        self.lidar_topic = self.get_parameter("lidar_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.angle_gate_rad = math.radians(
            float(self.get_parameter("angle_gate_deg").value))
        self.fx = float(self.get_parameter("camera_fx").value)
        self.cx = float(self.get_parameter("camera_cx").value)
        self.cam_yaw = math.radians(
            float(self.get_parameter("camera_yaw_offset_deg").value))
        self.cam_x = float(self.get_parameter("camera_x_offset").value)
        self.preview_width = float(
            self.get_parameter("camera_preview_width").value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # camera track_id -> [stable_id, last_seen]
        self.stable_of_camera = {}
        # lidar_id -> [stable_id, last_confirmed]   (camera-confirmed only)
        self.stable_of_lidar = {}
        # lidar_id -> (x, y, last_seen)
        self.lidar_tracks = {}
        # THESIS FIX (cluttered-room clustering instability): lidar_id
        # -> time first seen (continuous existence, not just latest
        # update) - see MIN_LIDAR_ANCHOR_AGE above.
        self.lidar_first_seen = {}
        # THESIS ADDITION (lidar-only person): lid -> [(x, y, t), ...]
        self.lidar_hist = {}
        self._yaw_prev = None      # (yaw, t) for robot yaw-rate estimate
        self._yaw_rate = 0.0
        # Running max displacement of each lidar id from where it was
        # first seen - the re-ID guard, see REID_MIN_SPAN_M.
        self.lidar_span = {}

        # THESIS FIX (bearing-only matching flapping): nearest_lidar()
        # re-picks the single best-bearing lidar cluster from scratch
        # every camera_callback, with zero range disambiguation. When
        # two people stand at similar bearings from the robot - exactly
        # the "close pair" scenario this pipeline targets - small noise
        # in either bearing estimate can flip which lidar cluster wins
        # the argmin from one cycle to the next, silently reassigning an
        # already-known camera track's STABLE ID to a DIFFERENT PHYSICAL
        # PERSON's position. Measured effect: /predicted_person_positions
        # showing the same track_id's position jump ~1-2m between
        # consecutive samples, apparent speed 1-2 m/s for two stationary
        # people, permanently failing social_group_detector's
        # CONV_MAX_SPEED gate.
        #
        # Fix: once a camera track is bound to a lidar track, stay bound
        # to that SAME lidar_id next cycle as long as it's still within
        # the angle gate, instead of re-searching argmin every time. Only
        # fall back to a fresh search if the sticky candidate has left
        # the gate or its lidar track vanished (genuine loss, not noise).
        self.lidar_of_camera = {}

        self.next_stable_id = 0
        self.reid_count = 0

        # THESIS FIX (position-based re-ID fallback) - see module
        # docstring. stable_id -> (x, y, orphaned_at).
        self.orphaned_identities = {}

        # THESIS FIX (camera-only false positive gate) - see module
        # docstring. stable_id -> (created_at, bearing_at_creation).
        # Bearing kept alongside so a drop can blacklist the DIRECTION,
        # not just the (reusable) camera track_id - see
        # BEARING_BLACKLIST_GATE_DEG. Removed the moment any lidar
        # corroboration is found for that stable_id; if it's still
        # here past CAMERA_ONLY_GRACE_PERIOD, the identity is dropped
        # as an untrusted visual-only detection.
        self.camera_only_since = {}

        # stable_id -> last time ANY message (camera or lidar-only) was
        # published under it. Used only to space out lidar-only coasting
        # publishes from a just-happened camera-confirmed one.
        self.last_published = {}

        # THESIS FIX (unvalidated coasting) - stable_id -> (x, y, t) of
        # the last CAMERA-CONFIRMED position. publish_lidar_only() below
        # checks candidate coast positions against this; without it the
        # node republishes whatever cluster the identity is bound to,
        # with no test that it is still a plausible place for that
        # person to be. See MAX_COAST_SPEED for the measurement.
        self.last_confirmed_pos = {}

        # THESIS FIX (identity teleporting) - stable_id -> (x, y, t) of
        # the last position published under that identity from EITHER
        # path. Distinct from last_confirmed_pos, which tracks only
        # camera-confirmed positions and anchors the coasting check.
        # This one enforces continuity on every publish.
        self.last_identity_pos = {}

        # THESIS FIX (moving-track exemption) - stable_id -> list of
        # (x, y, t) published positions (last few seconds), and
        # stable_id -> last time the track qualified as moving.
        self.pub_hist = {}
        self.last_moving = {}

        # THESIS FIX (respawn loop, bearing-based) - see module
        # docstring. list of (bearing, blacklisted_at) - NOT keyed by
        # camera track_id, since those get recycled by ByteTrack.
        self.bearing_blacklist = []

        self.create_subscription(
            String, self.lidar_topic, self.lidar_callback, 10)
        self.create_subscription(
            String, self.camera_topic, self.camera_callback, 10)

        self.pub = self.create_publisher(String, self.output_topic, 10)

        self.create_timer(1.0, self.prune)
        self.create_timer(LIDAR_ONLY_PUBLISH_PERIOD, self.publish_lidar_only)

        self.get_logger().info("Identity fusion node started")
        self.get_logger().info(f"Camera in : {self.camera_topic}")
        self.get_logger().info(f"Lidar  in : {self.lidar_topic}")
        self.get_logger().info(f"Fused out: {self.output_topic}")
        self.get_logger().info(
            f"Camera-lidar angle gate: {math.degrees(self.angle_gate_rad):.1f} deg "
            f"(PROVISIONAL - calibrate from 'pair-bearing' log lines)")
        self.get_logger().info(
            f"Occlusion budget (lidar binding timeout): "
            f"{LIDAR_BINDING_TIMEOUT:.0f} s")
        self.get_logger().info(
            f"Position-based re-ID fallback: radius={ORPHAN_REATTACH_RADIUS:.2f}m "
            f"window={ORPHAN_REATTACH_WINDOW:.1f}s")
        self.get_logger().info(
            f"Camera-only false-positive grace period: "
            f"{CAMERA_ONLY_GRACE_PERIOD:.1f}s")

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # -----------------------------------------------------------------
    def lidar_callback(self, msg):
        parts = msg.data.split(",")
        # THESIS FIX (field mismatch): lidar_person_detector.py publishes
        # "id,conf,x,y" (conf is a placeholder, currently always 1.00 -
        # see that file's comments). This previously assumed "id,x,y",
        # so parts[1] (conf) was read as x and parts[2] (real x) was
        # read as y - every LIDAR position landed in the wrong place
        # with no error raised. Kept conf in the message (not stripped
        # from the publisher) since it's the intended slot for a future
        # LIDAR detection-quality metric.
        if len(parts) < 4:
            return
        try:
            lid = int(float(parts[0]))
            x = float(parts[2])
            y = float(parts[3])
        except ValueError:
            return
        now = self.now()
        # THESIS FIX (cluttered-room clustering instability +
        # camera-only false positive gate): track each lidar id's
        # first-seen (time, x, y), not just time - the position is
        # needed to check REAL movement, not just continuous existence,
        # before trusting this cluster as corroboration for a
        # camera-only identity (see CAMERA_ONLY_MOVE_THRESHOLD below).
        # Preserve the original first_seen across updates - only set it
        # once, when this id is genuinely new.
        if lid in self.lidar_tracks:
            first_seen = self.lidar_first_seen.get(lid, (now, x, y))
        else:
            first_seen = (now, x, y)
        self.lidar_first_seen[lid] = first_seen
        self.lidar_tracks[lid] = (x, y, now)

        # THESIS ADDITION (lidar-primary re-ID across detector
        # fragmentation).
        #
        # MEASURED on hallway_7m_06: the detector emitted 32 distinct
        # ids in 50.7 s; the person alone accounted for at least six
        # (30, 33, 37, 62, 61, 78) in an unbroken chain t=0.2-50.7.
        # Every fragmentation orphaned the stable identity, because
        # publish_lidar_only() skips a binding whose lidar id has
        # vanished and find_orphan_to_reattach() was reachable ONLY
        # from camera_callback. With the camera dark (5.5 s of camera
        # in that run - the mount is pitched at the floor) orphans
        # expired unclaimed and fused output covered 3.8 s of 50.7.
        #
        # Lidar had the person throughout. So allow a NEW lidar track
        # to reclaim a recent orphan without waiting for the camera.
        #
        # GUARD - do not weaken: only a track that has genuinely MOVED
        # may claim an identity. 26 of those 32 ids were furniture, and
        # a static cluster sat at (-1.21, 9.45) while the person passed
        # through (-1.29, 9.52) - 0.11 m apart, well inside
        # ORPHAN_REATTACH_RADIUS (1.5 m). Existence is not evidence;
        # displacement is. This is the same test used at
        # CAMERA_ONLY_MOVE_THRESHOLD for camera-only corroboration.
        # Accumulate span on EVERY scan, not only when an orphan is
        # pending - otherwise a person who walked 8 m before anyone
        # was orphaned reads as span 0 at the moment it matters.
        _fs_t, fs_x, fs_y = first_seen
        span = max(self.lidar_span.get(lid, 0.0),
                   math.hypot(x - fs_x, y - fs_y))
        self.lidar_span[lid] = span

        if (REID_ON_LIDAR_MOTION
                and lid not in self.stable_of_lidar
                and self.orphaned_identities):
            if span >= REID_MIN_SPAN_M:
                sid = self.find_orphan_to_reattach(x, y, now)
                if sid is not None:
                    self.stable_of_lidar[lid] = (sid, now)
                    self.last_confirmed_pos[sid] = (x, y, now)
                    self.get_logger().info(
                        f"lidar re-ID: new lidar {lid} (span "
                        f"{span:.2f}m) reclaimed stable {sid} "
                        f"at ({x:.2f}, {y:.2f})")

        # THESIS ADDITION (lidar-only person) - see LIDAR_ONLY_PERSON_ENABLE.
        if LIDAR_ONLY_PERSON_ENABLE:
            h = self.lidar_hist.setdefault(lid, [])
            h.append((x, y, now))
            while h and now - h[0][2] > LIDAR_ONLY_WINDOW_S * 3:
                h.pop(0)
            if lid not in self.stable_of_lidar and span >= LIDAR_ONLY_MIN_SPAN_M:
                old = [p for p in h if now - p[2] >= LIDAR_ONLY_WINDOW_S]
                pose = self.robot_pose()
                yaw_ok = True
                if pose is not None:
                    _yaw = pose[2]
                    if self._yaw_prev is not None:
                        _dt = now - self._yaw_prev[1]
                        if _dt > 1e-3:
                            self._yaw_rate = abs(
                                self.wrap_angle(_yaw - self._yaw_prev[0])) / _dt
                    self._yaw_prev = (_yaw, now)
                    yaw_ok = self._yaw_rate < LIDAR_ONLY_MAX_YAW_RATE
                    if not yaw_ok:
                        self.get_logger().info(
                            f"lidar-only held: robot turning "
                            f"{self._yaw_rate:.2f} rad/s",
                            throttle_duration_sec=2.0)
                if old and pose is not None and yaw_ok:
                    ox, oy, ot = old[-1]
                    dt = max(now - ot, 1e-3)
                    seg = [p for p in h if p[2] >= ot]
                    path_len = sum(math.hypot(seg[i][0] - seg[i - 1][0],
                                              seg[i][1] - seg[i - 1][1])
                                   for i in range(1, len(seg)))
                    net = math.hypot(x - ox, y - oy)
                    straight = (net / path_len) if path_len > 1e-3 else 0.0
                    vx, vy = (x - ox) / dt, (y - oy) / dt
                    speed = math.hypot(vx, vy)
                    rx, ry, _ryaw = pose
                    dx, dy = rx - x, ry - y
                    dist = math.hypot(dx, dy)
                    if (LIDAR_ONLY_MIN_SPEED <= speed <= LIDAR_ONLY_MAX_SPEED
                            and net >= LIDAR_ONLY_MIN_NET_M
                            and straight >= LIDAR_ONLY_MIN_STRAIGHT
                            and dist > 1e-3):
                        cos_ang = (vx * dx + vy * dy) / (speed * dist)
                        if cos_ang >= math.cos(math.radians(LIDAR_ONLY_APPROACH_DEG)):
                            # dedup: a bound lidar track already nearby means
                            # this is a fragment of the same person
                            for olid in self.stable_of_lidar:
                                if olid == lid or olid not in self.lidar_tracks:
                                    continue
                                ox2, oy2, ot2 = self.lidar_tracks[olid]
                                if (now - ot2 < 1.0 and math.hypot(x - ox2, y - oy2)
                                        < LIDAR_ONLY_DUP_RADIUS_M):
                                    self.get_logger().info(
                                        f"lidar-only skipped: lidar {lid} is "
                                        f"{math.hypot(x - ox2, y - oy2):.2f} m from "
                                        f"bound lidar {olid} (stable "
                                        f"{self.stable_of_lidar[olid][0]})",
                                        throttle_duration_sec=2.0)
                                    return
                            sid = self.find_orphan_to_reattach(x, y, now)
                            if sid is None:
                                sid = self.allocate()
                            self.stable_of_lidar[lid] = (sid, now)
                            self.last_confirmed_pos[sid] = (x, y, now)
                            self.last_moving[sid] = now
                            self.get_logger().info(
                                f"lidar-only person: lidar {lid} -> new stable "
                                f"{sid} at ({x:.2f}, {y:.2f}) span {span:.2f}m "
                                f"speed {speed:.2f} m/s straight {straight:.2f} approaching "
                                f"{math.degrees(math.acos(max(-1.0, min(1.0, cos_ang)))):.0f} deg "
                                f"range {dist:.2f} m")

    # -----------------------------------------------------------------
    def robot_pose(self):
        """(x, y, yaw) of base_link in map, or None if TF isn't ready."""
        try:
            tf = self.tf_buffer.lookup_transform(
                MAP_FRAME, BASE_FRAME, Time())
        except Exception:
            return None

        rx = tf.transform.translation.x
        ry = tf.transform.translation.y
        q = tf.transform.rotation
        # yaw from quaternion (planar robot, roll/pitch ~0)
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return rx, ry, yaw

    @staticmethod
    def wrap_angle(a):
        while a > math.pi:
            a -= 2 * math.pi
        while a < -math.pi:
            a += 2 * math.pi
        return a

    def pixel_center_to_bearing(self, x1, x2):
        """Bbox horizontal center -> bearing relative to robot heading.

        Assumes the camera's optical axis is aligned with the robot's
        forward (base_link x) direction - true for the TurtleBot4's
        fixed forward-facing OAK-D mount. Positive = person is to the
        robot's left, matching the standard CCW-positive yaw convention.
        """
        u = (x1 + x2) / 2.0
        return -math.atan((u - self.cx) / self.fx) + self.cam_yaw
    
    def track_bearing_from_camera(self, lx, ly, pose):
        rx, ry, ryaw = pose
        dx, dy = lx - rx, ly - ry
        bx = math.cos(ryaw) * dx + math.sin(ryaw) * dy - self.cam_x
        by = -math.sin(ryaw) * dx + math.cos(ryaw) * dy
        return math.atan2(by, bx)
    
    def sticky_lidar(self, cam_id, cam_bearing_rel, bbox_range=None):
        """If cam_id was bound to a lidar track last cycle and that track
        is still within the angle gate, keep using it - avoids the
        argmin flip-flop when two lidar tracks have close bearings.
        Returns (lidar_id, ang_diff) or (None, None) if no sticky match
        applies (never bound before, or genuinely out of gate now).

        THESIS FIX: the range gate applies here too. Stickiness is what
        turns a single bad pick into a whole-encounter error - on bag
        headon2 the identity stayed on a static cluster at 7.4 m while
        the person walked past at 2.4 m, because the bearing stayed
        inside the gate the entire time.
        """
        prev_lid = self.lidar_of_camera.get(cam_id)
        if prev_lid is None or prev_lid not in self.lidar_tracks:
            return None, None

        pose = self.robot_pose()
        if pose is None:
            return None, None
        rx, ry, ryaw = pose

        lx, ly, _ = self.lidar_tracks[prev_lid]
        track_bearing_rel = self.track_bearing_from_camera(lx, ly, pose)
        diff = abs(self.wrap_angle(track_bearing_rel - cam_bearing_rel))

        if diff < self.angle_gate_rad:
            if not self.fov_ok(lx, ly, rx, ry, ryaw):
                self.get_logger().info(
                    f"sticky binding cam:{cam_id} -> lidar:{prev_lid} "
                    f"dropped on FOV gate "
                    f"({math.degrees(track_bearing_rel):+.1f} deg)")
                return None, None
            if not self.range_plausible(lx, ly, rx, ry, bbox_range):
                self.get_logger().info(
                    f"sticky binding cam:{cam_id} -> lidar:{prev_lid} "
                    f"dropped on range gate (track "
                    f"{math.hypot(lx - rx, ly - ry):.2f} m vs bbox "
                    f"{bbox_range:.2f} m)")
                return None, None
            return prev_lid, diff
        return None, None

    def fov_ok(self, lx, ly, rx, ry, ryaw):
        """Is a lidar track inside the camera's field of view?

        See FOV_GATE_DEG. Returns True when the gate is disabled, so
        callers degrade to the previous behaviour rather than rejecting
        everything.
        """
        gate = self.get_parameter("fov_gate_deg").value
        if gate is None or gate <= 0:
            return True
        rel = self.track_bearing_from_camera(lx, ly, (rx, ry, ryaw))
        return abs(math.degrees(rel)) <= gate

    def range_plausible(self, lx, ly, rx, ry, bbox_range):
        """Is a lidar track's range consistent with the camera's bbox-height
        range estimate, to within RANGE_RATIO_GATE?

        Returns True when the gate is disabled or no estimate is available,
        so callers degrade to the previous bearing-only behaviour rather
        than rejecting everything.
        """
        gate = self.get_parameter("range_ratio_gate").value
        if gate is None or gate <= 0 or bbox_range is None:
            return True
        track_range = math.hypot(lx - rx, ly - ry)
        if track_range <= 0.01:
            return True
        ratio = track_range / bbox_range
        return (1.0 / gate) <= ratio <= gate

    def nearest_lidar(self, cam_bearing_rel, bbox_range=None):
        """Nearest cached lidar track within ANGLE_GATE -> (lidar_id, ang_diff).

        Converts each lidar track's map-frame (x,y) into a bearing
        relative to the robot's current heading via TF, then compares
        against the camera's pixel-derived relative bearing.

        THESIS FIX: candidates whose range is inconsistent with the
        camera's bbox-height estimate are excluded before the argmin -
        see RANGE_RATIO_GATE. Without this the argmin selected a static
        cluster in 66% of matches on bag headon2.
        """
        pose = self.robot_pose()
        if pose is None:
            return None, None
        rx, ry, ryaw = pose

        best_id, best_diff, best_cost = None, None, None
        rejected = 0
        # THESIS FIX (stale candidates in the search): prune() keeps a
        # cache entry for LIDAR_TRACK_TIMEOUT (2.0 s), but a position
        # older than lidar_position_max_age will be refused downstream.
        # Selecting one here means discarding the cycle while a fresher
        # candidate sat in the same dict. Measured on static_walk_04:
        # 119 stale-position rejections and a doubled match-rejected
        # count, both from stale entries winning the argmin.
        _now = self.now()
        _max_age = self.get_parameter("lidar_position_max_age").value
        for lid, (lx, ly, _stamp) in self.lidar_tracks.items():
            if _now - _stamp > _max_age:
                continue
            track_bearing_rel = self.track_bearing_from_camera(lx, ly, pose)
            diff = abs(self.wrap_angle(track_bearing_rel - cam_bearing_rel))
            if diff >= self.angle_gate_rad:
                continue
            if not self.fov_ok(lx, ly, rx, ry, ryaw):
                continue
            if not self.range_plausible(lx, ly, rx, ry, bbox_range):
                rejected += 1
                continue
            # THESIS FIX: score bearing and range jointly instead of
            # gating on range and then taking the bearing argmin - see
            # RANGE_COST_WEIGHT.
            cost = diff / self.angle_gate_rad
            rcw = self.get_parameter("range_cost_weight").value
            if rcw > 0 and bbox_range:
                track_range = math.hypot(lx - rx, ly - ry)
                if track_range > 0.01:
                    cost += rcw * abs(
                        math.log(track_range / bbox_range))
            if best_cost is None or cost < best_cost:
                best_id, best_diff, best_cost = lid, diff, cost
        if rejected and best_id is None:
            self.get_logger().info(
                f"no candidate within range gate ({rejected} rejected, "
                f"bbox range {bbox_range:.2f} m)")
        return best_id, best_diff

    def stable_id_in_use(self, stable_id, exclude_cam_id, now, current_lid=None):
        for cam_id, (sid, last) in self.stable_of_camera.items():
            if cam_id == exclude_cam_id:
                continue
            if sid == stable_id and (now - last) < CAMERA_ACTIVE_WINDOW:
                # THESIS FIX: don't block re-ID just because the OLD
                # camera track was updated recently - if its own lidar
                # anchor has already vanished, it's very likely the
                # same physical person handed off to a new ByteTrack
                # id, not a second person genuinely still being seen.
                # Only treat this as a real conflict if the old track
                # still has a live lidar anchor backing it up.
                old_lidar = self.lidar_of_camera.get(cam_id)
                if old_lidar is None or old_lidar not in self.lidar_tracks:
                    continue
                # THESIS FIX (ByteTrack fragmentation): if the
                # "conflicting" old camera track is anchored on the
                # EXACT SAME lidar cluster as the new one, this cannot
                # genuinely be two different people - one lidar cluster
                # represents one set of legs, so it cannot simultaneously
                # belong to two real, physically-separate people.
                # Confirmed this session: ByteTrack fragmented one
                # walking person into three consecutive camera ids
                # (cam:2, cam:3, cam:4) all bearing-matching the SAME
                # lidar:28 within ~1.5s, and this guard's original,
                # broader check ("stable id reported recently by ANY
                # other camera track") blocked every one of them from
                # merging, producing three separate stable identities
                # (and three separate KF tracks, each restarting its own
                # velocity warm-up) for one physical person. Sharing the
                # identical lidar id is itself strong enough evidence of
                # fragmentation to allow the merge, while two DIFFERENT
                # lidar ids (the real-second-person case this guard
                # exists for) still correctly blocks it below.
                if current_lid is not None and old_lidar == current_lid:
                    continue
                return True
        return False

    def allocate(self):
        sid = self.next_stable_id
        self.next_stable_id += 1
        return sid

    # -----------------------------------------------------------------
    def find_orphan_to_reattach(self, x, y, now):
        """THESIS FIX (position-based re-ID fallback) - see module
        docstring. Nearest recently-orphaned stable identity within
        ORPHAN_REATTACH_RADIUS/ORPHAN_REATTACH_WINDOW of (x, y), or
        None. Removes it from the orphan pool once claimed so it can't
        be reattached to two different new tracks."""
        best_sid, best_d = None, ORPHAN_REATTACH_RADIUS
        for sid, (ox, oy, orphaned_at) in self.orphaned_identities.items():
            if now - orphaned_at > ORPHAN_REATTACH_WINDOW:
                continue
            d = math.hypot(x - ox, y - oy)
            if d < best_d:
                best_sid, best_d = sid, d
        if best_sid is not None:
            del self.orphaned_identities[best_sid]
            self.get_logger().info(
                f"Orphan reattach candidate: stable {best_sid} "
                f"({best_d:.2f}m from its last known position)")
        return best_sid

    # -----------------------------------------------------------------
    def camera_callback(self, msg):
        # THESIS CHANGE: yolo_leg_detector_lidar.py no longer computes
        # depth/map position - it publishes pixel-space detections only:
        #   track_id,conf,x1,y1,x2,y2,leg_kpts
        # (leg_kpts is "px:py;px:py;..." for 0-4 visible leg keypoints,
        # may be empty; harmless here since we split on "," and it's
        # already the last field with no further commas inside it.)
        parts = msg.data.split(",")
        if len(parts) < 6:
            self.get_logger().warn(f"Invalid camera msg: {msg.data}")
            return

        try:
            cam_id = int(float(parts[0]))
            conf = parts[1]
            x1 = float(parts[2])
            y1 = float(parts[3])
            x2 = float(parts[4])
            y2 = float(parts[5])
        except ValueError:
            self.get_logger().warn(f"Parse failed: {msg.data}")
            return

        # THESIS ADDITION (bearing from leg keypoints, not bbox centre).
        # yolo_leg_detector_lidar.py already publishes COCO keypoints
        # 13-16 (knees, ankles) as "px:py;..." in parts[6]; they were
        # documented above but never parsed. The bbox centre is a poor
        # bearing source on this robot: the camera is pitched at the
        # floor, so the box is cropped to knees-and-boots and its
        # horizontal centre shifts with stance and with which body
        # parts happen to be in frame. Ankles are at the RPLIDAR's own
        # scan height, so their bearing refers to the SAME physical
        # feature the lidar clusters on - and they stay visible in a
        # floor-pointed frame when the torso does not.
        # Falls back to the bbox centre whenever no keypoint is
        # available, so behaviour is unchanged for those detections.
        # Set USE_KEYPOINT_BEARING = False to restore the old path.
        kpt_px = []
        if len(parts) >= 7 and parts[6].strip():
            for pair in parts[6].split(";"):
                if ":" not in pair:
                    continue
                try:
                    px, py = pair.split(":")
                    kpt_px.append((float(px), float(py)))
                except ValueError:
                    continue

        now = self.now()
        if USE_KEYPOINT_BEARING and kpt_px:
            # Prefer the lowest keypoints in the frame (largest py):
            # ankles sit below knees and are closest to the scan plane.
            kpt_px.sort(key=lambda q: -q[1])
            use = kpt_px[:2]
            cx = sum(q[0] for q in use) / len(use)
            cam_bearing = self.pixel_center_to_bearing(cx, cx)
            bearing_src = f"kpt{len(use)}"
        else:
            cam_bearing = self.pixel_center_to_bearing(x1, x2)
            bearing_src = "bbox"

        # THESIS FIX: rough range from bbox height, used only as a ratio
        # gate on lidar candidates (see RANGE_RATIO_GATE). fy is taken
        # from the measured intrinsics; the preview is square so
        # fy ~= fx ~= 287.31 for a 250x250 frame (calibrated 11 Sep).
        bbox_h = y2 - y1
        # THESIS FIX (clipped bbox poisons the range estimate). The
        # camera is mounted low and pitched at the floor, so the
        # person's head is above the frame and y1 is 0-2 px on
        # essentially every detection (confirmed on hallway_7m_06:
        # the visible body is knee-down). bbox_h is then a fraction
        # of the true height and PERSON_HEIGHT_M * fy / bbox_h
        # OVERestimates range badly - which then drives
        # range_plausible() and the RANGE_COST_WEIGHT term to pick
        # the wrong lidar cluster and publish it as camera_confirmed.
        # Measured on hallway_7m_06: fused jumped (-1.873, 9.624) ->
        # (-4.707, 11.071), 3.2 m in 2.2 s, both camera_confirmed.
        # A clipped box has no usable height, so report range as
        # unknown rather than wrong: bbox_range=None already means
        # "skip the ratio gate and the range cost" downstream,
        # degrading to bearing-only matching.
        # Root fix is mounting geometry - tilt the camera up.
        bbox_clipped = (BBOX_CLIP_TOP_PX >= 0 and y1 <= BBOX_CLIP_TOP_PX)
        bbox_range = (PERSON_HEIGHT_M * CAMERA_FY / bbox_h
                      if (bbox_h > 0 and not bbox_clipped) else None)

        # THESIS FIX (respawn loop, bearing-based) - see module
        # docstring. If a recent drop happened at essentially this
        # same viewing direction, ignore this detection entirely
        # rather than letting it immediately spawn a replacement
        # identity for what's very likely the same static source.
        for bl_bearing, bl_at in self.bearing_blacklist:
            if now - bl_at >= CAMERA_TRACK_BLACKLIST_COOLDOWN:
                continue
            if abs(self.wrap_angle(cam_bearing - bl_bearing)) < math.radians(
                    BEARING_BLACKLIST_GATE_DEG):
                return

        lid, diff = self.sticky_lidar(cam_id, cam_bearing, bbox_range)
        pair_src = "sticky, " + bearing_src
        if lid is None:
            lid, diff = self.nearest_lidar(cam_bearing, bbox_range)
            pair_src = bearing_src
        if lid is not None:
            # Logged so ANGLE_GATE_DEG, RANGE_RATIO_GATE and
            # RANGE_COST_WEIGHT can each be calibrated from the observed
            # distribution rather than left at their guesses.
            _pose = self.robot_pose()
            if _pose is not None and lid in self.lidar_tracks:
                _lx, _ly, _ = self.lidar_tracks[lid]
                _tr = math.hypot(_lx - _pose[0], _ly - _pose[1])
            else:
                _tr = None
            _rng = (f" trk={_tr:.2f}m" if _tr is not None else " trk=NA")
            _rng += (f" bbox={bbox_range:.2f}m" if bbox_range is not None
                     else " bbox=NA")
            _rng += (f" ratio={_tr / bbox_range:.2f}"
                     if (_tr is not None and bbox_range) else " ratio=NA")
            _rng += f" h={bbox_h}px"
            self.get_logger().info(
                f"pair-bearing cam:{cam_id} <-> lidar:{lid} = "
                f"{math.degrees(diff):.1f} deg ({pair_src}){_rng}")

        if lid is not None:
            self.lidar_of_camera[cam_id] = lid

        known = self.stable_of_camera.get(cam_id)

        if known is not None:
            stable_id = known[0]
            # THESIS FIX (lidar-only person adoption, 9 Sep): a camera
            # track born camera-only (short range, no lidar anchor)
            # kept ITS id when it later matched a lidar track that
            # already carried a lidar-only identity - measured: stable
            # 7 (with KF velocity) replaced by camera-only 4 (KF reset
            # to zero velocity) at 4.5 m, exactly when the prediction
            # mattered. If this camera identity has no lidar binding of
            # its own and the matched lidar already has one, adopt the
            # lidar's identity.
            if lid is not None and lid in self.stable_of_lidar:
                lid_sid = self.stable_of_lidar[lid][0]
                cam_sid_bound = any(s == stable_id
                                    for s, _t in self.stable_of_lidar.values())
                if (lid_sid != stable_id and not cam_sid_bound
                        and not self.stable_id_in_use(lid_sid, cam_id, now,
                                                      current_lid=lid)):
                    self.get_logger().info(
                        f"camera {cam_id}: stable {stable_id} (camera-only) "
                        f"-> adopting lidar {lid}'s stable {lid_sid}")
                    self.camera_only_since.pop(stable_id, None)
                    stable_id = lid_sid
        else:
            stable_id = None

            # --- the re-ID step (id-based) ------------------------------
            # A new camera id landed near a lidar track that still holds
            # a binding from an earlier, now-lost camera id. Adopt it.
            if lid is not None and lid in self.stable_of_lidar:
                candidate = self.stable_of_lidar[lid][0]
                if not self.stable_id_in_use(candidate, cam_id, now, current_lid=lid):
                    stable_id = candidate
                    self.reid_count += 1
                    self.get_logger().info(
                        f"RE-ID #{self.reid_count}: new camera track "
                        f"{cam_id} adopted stable id {stable_id} via "
                        f"lidar track {lid} (bearing diff "
                        f"{math.degrees(diff):.1f} deg)")
                else:
                    self.get_logger().info(
                        f"Lidar {lid} suggests stable {candidate} for "
                        f"camera {cam_id}, but that identity is already "
                        f"active on another camera track - allocating new")

            # --- the re-ID step (position-based fallback) ---------------
            # THESIS FIX - see module docstring. The id-based step above
            # only fires if THIS lidar id is already a stable_of_lidar
            # key. If lidar_person_detector's own clustering also
            # churned (a fresh lidar id for the same physical person),
            # fall back to checking whether this detection's
            # lidar-anchored position lands near a recently orphaned
            # identity instead. Only runs when the id-based step above
            # found nothing.
            if stable_id is None and lid is not None:
                lx, ly, _ = self.lidar_tracks[lid]
                orphan_sid = self.find_orphan_to_reattach(lx, ly, now)
                if orphan_sid is not None:
                    if not self.stable_id_in_use(orphan_sid, cam_id, now, current_lid=lid):
                        stable_id = orphan_sid
                        self.reid_count += 1
                        self.get_logger().info(
                            f"RE-ID #{self.reid_count} (position-based): "
                            f"new camera track {cam_id} adopted stable id "
                            f"{stable_id} via orphan reattachment near "
                            f"lidar track {lid}")
                    else:
                        self.get_logger().info(
                            f"Orphan stable {orphan_sid} near lidar {lid} "
                            f"is already active on another camera track "
                            f"- allocating new")

            if stable_id is None:
                # THESIS FIX (cluttered-room clustering instability): a
                # brand-new identity should only be anchored on a lidar
                # track that has existed continuously long enough to be
                # trustworthy - a flickering furniture fragment almost
                # never survives this long. Does NOT gate the id-based
                # or position-based re-ID paths above; those only ever
                # reuse an anchor that already proved itself by donating
                # an identity in the past.
                anchor_lid = lid
                if anchor_lid is not None:
                    fs_time, _fs_x, _fs_y = self.lidar_first_seen.get(
                        anchor_lid, (now, 0.0, 0.0))
                    age = now - fs_time
                    if age < MIN_LIDAR_ANCHOR_AGE:
                        self.get_logger().info(
                            f"Lidar {anchor_lid} too young to anchor a new "
                            f"identity ({age:.2f}s < {MIN_LIDAR_ANCHOR_AGE:.1f}s) "
                            f"- likely a fragment, not a person")
                        anchor_lid = None

                stable_id = self.allocate()
                self.get_logger().info(
                    f"New identity: camera track {cam_id} -> stable "
                    f"{stable_id}"
                    + (f" (anchored on lidar {anchor_lid})"
                       if anchor_lid is not None
                       else " (no lidar anchor in range)"))
                lid = anchor_lid

                # THESIS FIX (camera-only false positive gate) - see
                # module docstring. Start the grace-period clock only
                # for identities born with NO lidar anchor at all.
                if anchor_lid is None:
                    self.camera_only_since[stable_id] = (now, cam_bearing)

        self.stable_of_camera[cam_id] = [stable_id, now]

        # Confirm/refresh the lidar binding. THIS is what makes a lidar
        # track eligible to donate an identity later - and what keeps
        # never-confirmed wall fragments permanently ineligible.
        if lid is not None:
            self.stable_of_lidar[lid] = [stable_id, now]
            # THESIS FIX (camera-only false positive gate, hardened):
            # only clear the grace-period flag if this lidar anchor has
            # shown REAL movement from its first-seen position, not
            # merely existed - see CAMERA_ONLY_MOVE_THRESHOLD above for
            # why plain existence proved too weak a bar. If it hasn't
            # moved enough yet, leave the flag set; it may still clear
            # on a later cycle once real displacement accumulates, or
            # the grace period will expire and prune() will drop the
            # identity - the correct outcome for something that never
            # genuinely moves.
            if stable_id in self.camera_only_since:
                _fs_time, fs_x, fs_y = self.lidar_first_seen.get(lid, (now, 0.0, 0.0))
                lx_now, ly_now, _ = self.lidar_tracks[lid]
                moved = math.hypot(lx_now - fs_x, ly_now - fs_y)
                if moved >= CAMERA_ONLY_MOVE_THRESHOLD:
                    self.camera_only_since.pop(stable_id, None)

        # Position: LIDAR is the only real-world position source now
        # (camera node no longer computes depth/map position at all).
        # Without a lidar match this cycle there is no position to
        # report - skip publishing rather than emit a bogus (0,0).
        if lid is None:
            return

        lx, ly, lidar_stamp = self.lidar_tracks[lid]

        # THESIS FIX (stale position) - see LIDAR_POSITION_MAX_AGE.
        # The camera detection is fresh this cycle, but the position
        # attached to it comes from the bound lidar track's cache. If
        # that cache has not been refreshed recently the pair is a fresh
        # confirmation of a stale point - do not publish it.
        pos_age = now - lidar_stamp
        max_age = self.get_parameter("lidar_position_max_age").value
        if pos_age > max_age:
            self.get_logger().info(
                f"stale position id:{stable_id} lidar:{lid} "
                f"age {pos_age:.2f}s (> {max_age}s) - not publishing")
            # Drop the sticky binding, same as the identity-speed check
            # below. Without this the camera track stays welded to a
            # lidar id whose cache has gone stale: sticky_lidar() picks
            # it again next cycle, it is staler still, and nearest_lidar()
            # is never reached - so fresh clusters arriving at ~5 Hz are
            # never considered until prune() drops the entry at 2.0 s.
            # Measured on static_walk_02: 217 such rejections in 146 s.
            self.lidar_of_camera.pop(cam_id, None)
            return

        # THESIS FIX (identity teleporting): reject a match that would
        # move this identity faster than a person can walk. See
        # MAX_IDENTITY_SPEED. The binding is dropped rather than
        # published, so the next cycle re-matches from scratch instead
        # of inheriting a wrong sticky target - and publish_lidar_only()
        # will coast on the last good position in the meantime.
        prev_pub = self.last_identity_pos.get(stable_id)
        if prev_pub is not None:
            gap = max(now - prev_pub[2], 1e-3)
            jump = math.hypot(lx - prev_pub[0], ly - prev_pub[1])
            # THESIS FIX (jitter vs motion): a pure speed threshold
            # cannot separate a walking person from lidar centroid
            # jitter. Clustering changes what it groups as one object
            # between scans (wall corner, door frame, passing legs), so
            # the centroid hops up to ~0.69 m in a single 0.13 s scan -
            # 5.3 m/s implied, with no motion at all. Jitter is a fixed
            # offset; motion scales with elapsed time. Allow both terms
            # separately. Measured on static_walk_05: this passes 106 of
            # 115 previously-rejected matches while still refusing the
            # 3.7-4.2 m identity teleports (8%), which a pure speed gate
            # loose enough to pass the same 92% would have admitted.
            if jump > IDENTITY_JITTER_M + MAX_IDENTITY_SPEED * gap:
                self.get_logger().info(
                    f"match rejected id:{stable_id} lidar:{lid} - implies "
                    f"{jump / gap:.1f} m/s ({jump:.2f} m in {gap:.2f} s)")
                self.lidar_of_camera.pop(cam_id, None)
                return

        fields = [
            str(stable_id), conf, f"{lx:.3f}", f"{ly:.3f}",
            "0.0", "0", "0",
            f"{int(x1)}", f"{int(y1)}", f"{int(x2)}", f"{int(y2)}",
            "camera_confirmed",
        ]

        out = String()
        out.data = ",".join(fields)
        self.pub.publish(out)
        self.last_published[stable_id] = now
        self.last_identity_pos[stable_id] = (lx, ly, now)
        self._note_published(stable_id, lx, ly, now)
        # Anchor for the coasting plausibility check in
        # publish_lidar_only() - this is the last position we actually
        # had camera agreement on.
        self.last_confirmed_pos[stable_id] = (lx, ly, now)

    # -----------------------------------------------------------------
    def _note_published(self, stable_id, x, y, now):
        """Record a published position and update the moving flag."""
        h = self.pub_hist.setdefault(stable_id, [])
        h.append((x, y, now))
        while h and now - h[0][2] > MOVING_WINDOW_S * 3:
            h.pop(0)
        old = [p for p in h if now - p[2] >= MOVING_WINDOW_S]
        if old:
            ox, oy, _ = old[-1]
            if math.hypot(x - ox, y - oy) >= MOVING_MIN_DISP_M:
                self.last_moving[stable_id] = now

    def _is_moving(self, stable_id, now):
        return (now - self.last_moving.get(stable_id, -1e9)) < MOVING_GRACE_S

    def publish_lidar_only(self):
        """Keep a person's stable_id alive while camera-occluded.

        For every lidar->stable binding that is still camera-confirmed
        (i.e. was donated by a real camera detection at some point and
        hasn't been pruned), if no camera track is currently reporting
        that stable_id, publish the lidar position under it anyway.
        Without this, the moment camera loses someone the whole
        downstream chain (KF, predicted cloud, SocialCritic) sees
        nothing for them until camera re-acquires - exactly the gap
        that caused "no person data this cycle" during occlusion.
        """
        now = self.now()

        active_stable_ids = {
            sid for sid, last in self.stable_of_camera.values()
            if (now - last) < CAMERA_ACTIVE_WINDOW
        }

        for lid, (stable_id, _confirmed_at) in list(self.stable_of_lidar.items()):
            if stable_id in active_stable_ids:
                continue  # camera has this one this cycle, camera_callback publishes it
            if lid not in self.lidar_tracks:
                continue  # lidar track already gone; prune() will drop the binding

            last_pub = self.last_published.get(stable_id, 0.0)
            if (now - last_pub) < LIDAR_ONLY_PUBLISH_PERIOD * 0.5:
                continue  # camera_callback just published this stable_id, skip

            lx, ly, lidar_stamp = self.lidar_tracks[lid]

            # THESIS FIX (stale position, coast path) - see
            # LIDAR_POSITION_MAX_AGE. camera_callback already refuses to
            # publish a position older than this; without the same test
            # here the coast path republishes the identical frozen point
            # at 10 Hz. The MAX_COAST_SPEED check below cannot catch it:
            # a frozen point has zero drift, so it passes every time.
            # Measured on hallway_7m_13: every lidar_only message in the
            # run carried a position 1.92-2.82 s old.
            pos_age = now - lidar_stamp
            if pos_age > self.get_parameter("lidar_position_max_age").value:
                self.get_logger().info(
                    f"stale coast id:{stable_id} lidar:{lid} "
                    f"age {pos_age:.2f}s - not publishing")
                continue

            # THESIS FIX (unvalidated coasting): re-check the binding
            # before coasting on it. Nothing else here tests that this
            # cluster is still a plausible position for this identity -
            # the camera gate ran when the binding was made and never
            # again. If the identity is holding a static cluster, the
            # whole downstream chain gets a frozen position at 10 Hz
            # with full confidence and no error anywhere.
            prev = self.last_confirmed_pos.get(stable_id)
            if prev is not None:
                gap = max(now - prev[2], 0.1)
                drift = math.hypot(lx - prev[0], ly - prev[1])
                if drift > MAX_COAST_SPEED * gap:
                    self.get_logger().info(
                        f"coast rejected id:{stable_id} lidar:{lid} "
                        f"drift {drift:.2f} m in {gap:.1f} s "
                        f"(> {MAX_COAST_SPEED} m/s)")
                    continue

            # THESIS FIX (coast drift): the check above anchors on the last
            # CAMERA-CONFIRMED position, so its time budget grows for the
            # whole duration of the coast - 5 s in, MAX_COAST_SPEED permits
            # 7.5 m of drift and constrains nothing. Measured on
            # phantom_test: sid 0 stepped from -40.7 deg / 3.35 m to
            # -50.4 deg / 3.91 m in one 0.1 s cycle (~8 m/s) and was
            # accepted, then held that wall cluster to the end of its life.
            # Anchoring a SECOND check on the previous published position
            # bounds each step at walking speed regardless of how long the
            # coast has run. Both checks apply: the anchored one bounds
            # total displacement, this one bounds per-cycle jumps.
            prev_step = self.last_identity_pos.get(stable_id)
            if prev_step is not None:
                step_gap = max(now - prev_step[2], 0.1)
                step_drift = math.hypot(lx - prev_step[0], ly - prev_step[1])
                # THESIS FIX (real robot, 9 Sep): same jitter model as the
                # camera-path identity gate (IDENTITY_JITTER_M). Speed-only
                # rejected 'step 0.51 m in 0.10 s' - one centroid hop - and
                # then kept rejecting as the gap grew, so a walking person
                # was never coasted.
                if step_drift > IDENTITY_JITTER_M + MAX_COAST_SPEED * step_gap:
                    self.get_logger().info(
                        f"coast step rejected id:{stable_id} lidar:{lid} "
                        f"step {step_drift:.2f} m in {step_gap:.2f} s "
                        f"(> {MAX_COAST_SPEED} m/s)")
                    continue
            # FOV gate on the coast too - see FOV_GATE_DEG. A coast that
            # drifts outside the camera's view can never be re-confirmed,
            # so holding it only feeds the costmap a phantom.
            pose = self.robot_pose()
            if pose is not None:
                prx, pry, pryaw = pose
                if not self.fov_ok(lx, ly, prx, pry, pryaw):
                    if self._is_moving(stable_id, now):
                        self.get_logger().info(
                            f"coast outside FOV allowed id:{stable_id} "
                            f"lidar:{lid} (moving track)",
                            throttle_duration_sec=2.0)
                    else:
                        self.get_logger().info(
                            f"coast rejected id:{stable_id} lidar:{lid} "
                            f"outside camera FOV")
                        continue

            # human_kf_predictor parses both the same way. depth/u/v/
            # bbox fields have no camera detection to report this
            # cycle - left at 0 rather than omitted, so field count
            # (and therefore field indices) never changes with source.
            fields = [
                str(stable_id), "0.00", f"{lx:.3f}", f"{ly:.3f}",
                "0.0", "0", "0", "0", "0", "0", "0", "lidar_only",
            ]

            out = String()
            out.data = ",".join(fields)
            self.pub.publish(out)
            self.last_published[stable_id] = now
            self.last_identity_pos[stable_id] = (lx, ly, now)
            self._note_published(stable_id, lx, ly, now)

    # -----------------------------------------------------------------
    def prune(self):
        now = self.now()

        # THESIS FIX (camera-only false positive gate) - see module
        # docstring. An identity that's never gotten lidar corroboration
        # within CAMERA_ONLY_GRACE_PERIOD is dropped entirely: every
        # camera track currently pointing at it is removed, so it stops
        # publishing and a later re-detection (real or not) starts fresh
        # rather than silently reusing the untrusted identity.
        for sid in [s for s, (t, _b) in self.camera_only_since.items()
                    if now - t > CAMERA_ONLY_GRACE_PERIOD]:
            created_at, bearing_at_creation = self.camera_only_since[sid]
            dead_cam_ids = [c for c, (s2, _) in self.stable_of_camera.items()
                            if s2 == sid]
            for c in dead_cam_ids:
                del self.stable_of_camera[c]
                self.lidar_of_camera.pop(c, None)
            del self.camera_only_since[sid]
            # THESIS FIX (respawn loop, bearing-based): blacklist the
            # VIEWING DIRECTION this drop happened at, not the camera
            # track_id - see module docstring for why id-based
            # blacklisting silently discarded a real, later person.
            self.bearing_blacklist.append((bearing_at_creation, now))
            self.get_logger().info(
                f"Dropped stable {sid} - never got lidar corroboration "
                f"within {CAMERA_ONLY_GRACE_PERIOD:.1f}s (likely a static "
                f"visual false positive, e.g. a door/frame mistaken for "
                f"a person). Blacklisting bearing "
                f"{math.degrees(bearing_at_creation):.1f} deg for "
                f"{CAMERA_TRACK_BLACKLIST_COOLDOWN:.0f}s")

        self.bearing_blacklist = [
            (b, t) for b, t in self.bearing_blacklist
            if now - t < CAMERA_TRACK_BLACKLIST_COOLDOWN
        ]

        for cam_id in [c for c, (_, t) in self.stable_of_camera.items()
                       if now - t > CAMERA_TRACK_TIMEOUT]:
            del self.stable_of_camera[cam_id]
            self.lidar_of_camera.pop(cam_id, None)

        for cam_id in [c for c in self.lidar_of_camera
                       if c not in self.stable_of_camera]:
            del self.lidar_of_camera[cam_id]

        # THESIS FIX (position-based re-ID fallback): capture each
        # about-to-vanish lidar track's last known position BEFORE
        # deleting it below, so the stable_of_lidar pruning further down
        # (which runs after and may drop a binding anchored on a track
        # vanishing this same cycle) can still record an orphan position
        # for it - otherwise the position would already be gone by the
        # time we get there.
        vanishing_lidar_pos = {
            lid: (x, y) for lid, (x, y, t) in self.lidar_tracks.items()
            if now - t > LIDAR_TRACK_TIMEOUT
        }

        for lid in [l for l, v in self.lidar_tracks.items()
                    if now - v[2] > LIDAR_TRACK_TIMEOUT]:
            del self.lidar_tracks[lid]
            self.lidar_first_seen.pop(lid, None)
            self.lidar_span.pop(lid, None)
            self.lidar_hist.pop(lid, None)

        # THESIS FIX (phantom identities): a binding whose camera
        # corroboration has gone stale is dropped, not coasted. See
        # CAMERA_CORROBORATION_TIMEOUT for the measurement and the
        # occlusion trade-off. last_confirmed_pos[sid][2] is the time of
        # the last camera-confirmed publish for that identity.
        stale_corroboration = set()
        for lid, (sid, _t) in self.stable_of_lidar.items():
            prev = self.last_confirmed_pos.get(sid)
            # THESIS FIX (never-corroborated phantom): a binding with no
            # last_confirmed_pos entry was exempt from this check entirely,
            # so it survived on LIDAR_BINDING_TIMEOUT (300 s) for as long as
            # its lidar track stayed alive - and a wall track never dies.
            # Measured on calib_2m: stable id 0 held a wall cluster at
            # 3.70 m / -14.4 deg for the whole 43 s bag, 437/437 messages
            # lidar_only, never once camera-confirmed. The comment here
            # claimed camera_only_since owned this case, but that dict is
            # only populated for freshly ALLOCATED ids (line ~961), not for
            # orphans handed back by find_orphan_to_reattach, so a
            # reattached identity that never reached a confirmed publish
            # fell through both guards permanently.
            # Anchoring on the binding time when there is no confirmation
            # leaves nothing exempt. CAMERA_ONLY_GRACE_PERIOD (4.0 s) still
            # fires first for newly allocated ids.
            anchor = prev[2] if prev is not None else _t
            limit = (MOVING_CORROBORATION_TIMEOUT if self._is_moving(sid, now)
                     else CAMERA_CORROBORATION_TIMEOUT)
            if now - anchor > limit:
                stale_corroboration.add(lid)

        for lid in [l for l, (_, t) in self.stable_of_lidar.items()
                    if now - t > LIDAR_BINDING_TIMEOUT
                    or l not in self.lidar_tracks
                    or l in stale_corroboration]:
            sid = self.stable_of_lidar[lid][0]
            age = now - self.stable_of_lidar[lid][1]
            # last_confirmed_pos may be absent for a never-corroborated
            # binding, which now reaches this branch - fall back to the
            # binding time so the log line cannot KeyError.
            _anch = self.last_confirmed_pos.get(sid)
            _anch_t = _anch[2] if _anch is not None else self.stable_of_lidar[lid][1]
            reason = ("lidar track vanished"
                      if lid not in self.lidar_tracks
                      else "no camera corroboration for "
                           f"{now - _anch_t:.1f}s"
                      if lid in stale_corroboration
                      else f"binding expired after {age:.1f}s")

            # THESIS FIX (position-based re-ID fallback) - see module
            # docstring. Remember this identity's last known position
            # before dropping it, so find_orphan_to_reattach() can hand
            # it back to a nearby fresh detection instead of a new
            # stable id being allocated for the same physical person.
            # Position comes from lidar_tracks if the binding merely
            # aged out (track still alive), or from the snapshot taken
            # above if the track vanished THIS cycle.
            if lid in self.lidar_tracks:
                ox, oy, _ = self.lidar_tracks[lid]
                self.orphaned_identities[sid] = (ox, oy, now)
            elif lid in vanishing_lidar_pos:
                ox, oy = vanishing_lidar_pos[lid]
                self.orphaned_identities[sid] = (ox, oy, now)

            del self.stable_of_lidar[lid]
            # Clear the corroboration anchor with the binding. Without
            # this the entry outlives the identity: if the same stable
            # id is later handed back by find_orphan_to_reattach(), the
            # stale timestamp would trip the check above immediately and
            # kill the reattached identity on its first prune. It also
            # stops the dict growing without bound over a long run.
            self.last_confirmed_pos.pop(sid, None)
            self.last_identity_pos.pop(sid, None)
            self.get_logger().info(
                f"Dropped lidar binding {lid} -> stable {sid} ({reason})")

        for sid in [s for s, (_, _, t) in self.orphaned_identities.items()
                    if now - t > ORPHAN_REATTACH_WINDOW]:
            del self.orphaned_identities[sid]


def main(args=None):
    rclpy.init(args=args)
    node = IdentityFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    # See lidar_person_detector.py's main() for why this is guarded.
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()