# Thesis: Social Navigation for TurtleBot4 (ROS2 Jazzy)

Predictive, social-aware human-obstacle representation for Nav2, using
YOLO + ByteTrack detection, a Kalman-filter motion predictor, and a
direction-oriented elliptical risk zone published into the local costmap.

## Requirements

- Docker + Docker Compose
- An X11 display (Linux host) for Gazebo/RViz GUI
- A GPU with Intel/Mesa OpenGL drivers (see `docker-compose.yaml` env vars;
  adjust `MESA_LOADER_DRIVER_OVERRIDE` if you're not on Intel graphics)

## Setup

Clone the repo and build the image:

    git clone git@github.com:Brian-Lim-Tze-Zhen/Turtlebot4-Social-Navigation.git
    cd Turtlebot4-Social-Navigation/docker
    docker compose build
    docker compose up -d

The image build automatically:
- Installs ROS2 Jazzy, Nav2, Gazebo Harmonic (`ros-jazzy-ros-gz*`), and
  TurtleBot4 packages
- Sets up a Python venv with Ultralytics (YOLO) and dependencies
- Pre-downloads required Gazebo Fuel models

### One-time: download YOLO weights

Pretrained weights (`yolov8n.pt`, `yolov8s.pt`) are not stored in this repo.
Ultralytics downloads them automatically on first use, or fetch manually:

    docker exec -it thesis_social_nav bash
    source /root/venv/bin/activate
    python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); YOLO('yolov8s.pt')"

### One-time: build the workspace

    docker exec -it thesis_social_nav bash
    cd /root/thesis_social_navigation_ws
    colcon build
    source install/setup.bash

## Running the simulation

    docker exec -it thesis_social_nav bash
    ./run_sim.sh

This launches:
1. Gazebo + TurtleBot4 + Nav2 + AMCL localization + RViz
   (`turtlebot4_gz_bringup`)
2. The social perception pipeline: simulated person motion, YOLO+ByteTrack
   detection, Kalman-filter predictor, RViz prediction marker, and the
   predicted-person point-cloud publisher feeding the Nav2 local costmap

After launch, in RViz: use **2D Pose Estimate** to set the robot's initial
pose on the map before sending navigation goals (required since this runs
in AMCL localization mode, not SLAM).

## Ablation F — conversation group, wide (open space)

Condition F gives a detected conversation pair a graded social zone,
published by `social_zone_costmap_node_sim.py` as a KeepoutFilter mask on
the **global** costmap. The zone has two shapes, selected by the effective
buffer (field 10 of `/social_groups`) that the group detector measures:

- **wide** (buffer ≥ 0.4 m): o-space cost 90, so the planner routes around
  the pair.
- **narrow** (buffer < 0.4 m): o-space cost 35, flank lobes 60, so passing
  between the members is the cheapest route when there is no room outside.

Both shapes keep a 0.25 m lethal body core (100) and a 0.8 m
personal-space halo (80) around each member.

The wide and narrow cases use the same condition F stack: the same config,
nodes and SocialCritic build. Only the world differs, which decides the shape.

### Wide test world

`conversation_test.sdf` places the pair at (3.0, ±0.75) facing each other in
open space. The spawn is (−1, 0, π), facing **away** from the pair, and the
goal is (6, 0). The free flank gap beyond each member, measured along the pair axis on
the static `/map`, is 4.25 m and 4.15 m. The detector therefore reports
buffer 0.400 (wide) and the zone paints o-space 90 with no flank lobes, so
routing around the pair is cheaper than passing between them.

The narrow-case changes (1–3, in the narrow section below) are inactive
here, by design. The probe
finds room on both flanks. The halo-overwrite in the gap exists only in the
narrow branch. `group_aware` only relaxes the distance for members of
*narrow* groups, so SocialCritic keeps `social_distance` 0.94 m.

### What had to change for the wide case

**Body core and halo in the wide branch** (`social_zone_costmap_node_sim.py`,
`_regions()`). The 0.25 m lethal core and 0.8 m halo used to be yielded only
after the narrow-only `return`. `predicted_person_cloud_node_lidar.py` defers
conversation members (`not in ("queue", "conversation")`), so in the wide
case nothing painted the bodies. The global obstacle layer only marks people
from `/scan` within `obstacle_max_range` 2.5 m, so at planning time (about
4 m away) the pair existed in the global costmap only as the o-space.

A costmap probe at spawn showed the fix. Before: members 89, 0.6 m outside
each member 0. After: members 100, 0.6 m outside 79, zone centre 89
(unchanged). An earlier pre-fix pass had skimmed a member at 0.622 m
centre-to-centre (n = 1).

### Running a wide trial

A fresh launch each trial, headless:

    ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py world:=conversation_test slam:=false nav2:=true localization:=true rviz:=true map:=/root/thesis_social_navigation_ws/maps/map_name.yaml params_file:=/root/thesis_social_navigation_ws/config/social_nav2_ablation_F_socialzone_sim.yaml headless:=true x:=-1.0 y:=0.0 yaw:=3.14

The steps are as listed under *Running a narrow trial* below, with these
differences:

- Step 1: publish `/initialpose` at (−1, 0, 3.14), matching the spawn yaw.
- Step 3: wait until `/social_groups` field 10 reads **0.400** (wide) and
  the detector logs `Pair (a,b) confirmed: facing`.
- The bags are `bags/conv_F_wide_trial1` … `trial5`, recorded with the
  same topic list as the narrow trials.
- Provenance uses the world `conversation_test.sdf` and the `nav2_F_wide*`
  log. Copy it right after each trial, because the command takes the newest log.

Analyse all five trials with the same script as the narrow case, so both
cases use identical metric definitions:

    for b in /root/thesis_social_navigation_ws/bags/conv_F_wide_trial*; do python3 /root/thesis_social_navigation_ws/analysis/analyse_F_narrow.py $b; done 2>&1 | tee /root/thesis_social_navigation_ws/analysis/conv_F_wide_report.txt

### Results (simulation, 24 Sep 2026)

Wide case, same F config (`group_aware=on`, inactive for wide groups), no
speed limiter, n = 5 protocol trials:

| Metric | Mean ± SD | Range |
|---|---|---|
| Goal reached | 5 / 5 | — |
| Time to goal | 27.5 ± 0.4 s | 26.8 – 28.0 |
| Min centre distance (GT) | 0.851 ± 0.014 m | 0.837 – 0.869 |
| Min surface clearance (GT) | +0.412 ± 0.014 m | +0.398 – +0.430 |
| Stopped / spin-in-place time | 0.0 / 0.0 s | all runs |
| vx in gap window (mean / min) | 0.28 ± 0.01 / 0.20 ± 0.04 m/s | min 0.15 |
| Passing side / max \|y\| | around +y in 5 / 5 / 1.67 ± 0.06 m | 1.62 – 1.77 |
| Plans through the gap | 0 in all runs | — |
| AMCL error in gap window (max / lateral) | 0.056 ± 0.021 / 0.036 ± 0.013 m | lateral ≤ 0.050 |
| Member position error (p1 / p2) | 0.169 ± 0.012 / 0.163 ± 0.019 m | ≤ 0.189 |
| Wide classification | 100 % of `/social_groups` msgs | buffer 0.400 in all msgs |
| Real-time factor (headless) | 0.52 ± 0.04 | 0.45 – 0.57 |

In open space the planner routes around the pair in every run, on the same
side, with a 1.4 cm spread in clearance and no hesitation.

**Caveats**

- The minimum centre distance (0.851 m) is below `social_distance` 0.94 m.
  SocialCritic does not hold its full target distance here; the halo and
  the global plan decide the passing distance.
- The last zone mask in trials 1–2 reads 0, against 90 in trials 3–5. The
  analysis reads the **last** mask in the bag, and the zone node publishes
  an empty mask on shutdown, so it was most likely stopped before the bag.
  The trajectories match trials 3–5, but the bags do not prove the zone was
  active during the pass. A per-pass mask check is still to do.
- The member position error (~0.17 m) is larger than in the narrow case
  (~0.10 m). It is a systematic offset toward the robot, since detection
  measures the robot-facing body surface rather than the centre.
- Trial 2 has an AMCL error of 0.618 m at t = 21.2 s, during initial
  convergence before the goal was sent (goal at t = 27). In the gap window
  it stayed at 0.029 m.
- The spawn faces away from the pair (yaw π), unlike the narrow case (yaw 0).
  The robot turns in place before driving. The analysis run window starts at
  the first forward motion (|vx| > 0.05 m/s), so the turn is excluded from
  the time and stop metrics.
- The RTF is 0.52, below real time, and similar to the narrow set (0.55).
  Perception latency therefore counts for fewer sim-seconds than it would
  on hardware.

## Ablation F — conversation group, narrow corridor

The zone shapes and shared stack are described in the wide section above.

### Narrow test world

`conversation_test_narrow.sdf`: a 2.0 m corridor (inner walls at
y = ±1.00 m, x −2.0 … 8.0), static grey walls, and the pair at
(3.0, ±0.75) facing each other. The spawn is (−1, 0, 0) and the goal (6, 0).
That leaves 0.25 m between each person's centre and the wall, so their
bodies block the flanks completely. The static map
(`maps/conversation_test_narrow.{pgm,yaml}`) is generated from the SDF
wall geometry by `simulation_models/worlds/make_narrow_map.py`, not SLAM'd.
The walls have an explicit material because unlit (black) walls produced
phantom YOLO-pose detections that broke the facing classifier.

### What had to change for the narrow case

1. **Narrow/wide probe** (`social_group_detector_node_lidarhold_sim.py`,
   `_effective_buffer`): the old probe measured perpendicular to the pair
   axis, i.e. along the corridor, and always read wide. It now probes
   **along the pair axis** from the midpoint, on the static `/map`
   (`costmap_topic:=/map`). Free gap per side = probe − separation/2 −
   0.25 m body radius. The pair is wide if **either** side has ≥ 0.4 m.
   Subtracting the body radius gives the decision a ~0.2 m margin against
   detection error (member positions were off by up to 0.17 m).
2. **Narrow gap vs halo** (`social_zone_costmap_node_sim.py`): regions are
   combined with a per-cell maximum, so at separations below
   2 × 0.8 m = 1.6 m the two halos (80) buried the narrow o-space (35).
   In the narrow branch only, the gap between the two bodies is now painted
   with *overwrite* after the halos and before the cores.
   The wide branch is unchanged.
3. **Group-aware SocialCritic** (`social_critic` C++ plugin): the zone
   acts only on the global planner, while SocialCritic kept every person
   at `social_distance` 0.94 m. In a ~1.4 m gap MPPI therefore rejected
   the planner's path through the pair and stalled at the gap entrance.
   The new parameter `group_aware` (default **off**) makes the critic
   subscribe to `/social_groups`. People within 0.5 m of a member of a
   narrow group get `narrow_social_distance` (0.60 m) instead.
   It is enabled **only** in `config/social_nav2_ablation_F_socialzone_sim.yaml`,
   so conditions A–E behave exactly as before. It requires a rebuild:
   `colcon build --symlink-install --packages-select social_critic`.

Minimum pair separation for a penalty-free pass with these settings:
2 × `narrow_social_distance` = 1.20 m centre-to-centre. Physical contact
occurs at 2 × (0.25 + 0.189) = 0.88 m.

### Running a narrow trial

Every trial uses a fresh launch, headless:

    ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py world:=conversation_test_narrow slam:=false nav2:=true localization:=true rviz:=false map:=/root/thesis_social_navigation_ws/maps/conversation_test_narrow.yaml params_file:=/root/thesis_social_navigation_ws/config/social_nav2_ablation_F_socialzone_sim.yaml headless:=true x:=-1.0 y:=0.0 yaw:=0.0

Then, one terminal each:

1. Publish `/initialpose` at (−1, 0, 0).
2. Start the zone node, `costmap_filter_info_sim.launch.py`,
   `perception_camray_bringup_sim.launch.py`, and the group detector with
   `-p costmap_topic:=/map`.
3. Wait until `/social_groups` field 10 is below 0.39 (narrow).
4. Start `ros2 bag record`, including `/sim_ground_truth_pose`, and copy
   the provenance files into the bag folder: config, zone node, detector,
   `social_critic.cpp`, the world, and the SocialCritic startup lines.
5. Send the goal (6, 0) with `ros2 action send_goal`, not via RViz.
6. Stop the bag at `Goal succeeded`.

Check the Nav2 log for
`SocialCritic: group_aware=on (narrow_social_distance=0.60 m ...)`.

Analyse a trial with:

    python3 /root/thesis_social_navigation_ws/analysis/analyse_F_narrow.py /root/thesis_social_navigation_ws/bags/<bag_name>

Clearance is measured against Gazebo ground truth
(`/sim_ground_truth_pose`, world frame = map frame), with true person
poses taken from the bag's `world_used.sdf`. `/odom` is used for velocities
only, because its pose drifted by several metres in this setup.

### Results (simulation, 24 Sep 2026)

Narrow case, `group_aware=on`, no speed limiter, n = 5 protocol trials:

| Metric | Mean ± SD | Range |
|---|---|---|
| Goal reached | 5 / 5 | — |
| Time to goal | 23.4 ± 0.5 s | 23.1 – 24.2 |
| Min centre distance (GT) | 0.717 ± 0.033 m | 0.660 – 0.741 |
| Min surface clearance (GT) | +0.278 ± 0.033 m | +0.221 – +0.302 |
| Stopped / spin-in-place time | 0.0 / 0.0 s | all runs |
| vx in gap (mean / min) | 0.30 ± 0.01 / 0.22 ± 0.03 m/s | min 0.17 |
| AMCL error in gap (max / lateral) | 0.156 ± 0.015 / 0.029 ± 0.010 m | lateral ≤ 0.041 |
| Member position error (p1 / p2) | 0.094 ± 0.024 / 0.101 ± 0.028 m | ≤ 0.127 |
| Narrow classification | 100 % of `/social_groups` msgs | buffer max 0.346 |
| Real-time factor (headless) | 0.55 ± 0.02 | 0.53 – 0.57 |

All runs stayed inside the designed window: above `narrow_social_distance`
(0.60 m) and contact (0.44 m), below `social_distance` (0.94 m).

**Without group awareness** (`group_aware=off`, pilot, n = 1), the robot
stalled at the gap entrance about 0.94 m from both members.
It logged `Failed to make progress` twice and then looped in recovery spins.
With SocialCritic disabled at runtime (diagnostic only), it passed, which
confirms the critic as the sole cause of the stall.

**Caveats**

- Trial 4 has incomplete provenance (no `world_used.sdf`); its person poses
  fell back to the known (3.0, ±0.75).
- The stall comparison is n = 1.
- AMCL shows a consistent along-track lag in the corridor (worst error
  −0.223 ± 0.068 m, always negative x). The lateral error in the gap stays
  ≤ 0.041 m, so the passage is unaffected.
- The narrow/wide margin is thin: the maximum buffer was 0.346 against a
  threshold of 0.39.
- SocialCritic reads `/person_positions_fused`, which carries zero
  velocity, so it treats people as stationary. That is fine for this static
  pair, but relevant for moving-person conditions.
- `social_zone_speed_limiter.py`, the "passes through slowly" behaviour,
  was not part of these runs.

## Ablation F — wide vs narrow (n = 5 each)

| Metric | Wide (open space) | Narrow (2.0 m corridor) |
|---|---|---|
| Route | around the pair (+y) | through the gap |
| Goal reached | 5 / 5 | 5 / 5 |
| Time to goal | 27.5 ± 0.4 s | 23.4 ± 0.5 s |
| Min centre distance (GT) | 0.851 ± 0.014 m | 0.717 ± 0.033 m |
| Min surface clearance (GT) | +0.412 ± 0.014 m | +0.278 ± 0.033 m |
| Stopped / spin time | 0 / 0 s | 0 / 0 s |
| Classification | 100 % wide (buffer 0.400) | 100 % narrow (buffer ≤ 0.346) |
| RTF | 0.52 ± 0.04 | 0.55 ± 0.02 |

The zone does what it was designed to do. With room outside, the robot
avoids the o-space at +0.41 m surface clearance. Without room, it passes
through the conversation at a reduced but non-contact +0.28 m. It never
stops in either case.

## Ablation F — head-on, corridor (hardware pipeline)

A single person walks straight at the robot in a corridor, using the same
perception pipeline and tuning as the real TurtleBot4 (camera-ray
perception + the F keepout config, as run on hardware on 2 Oct 2026).

### Head-on test world

`corridor_headon.sdf`: a 2.5 m corridor (inner walls at y = ±1.25 m,
x −4.45 … 10.45). The robot spawns **on the dock** at (−3, 0) with yaw
180°, undocks (backs off and turns to face +x), and drives to the goal
(8, 0). `person_1` walks one-way from (9.5, 0) to (−2.0, 0) at 1.2 m/s on
the corridor centreline, so it is a true collision course. The person and
the goal are started together, about 12 m apart.

The static map (`maps/corridor_headon_aligned.{pgm,yaml}`) is generated
from the SDF wall geometry by
`simulation_models/worlds/make_corridor_headon_map.py`. The older SLAM'd
`maps/corridor_headon.yaml` is shifted about 3 m in x from the Gazebo
frame, so ground truth does not line up with it.

### What was synced from hardware

The sim ports were brought up to the hardware repo's 2 Oct state. Only the
sim adaptations differ (no `/turtlebot4` namespace, raw `Image`, Gazebo
camera intrinsics and 640 px image width).

1. `camera_ray_person_node_sim.py`: motion release for a stuck range lock.
2. `camera_ray_identity_node_sim.py`: stationary hold (`lidar_hold`).
3. `human_kf_predictor_lidar.py`: `q_pos` 0.05 → 0.02, `q_vel` 0.05 → 0.15,
   `lidar_hold` handling.
4. `predicted_person_cloud_node_lidar.py`: `GROUP_MEMBER_RADIUS` 0.50 → 0.70,
   `side_by_side` members deferred to the zone node.
5. `config/social_nav2_ablation_F_socialzone_sim.yaml`: `time_steps`
   200 → 120, `wz_std` 1.5 → 0.7, PathFollowCritic weight 4 → 6, global
   `inflation_radius` 0.25 → 0.35.

Not synced: the SocialCritic group logic, the group detector and the zone
node, which a lone walker does not exercise. Item 5 also applies to any new
conversation trials; the wide and narrow results above were recorded before it.

### Running a head-on trial

One command per trial, inside the container. It does a fresh headless
launch, sets `/initialpose`, undocks, starts perception and the zone stack,
snapshots provenance, records the bag, starts the person and sends the
goal together, and tears everything down at the goal result:

    ./run_headon_F_trial.sh headon_F_corridor_trial1

`RVIZ=true HEADLESS=false` shows the run. `PERSON_SPEED`, `PERSON_Y`,
`PERSON_X0`, `PERSON_X1`, `GOAL_X` and `GOAL_Y` override the protocol.

Analyse one or more trials (per-trial report plus mean ± SD):

    python3 /root/thesis_social_navigation_ws/analysis/analyse_headon_F.py "/root/thesis_social_navigation_ws/bags/headon_F_corridor_trial*" | tee /root/thesis_social_navigation_ws/analysis/headon_F_corridor_report.txt

Metric definitions match `analyse_F_narrow.py`, except that the person's
position comes from `/person_ground_truth` because it moves.

### Results (simulation, 2–3 Oct 2026)

Centreline head-on at 1.2 m/s, n = 5 protocol trials:

| Metric | Mean ± SD | Range |
|---|---|---|
| Goal reached | 5 / 5 | — |
| Time to goal | 42.9 ± 1.9 s | 40.6 – 44.8 |
| Min centre distance (GT) | 0.058 ± 0.068 m | 0.008 – 0.177 |
| Min surface clearance (GT) | −0.381 ± 0.068 m | −0.431 – −0.262 |
| Max lateral deviation (whole run) | 0.27 ± 0.21 m | 0.09 – 0.61 |
| Stopped / spin / reverse time | 2.3 ± 2.7 / 0.6 ± 0.1 / 3.2 ± 1.9 s | stopped 0.0 – 5.3 |
| wz sign flips | 13.6 ± 1.1 | 12 – 15 |
| First detection range (true gap) | 7.50 ± 0.36 m | 7.04 – 7.87 |
| Detection → KF speed 0.8 m/s | 0.83 ± 0.08 s | 0.74 – 0.94 |
| Max AMCL error | 0.299 ± 0.130 m | 0.137 – 0.489 |
| Real-time factor (headless) | 0.59 ± 0.01 | 0.58 – 0.60 |

**The robot does not avoid the person in this protocol.** The negative
surface clearance in all five trials means the person's body overlapped the
robot. The goal is reached only because the person then walks on. The
fix is in "Head-on avoidance" below.

The pipeline itself works (pilot bag): the person is detected at about
7.5 m, the KF reaches walking speed within a second, the cloud is published
and the global plan bends 0.6 – 0.8 m sideways. But the plan bends only
next to the person's current position and flips side between replans, and
the closing speed is 1.5 m/s, so about 5 s pass between detection and
encounter. The robot drives straight until the person is under 3 m away.
The hardware runs of 2 Oct show the same pattern (path bends earlier, robot
does not turn earlier, passes of 0.3 – 0.4 m from robot centre).

**Caveats**

- The sim person is moved with `set_pose`, walks dead centre and never
  yields. A real pedestrian side-steps, so these numbers are a worst case
  and not directly comparable with hardware pass distances.
- Because the person is teleported through the robot, part of the
  "reverse" time may be the robot being pushed; this was not separated out.
- Passing side at a 0.01 – 0.18 m minimum distance is not meaningful and is
  not reported.
- SocialCritic reads `/person_positions_fused`, which carries zero
  velocity, so it treats the walker as stationary, as on hardware.
- The person's SDF collision is two legs and a torso (radius 0.20 m);
  `PERSON_R` 0.25 m is kept for comparability with the conversation cases.
- Undock failed once and a parameter dump hung once during the series;
  both trials were rerun from a fresh launch after adding a retry and a
  timeout to the script.

## Head-on avoidance (corridor, hardware perception limits)

The baseline above overlaps the walker in every trial. This section is the
follow-up: the same corridor and walker, with the SocialCritic extended so
the robot passes at 0.8 m or more (robot centre to person centre) without
stopping, spinning or reversing. Perception runs at the real robot's limits
(YOLO at 320 px, publish confidence 0.45, 8 m range cap), so the person is
first seen at about 7 m.

A surface clearance of 0.8 m is not possible here: the corridor is 2.5 m
wide, and the robot centre cannot get further than about 0.9 m from a
centreline walker before it jams against the wall.

### Why the baseline fails

1. The critic read `/person_positions_fused`, which has no velocity, so it
   treated a 1.2 m/s walker as standing still.
2. Left and right cost the same in a head-on, so the side was picked by
   noise and changed between control cycles.
3. Reversing was allowed (`vx_min` −0.31), and backing away was cheaper
   than committing to a side.
4. A failed replan triggered the behaviour tree's Spin recovery mid-encounter.

### What was added to SocialCritic

Every new parameter is off by default, so the other conditions are unchanged.

| Rule | Parameters | What it does |
|---|---|---|
| Velocity | `topic: /predicted_person_positions`, `max_prediction_time: 6.0` | Reads the KF output and moves the person forward in time along each rollout. |
| Frozen lane | `pass_side_weight`, `pass_side_margin` 0.80, `pass_side_max_offset` 0.90 | Draws a line from the robot's position at first sight towards the walker and keeps it fixed for that track. Rollouts must stay 0.80 – 0.90 m to one side of it. |
| Lane on first sight | `lane_on_first_sight_s` 1.5 | Uses a provisional lane until the KF confirms the approach, which takes about 1 s. |
| Walker held on lane | `lane_lateral_trust` 0 | Ignores the 0.2 – 0.4 m sideways swing of the estimate while the robot turns. |
| No retreat | `no_retreat_weight` 300, `vx_min` 0.0 | Penalises rollout steps that lose ground, only while the walker is still ahead. |
| Side selection | `pass_side_auto`, `side_decision_s` 0.6, `side_switch_offset` 0.10, `side_commit_offset` 0.06 | Keeps right by default; goes left if the walker reads more than 0.10 m to the right. The side locks once the robot is 6 cm off the lane. |
| Lane carry-over | `lane_carry_radius` 0.6, `lane_carry_along` 2.0 | A re-acquired track with a new id inherits the old lane and side. |
| Keep-side block | `publish_lane_block`, `lane_block_width` 0.4 | Publishes `/social_critic/lane_block`, marked in the **global** costmap only, so NavFn plans on the same side as the controller. |
| Nearest first | `lane_nearest_only` | With several walkers, only the nearest one still ahead keeps the lane rule, no-retreat and block. |
| Slow-down at a pass | `occlusion_slow_weight` 300, `occlusion_slow_speed` 0.15 | Holds about 0.15 m/s while a walker goes by and 1 s after, because someone may be hidden behind them. |
| Track-jump handling | `lane_jump_reset` 1.5 | A track id that jumps more than 1.5 m is a different person. They get a lane parallel to the previous one, through their own position; the side is the one the robot is already on. |
| Markers | `publish_markers` | Lane and target strip on `/social_critic/lane_markers` for RViz. |

Other changes in the config: `wz_std` 0.7 (above about 0.9 the real robot
wobbles), `movement_time_allowance` 10, the person cloud reduced to a body
disk of radius 0.30 m (the forward lane trapped the robot mid-crossing), and
`behavior_trees/navigate_to_pose_keep_last_plan.xml`, which has no Spin or
BackUp and keeps the last plan when a replan fails.

### Running it

    CFG=$PWD/config/social_nav2_headon_F_hwreq_block_sim.yaml \
    YOLO_IMGSZ=320 YOLO_MIN_CONF=0.45 MAX_PERSON_RANGE=8.0 \
    RAY_COAST=true RAY_COAST_S=1.5 COAST_TIMEOUT=1.5 \
    LANE_B=0.15 DISK_R=0.30 LANE_SLOPE=0.0 LANE_MAX=0.3 \
    SHOW_RVIZ=true HEADLESS=false ./run_headon_F_trial.sh <bag_name>

A second walker: `WORLD=two_human PERSON2_Y=<m>` (with `PERSON2_X0`,
`PERSON2_X1`); both walkers start at the same sim time.
`PERSON_Y` moves the walker sideways, `WORLD=empty_human` runs in open
space, `TRACK_DROPOUT_AT=<m>` loses the track at that range and returns it
under a new id. `run_headon_F_batch.sh <prefix> <n> ENV=val…` runs n trials
and aggregates. `SHOW_RVIZ=false HEADLESS=true` runs without windows.
`social_nav2_headon_F_hwreq_sim.yaml` is the same config without the block.

### Results (simulation, 3 Oct 2026)

The n = 5 table below was measured with a 1.5 m block and before the
nearest-first, slow-down and track-jump rules were added. The current
config has not had its own n = 5 run; its single-walker runs are listed
under "Current config" further down.

Centreline walker at 1.2 m/s, `hwreq_block` config, n = 5
(`analysis/headon_hwreq_z5_corr_report.txt`):

| Metric | Baseline | Avoidance (mean ± SD) | Range |
|---|---|---|---|
| Goal reached | 5 / 5 | 5 / 5 | — |
| Min centre distance (GT) | 0.058 m | 0.834 ± 0.036 m | 0.804 – 0.891 |
| Min surface clearance (GT) | −0.381 m | 0.395 ± 0.036 m | 0.365 – 0.452 |
| Pass side | — | right 5 / 5 | — |
| Stopped / spin / reverse time | 2.3 / 0.6 / 3.2 s | 0.0 / 0.0 / 0.0 s | — |
| Time to goal | 42.9 s | 37.1 ± 0.2 s | 36.8 – 37.4 |
| wz sign flips | 13.6 | 10.0 ± 2.0 | 7 – 12 |
| First detection range (true gap) | 7.50 m | 7.19 ± 0.37 m | 6.81 – 7.80 |

Other cases, same config unless noted (single runs or small n):

| Case | n | Pass side | Min centre distance |
|---|---|---|---|
| Walker 0.4 m to the robot's right | 3 | left | 1.127 – 1.190 m |
| Walker 0.7 m to the right | 1 | left | 1.392 m |
| Walker at the right wall (0.95 m) | 1 | left | 1.349 m |
| Walker 0.4 m to the left | 2 | right | 1.211, 1.263 m |
| Track lost at 5.5 m for 1 s, new id | 2 | right | 0.863, 0.885 m |
| Open space, with block (earlier side settings) | 5 | right | 0.773 – 0.902 m |
| Without the block (`hwreq`) | 5 | right | 0.854 ± 0.037 m |
| Long range (YOLO 640 px, seen at 11 m, `avoid_gentle`) | 5 | right | 0.879 ± 0.021 m |

Block width, centred walker (the block only needs to make the wrong side
the longer way round; all sizes steered NavFn equally, 0 – 6 of about 30
plans per run on the wrong side):

| Block width | Corridor | Open space |
|---|---|---|
| 1.5 m | 0.834 ± 0.036 m (n = 5) | 0.811 m (n = 1) |
| 0.8 m | 0.824, 0.922, 0.894 m | 0.794, 0.799, 0.899 m |
| 0.4 m (current) | 0.843, 0.783 m | 0.766, 0.757 m |

Current config (0.4 m block, nearest first, slow-down, track-jump
handling), one centred walker in the corridor:

| Slow speed | Runs | Pass side | Min centre distance | Stopped time |
|---|---|---|---|---|
| 0.10 m/s | 2 | right | 0.853, 0.857 m | 0.0, 0.3 s |
| 0.15 m/s (current) | 3 | left | 0.748, 0.749, 0.759 m | 0.0 s |

The three left passes were caused by the first reading of the walker's
sideways position (−0.25, −0.33, −0.14 m for a centred walker), not by the
slow speed. The slow-down adds about 1.5 s to the time to goal.

### Two walkers (open space, 3 – 4 Oct 2026)

Each cell is the minimum centre distance to walker A / walker B.

| Case | Runs | Result | Verdict |
|---|---|---|---|
| Single file, B 1.5 m behind A on the same line | 1 | 0.876 / 0.925 m | works |
| Staggered, B 3 m behind A and 0.5 m to the robot's left | 1 | 0.840 / 1.416 m | works |
| Staggered, B 3 m behind A and 0.5 m to the robot's right (B hidden behind A) | 4 | A 0.787 – 0.865 m, B 0.393 – 0.721 m | under target |
| Side by side, 0.7 m apart | 1 | 0.818 / 0.120 m | fails |

Hidden walker: B is only about 4° off A's direction and the camera cannot
separate them, so B is first tracked at about 3 m, after A has gone by. The
tracker reuses A's id for B. Development of that case:

| Version | Distance to B | Notes |
|---|---|---|
| B kept on A's lane (same id) | 0.345 m | 0.7 s reversing |
| Lane reset + slow-down | 0.430, 0.570 m | |
| Lane from the robot towards B, side from the path direction | 0.616, 0.248 m | wrong side once, 2.3 s spinning |
| Parallel lane through B (current) | 0.721, 0.592, 0.393, 0.519 m | no stop, spin or reverse |

About 2 s remain once B is seen; the robot cannot move the missing 0.3 –
0.4 m sideways in that time. This is accepted as a limit.

**Limits**

- The margin over 0.8 m is thin: the lowest corridor run was 0.804 m and
  one open-space run was 0.773 m.
- The walker's sideways position is only known to about ±0.23 m at 7 m, so
  a centred walker is sometimes passed on the left.
- A walker hugging the wall was not detected in one earlier run (YOLO
  confidence 0.16 – 0.41, below the 0.45 threshold).
- One oncoming person, or several in single file. Two people side by side
  get contradictory lane rules and the robot ends up between them
  (0.12 m). Grouping them into one lane is designed but not built: 1.6 m
  or more apart, pass between; closer, take whichever of "between" and
  "outside" leaves more room, and slow down if that is under 0.8 m.
- A person hidden behind another is passed at about 0.4 – 0.7 m. The
  LiDAR does see the hidden person's legs, but LiDAR-only detection was
  ruled out because walls read as moving on the real robot.
- The pass side for a centred walker follows a first reading that is good
  to about ±0.23 m, so it can be left or right. Left passes were about
  0.1 m tighter (0.75 m); the cause is not known.
- The sim walker goes straight at 1.2 m/s and never yields. The hardware
  bags of 25 Sep show 1.36 – 1.41 m/s and a lost track at the turn-around
  in 7 of 11 runs.
- The numbers `social_distance` 0.86 and the 0.80 – 0.90 strip are fitted
  to this 2.5 m corridor.
- Not yet ported to the real robot (Humble). The hardware SocialCritic has
  diverged from this one and needs the lane code merged by hand.
- About 1 launch in 4 fails at Nav2 startup; the trial script retries.

## Custom Gazebo worlds/models - known issue and workaround

`turtlebot4_gz_bringup`'s launch file does not reliably resolve
`model://` URIs via `GZ_SIM_RESOURCE_PATH` for nested custom models when
Gazebo is spawned through `ros2 launch` (confirmed empirically: works for
a standalone `gz sim` call, fails under `ros2 launch` with the identical
environment). `docker/setup_gazebo_worlds.sh` works around this by
copying custom worlds/models into `turtlebot4_gz_bringup`'s own installed
`worlds/` directory at container startup, and is also called explicitly
at the top of `run_sim.sh` for non-interactive invocations.

## Attribution

The `person_standing` Gazebo model (`ros2_ws/simulation_models/`) was
created by **Marina Kollmitz** (University of Freiburg) using MakeHuman.
Not an original contribution of this thesis - included with attribution
intact per `model.config`.

## Repository structure

    docker/              Dockerfile, docker-compose.yaml, Gazebo world-sync script
                         docker/notes/ — debugging notes (TF issues, QoS, etc.)
    ros2_ws/             (mounted as /root/thesis_social_navigation_ws in container)
      src/
        social_perception/   ROS2 package: YOLO detection, KF prediction,
                              cloud publisher, person mover, group detection
        social_critic/       C++ Nav2 critic plugin
      config/
        social_nav2.yaml     Base Nav2 config (full social pipeline)
        ablation/            Per-ablation-condition configs (A–E)
        social_nav2_ablation_F_socialzone_sim.yaml
                             Condition F (social zone + group-aware SocialCritic)
      launch/              Launch files (simulation scenarios, perception stack)
      analysis/            Offline evaluation scripts + QoS override YAML
                           (analyse_F_narrow.py: condition F runs, narrow and wide)
      maps/                Static maps for AMCL localization
                           (conversation_test_narrow.* generated from the SDF)
      simulation_models/   Custom Gazebo worlds + person_standing model
                           (worlds/make_narrow_map.py regenerates the narrow map)
                           (temp_models/ is excluded — see Dockerfile to regenerate)
      behavior_trees/      Custom Nav2 BT XMLs
      run_sim.sh           Full simulation launch script
      record_trial.sh      Bag recording with provenance snapshots
