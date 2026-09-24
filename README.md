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

## Ablation F — conversation group, narrow corridor

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

## Ablation F — conversation group, wide (open space)

This is the same condition F stack as the narrow case, with the same config,
nodes and SocialCritic build. Only the world changes, so the zone detector
selects the **wide** shape.

### Wide test world

`conversation_test.sdf` places the pair at (3.0, ±0.75) facing each other in
open space. The spawn is (−1, 0, 0) and the goal (6, 0), as in the narrow
case. The free flank gap beyond each member, measured along the pair axis on
the static `/map`, is 4.25 m and 4.15 m. The detector therefore reports
buffer 0.400 (wide) and the zone paints o-space 90 with no flank lobes, so
routing around the pair is cheaper than passing between them.

The narrow-case changes (1–3 above) are inactive here, by design. The probe
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

Same protocol as the narrow case, with a fresh launch each trial, headless:

    ros2 launch turtlebot4_gz_bringup turtlebot4_gz.launch.py world:=conversation_test slam:=false nav2:=true localization:=true rviz:=true map:=/root/thesis_social_navigation_ws/maps/map_name.yaml params_file:=/root/thesis_social_navigation_ws/config/social_nav2_ablation_F_socialzone_sim.yaml headless:=true x:=-1.0 y:=0.0 yaw:=0.0

Differences from the narrow protocol:

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
- The RTF is 0.52, below real time, and similar to the narrow set (0.55).
  Perception latency therefore counts for fewer sim-seconds than it would
  on hardware.

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
