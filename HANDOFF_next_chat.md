# Handoff: from head-on avoidance to the conversation scenario

Written 4 Oct 2026. Paste the "Prompt" section into a new chat. The rest is
reference for whoever picks the work up.

## Prompt

    We are continuing my thesis work in ~/thesis_social_navigation (ROS 2 Jazzy
    sim, container thesis_social_nav, workspace mounted at
    /root/thesis_social_navigation_ws). Read HANDOFF_next_chat.md in the repo
    root and the README first.

    The container may be stopped (it was on 4 Oct). Check with
    `docker ps -a --filter name=thesis_social_nav` and, if it is not running,
    start it with `docker start thesis_social_nav` before anything else.

    The head-on avoidance work is finished for now and is on branch
    headon-avoidance. Next is the conversation scenario: two men standing
    apart, with the distance between them between the thresholds. Before
    changing anything, tell me which thresholds you think I mean and what
    separations you would test, and wait for my answer.

    Rules:
    - The original code must stay available for the thesis. It is tagged
      (thesis-conv-F-original, thesis-headon-avoidance-2026-10-04). Do not
      delete or strip code, scripts or configs that produced thesis results;
      switch things off by parameter or config instead, and ask before
      deleting anything.
    - Work on a new branch made from headon-avoidance. Do not commit to main.
    - /home/brian/Desktop/Turtlebot4 (the real robot's repo) is read-only.
    - New SocialCritic parameters must default to off.
    - Check the container for a running trial before starting one or
      rebuilding. I often run trials myself.
    - Report numbers with the number of runs. Commit only when I ask.

## Where everything is

| Thing | Location |
|---|---|
| Repo | `~/thesis_social_navigation`, GitHub `Brian-Lim-Tze-Zhen/Turtlebot4-Social-Navigation` |
| Current branch | `headon-avoidance` (all head-on work; `main` is untouched) |
| Sim container | `thesis_social_nav`; workspace `/root/thesis_social_navigation_ws` = `ros2_ws/`. Found stopped (Exited 137) on 4 Oct: `docker start thesis_social_nav`, then `docker exec -it thesis_social_nav bash`. |
| Real robot repo (read-only) | `/home/brian/Desktop/Turtlebot4/humble_client/workspace` (Humble) |
| Bags and logs | `ros2_ws/bags`, `ros2_ws/logs` (git-ignored, on this machine only) |
| Result reports | `ros2_ws/analysis/*_report.txt` (in git, including `headon_ablation_AE_report.txt` and `headon_ablation_AE_results.csv`, the A – E head-on ablation recomputed on 4 Oct from the August bags) |
| Thesis draft | `~/Draft - Thesis/thesis.md` |
| Previous chat | `~/Documents/claude-chat-backups/` (raw transcript copy) |

## Backups of the original code

| Tag | Commit | What it preserves |
|---|---|---|
| `thesis-conv-F-original` | `9d7b406` (= `main`) | The conversation code (ablation F wide and narrow) that produced the n = 5 results of 24 Sep in the README. |
| `thesis-headon-avoidance-2026-10-04` | `729248a` | The head-on avoidance work as it stands. |

To look at or run the original: `git checkout thesis-conv-F-original`, then
rebuild the critic (below). Each bag folder also holds the exact config and
code used for that run (`config_used.yaml`, `*_used.py`, `social_critic_used.cpp`).

**The conversation stack on `headon-avoidance` is not identical to the tagged
original.** These shared files changed during the head-on work:

- `config/social_nav2_ablation_F_socialzone_sim.yaml`: MPPI tuning synced from
  the robot (`time_steps` 200 → 120, `wz_std` 1.5 → 0.7, `wz_max` 1.9 → 1.5,
  PathFollow `cost_weight` 4 → 6, global `inflation_radius` 0.25 → 0.35).
  These five are the whole diff of that file against the tag.
- Perception nodes synced to the robot's 2 Oct code (motion release,
  stationary hold, KF `q_pos`/`q_vel`, cloud `GROUP_MEMBER_RADIUS` 0.50 → 0.70,
  `side_by_side` members deferred to the zone node).
- `social_critic`: many new parameters, all default off, so the F config
  behaves as before unless they are set.

So a conversation trial on this branch may not reproduce the 24 Sep numbers
exactly. Re-run the wide and narrow baselines on the branch before comparing
anything new against them.

## How the head-on scenario is run

Inside the container, from `/root/thesis_social_navigation_ws`. One command
per trial: fresh launch, AMCL initial pose, undock, perception, provenance
snapshot, bag, goal, person, teardown.

Avoidance (current best config):

    CFG=$PWD/config/social_nav2_headon_F_hwreq_block_sim.yaml \
    SHOW_RVIZ=true HEADLESS=false ./run_headon_F_trial.sh <new_bag_name>

Baseline (hardware config, person cloud ellipse, overlaps the walker):

    ./run_headon_F_trial.sh <new_bag_name>

| Option | Effect |
|---|---|
| `WORLD=empty_human` | open space instead of the 2.5 m corridor |
| `PERSON_Y=0.4` / `-0.4` | walker to the robot's left / right |
| `WORLD=two_human PERSON2_Y=<m> [PERSON2_X0 PERSON2_X1]` | second walker |
| `SHOW_RVIZ=false HEADLESS=true` | no windows |
| `./run_headon_F_batch.sh <prefix> <n> ENV=val…` | n trials with one retry, then mean ± SD |

Analyse (source ROS and the workspace first):

    python3 analysis/analyse_headon_F.py bags/<name>

Rebuild the critic, never while a trial runs:

    colcon build --symlink-install --packages-select social_critic

Things that go wrong:

- About 1 launch in 4 hangs at Nav2 startup. Ctrl+C and rerun; the batch
  script retries by itself.
- A bag name can be used once.
- Edit `run_headon_F_trial.sh` atomically (write a new file, then replace):
  bash reads a running script as it goes.
- The trial script installs headless-capable launch files into the container
  and keeps the originals as `.orig`.

## Head-on results in one table

Hardware perception limits (person first seen at about 7 m), walker 1.2 m/s,
minimum centre-to-centre distance.

| Case | Result | Runs |
|---|---|---|
| Baseline | 0.058 m (overlap) | 5 |
| One centred walker, corridor | 0.834 ± 0.036 m, no stop/spin/reverse | 5 (before the last three rules) |
| Same, current config | 0.853, 0.857 (right); 0.748 – 0.759 (left) | 2 + 3 |
| One centred walker, open space | 0.76 – 0.90 m | about 10 |
| Walker 0.4 m to either side | 1.09 – 1.26 m | 8 |
| Two walkers, single file | 0.88 / 0.93 m | 1 |
| Second walker hidden behind the first | 0.39 – 0.72 m (accepted limit) | 4 |
| Two walkers side by side | 0.12 m (fails) | 1 |

Open head-on items: an n = 5 run of the current config; why left passes are
about 0.1 m tighter; grouping for side-by-side walkers; a fallback when no
side is free; the port to the real robot.

## The conversation scenario as it stands

Described in the README under "Ablation F — conversation group, wide" and
"… narrow corridor". Summary:

- Two people stand facing each other at (3.0, ±0.75), so 1.5 m apart.
- `social_group_detector_node_lidarhold_sim.py` publishes `/social_groups`.
  `social_zone_costmap_node_sim.py` turns a group into a KeepoutFilter mask
  on the **global** costmap. The SocialCritic (`group_aware: true`) relaxes
  its distance for members of a narrow group.
- Worlds: `conversation_test` (open, map `maps/map_name.yaml`, spawn
  (−1, 0, π)) and `conversation_test_narrow` (2.0 m corridor, map
  `maps/conversation_test_narrow.yaml`, spawn (−1, 0, 0)). Goal (6, 0).
- Config: `config/social_nav2_ablation_F_socialzone_sim.yaml`.
- Trials were run by hand, one terminal per step (see the README); there is
  no one-command trial script for it yet. `run_headon_F_trial.sh` already
  starts the same zone node, filter info server, perception launch and group
  detector, so it is the natural base for one.
- Analysis: `analysis/analyse_F_narrow.py <bag>` (used for both wide and
  narrow), `analysis/aggregate_F_trials.py` for mean ± SD.

Thresholds in the current code (the ones "between the thresholds" could refer to):

| Threshold | Value | Where | Meaning |
|---|---|---|---|
| `CONV_MAX_DIST` | 1.8 m | group detector | Farther apart than this, two people are not a conversation pair. |
| `CONV_MIN_DIST` | 0.3 m | group detector | Closer than this is treated as one person. |
| `CONV_MAX_SPEED` | 0.30 m/s | group detector | Both must be near-stationary. |
| `CONV_MIN_DURATION` | 0.75 s | group detector | Closeness must last this long. |
| Wide/narrow buffer (`ZONE_BUFFER`) | 0.4 m | group detector `_effective_buffer` | Free flank gap along the pair axis; at or above it the pair is "wide" (route around), below it "narrow" (pass between). The detector publishes the resulting buffer as field 10 of `/social_groups` (0.400 = wide). |
| `narrow_buffer_threshold` | 0.39 | SocialCritic (default; not set in the F config) | The critic's own test on that field: a group is narrow if field 10 is **below 0.39**. It is 0.01 under the detector's 0.4 so that a published 0.400 reads as wide. The two values must be changed together. |
| Halo overlap | 1.6 m | zone node (2 × 0.8 m halo) | Below this separation the two personal-space halos meet in the gap. |
| Penalty-free pass | 1.20 m | SocialCritic (2 × `narrow_social_distance` 0.60) | Smallest separation the robot can pass between without critic cost. |
| Physical contact | 0.88 m | 2 × (0.25 + 0.189) | Bodies and robot touch. |
| `social_distance` | 0.94 m | F config | Critic distance to a person not in a narrow group. |

From the head-on discussion, a related rule was agreed for two **walking**
people and not built: 1.6 m or more apart, pass between; closer, treat as one
group and take whichever of "between" and "outside" leaves more room; slow
down if that is under 0.8 m.

## Decisions already made by the user

- No stopping or reversing while avoiding.
- `wz_std` stays at or below about 0.9; above that the real robot wobbles.
- LiDAR-only detection of moving people is ruled out: walls read as moving
  on the real robot.
- The hidden-walker case is accepted as a limitation.
- Keep-side block width 0.4 m.
- Screen recordings in `~/Videos/Screencasts` are renamed
  `sim_<bag> <timestamp>.mp4` or `hw_<bag> <timestamp>.mp4` by matching bag times.
