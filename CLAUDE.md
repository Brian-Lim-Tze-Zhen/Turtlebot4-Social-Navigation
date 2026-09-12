# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ROS 2 (Jazzy) thesis workspace for **social robot navigation**: a TurtleBot4 navigates in Gazebo while perceiving, tracking, and predicting pedestrian motion to feed a predictive obstacle layer into Nav2.

The single ROS package is `src/social_perception` (ament_python build type).

---

## Build & Source

```bash
# From workspace root
colcon build --packages-select social_perception
source install/setup.bash
```

Always re-source after rebuilding. The workspace install is at `/root/thesis_social_navigation_ws/install/setup.bash`.

## Run the Full Stack

```bash
./run_sim.sh
```

This script sources the environment, syncs Gazebo world/model files, launches `turtlebot4_gz_bringup` (Gazebo + Nav2 + RViz), waits for `/map_server` to be active and `/map` to have data, then starts all five `social_perception` nodes in sequence.

**All perception nodes require `use_sim_time:=true`** to stay in sync with Gazebo's clock:
```bash
ros2 run social_perception <node_name> --ros-args -p use_sim_time:=true
```

## Run Individual Nodes

```bash
ros2 run social_perception move_person_gazebo2 --ros-args -p use_sim_time:=true
ros2 run social_perception yolo_detector --ros-args -p use_sim_time:=true
ros2 run social_perception human_kf_predictor --ros-args -p use_sim_time:=true
ros2 run social_perception prediction_marker_node --ros-args -p use_sim_time:=true
ros2 run social_perception predicted_person_cloud_node --ros-args -p use_sim_time:=true
```

## Tests & Linting

```bash
colcon test --packages-select social_perception
colcon test-result --verbose
```

Tests cover copyright, flake8, and pep257 only (in `src/social_perception/test/`).

---

## Architecture & Data Flow

```
Gazebo (TurtleBot4 + simulated people)
  │
  ├─ /oakd/rgb/preview/image_raw  ─────┐
  ├─ /oakd/rgb/preview/depth       ────┤
  └─ /oakd/rgb/preview/camera_info ────┘
                                        │
                               yolo_detector.py
                         (YOLOv8s + ByteTrack, class=person,
                          conf≥0.65, every 2nd frame, depth
                          from lower-middle body region,
                          TF: oakd_rgb_camera_optical_frame→map)
                                        │
                               /person_positions_map
                               "id,conf,map_x,map_y,depth,u,v"
                                        │
                          human_kf_predictor.py
                         (4-state KF: [x,y,vx,vy], split Q,
                          EMA velocity smoothing, horizon=3.0s)
                                        │
                         /predicted_person_positions
                         "id,conf,x,y,vx,vy,pred_x,pred_y,horizon"
                                   ┌────┴────┐
                 predicted_person_cloud_node  prediction_marker_node
              (PointCloud2 @ 10 Hz, per-track  (RViz MarkerArray:
               disk r=0.20 @ current pos;       blue sphere=current,
               ellipse a=0.80/b=0.40 along      red sphere=predicted,
               heading @ predicted pos;          green arrow, risk
               stale tracks pruned @ 0.3s)       cylinder)
                          │                           │
              /predicted_person_cloud       /predicted_person_markers
                          │
              Nav2 NonPersistentVoxelLayer
              (local costmap, config/social_nav2.yaml)
```

`move_person_gazebo2.py` drives the simulated people via `gz service /world/two_human/set_pose` (threaded subprocess) and publishes `/person_ground_truth` (PoseArray) for offline evaluation. It is **not** in the perception loop.

## Message Formats

All inter-node perception messages use `std_msgs/String` with CSV encoding (no custom message types):

| Topic | Format |
|---|---|
| `/person_positions_map` | `id,conf,map_x,map_y,depth,px_u,px_v` |
| `/predicted_person_positions` | `id,conf,x,y,vx,vy,pred_x,pred_y,horizon` |

## Nav2 Configuration

Key configs in `config/`:

- **`social_nav2.yaml`** — full pipeline; local costmap uses `NonPersistentVoxelLayer` consuming `/predicted_person_cloud`
- **`social_nav2_no_predicted_cloud.yaml`** — ablation baseline; identical except `nonpersistent_voxel_layer` is removed; lidar-only obstacle sensing
- **`config/ablation/`** — per-condition configs for ablation study (A_nolayer through E_critweight20_socialcritic_on)

The local costmap controller is **MPPI** (`nav2_mppi_controller`).

To switch between configs, change `params_file:=` in `run_sim.sh`.

## Environment Notes

- **ROS distro**: Jazzy (`$ROS_DISTRO`)
- **RMW**: FastDDS (default). CycloneDDS is intentionally avoided — it hits a DDS participant-index limit when Gazebo + Nav2 + RViz + 5 nodes all start simultaneously on domain 0.
- **ROS_DOMAIN_ID**: 0, `ROS_LOCALHOST_ONLY=0`
- **YOLO model**: `yolov8s.pt` at workspace root (also copied inside the package). `yolov8n.pt` is also present but unused by the active detector node.
- **Gazebo worlds**: `worlds/` (workspace root) and `simulation_models/worlds/` both contain SDF world files. `run_sim.sh` calls `/usr/local/bin/setup_gazebo_worlds.sh` to sync them into `turtlebot4_gz_bringup` before launch.
- **Custom person model**: `simulation_models/person_standing/` (SDF + mesh). `simulation_models/temp_models/` is a stock Gazebo model library with a `COLCON_IGNORE` — colcon will not build it.

## Key Tuning Parameters

| Node | Parameter | Default | Effect |
|---|---|---|---|
| `yolo_detector` | `process_every_n_frames` | 2 | Skip frames to reduce load |
| `yolo_detector` | `max_jump` | 0.8 m | Rejects map-frame position jumps per ByteTrack ID |
| `human_kf_predictor` | `prediction_horizon` | 3.0 s | How far ahead to extrapolate |
| `human_kf_predictor` | `smooth_alpha` | 0.3 | EMA weight for velocity used in extrapolation |
| `predicted_person_cloud_node` | `track_timeout` | 0.3 s | How long before a silent track is pruned from the cloud |
| `move_person_gazebo2` | `update_dt` | 1.0 s | `gz service` call interval per person (lower values stall Gazebo physics) |

---

## Known Issues & Fixes

### TF tree split — map not rendering in RViz, controller timeout

**Symptom**: After startup (especially after a Gazebo crash/restart), RViz shows no map and the controller log repeats:
```
Timed out waiting for transform from base_link to odom to become available,
tf error: Could not find a connection between 'odom' and 'base_link' because
they are not part of the same tree. Tf has two or more unconnected trees.
```

**Root cause**: The TF tree has two disconnected branches:
- `odom → base_link` (published by the diff-drive controller — always present)
- `map` (isolated — nothing linking it to `odom`)

The `map → odom` transform is only published by AMCL once it has a valid pose estimate. With no initial pose set, AMCL has no particle filter seed and publishes nothing, leaving the tree split.

**Fix**: In RViz, select the **"2D Pose Estimate"** tool in the toolbar, then click-and-drag on the map at the robot's actual starting location and heading. This seeds AMCL's particle filter, after which it begins publishing `map → odom`, the tree becomes `map → odom → base_link → ...`, and the errors stop.

**This is a manual one-time action required every time the simulation is started from scratch.** If this becomes a workflow bottleneck, an initial pose can be set programmatically at launch using `nav2_util`'s `set_initial_pose` action or by publishing to `/initialpose` (`geometry_msgs/PoseWithCovarianceStamped`) from a launch file node.
