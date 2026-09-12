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
      launch/              Launch files (simulation scenarios, perception stack)
      analysis/            Offline evaluation scripts + QoS override YAML
      maps/                Static map for AMCL localization
      simulation_models/   Custom Gazebo worlds + person_standing model
                           (temp_models/ is excluded — see Dockerfile to regenerate)
      behavior_trees/      Custom Nav2 BT XMLs
      run_sim.sh           Full simulation launch script
      record_trial.sh      Bag recording with provenance snapshots
