# Try the scenarios yourself

Five demo scenarios of the social-navigation stack, each started with one
command. A TurtleBot4 drives through a Gazebo world with simulated people; the
robot sees them with its camera and LiDAR, and Nav2 plans around them.

Each run starts everything from scratch (Gazebo, Nav2, localisation,
perception), sends the robot to its goal, records a bag, shuts everything down
and prints a short result. A run takes 2 to 4 minutes.

| Scenario | What is in the world | What the robot should do |
|---|---|---|
| `headon` | One person walks straight at the robot at 1.2 m/s in a 2.5 m wide corridor | Move to one side early and pass without stopping |
| `headon_blocked` | Two people stand side by side across the same corridor, then walk away after 20 s | Drive up, stop about 1 m short, beep every 3 s until they leave, then carry on |
| `conversation_wide` | Two people stand facing each other, 1.5 m apart, in open space | Go around the pair, not between them |
| `conversation_narrow` | The same pair in a 2.0 m wide corridor, so there is no way around | Pass between them without stopping |
| `queue` | Four people stand in a line, 1.2 m apart, across the robot's way | Go around the end of the line, not through a gap in it |

## What you need

- Linux with Docker and Docker Compose
- An X11 display, for the Gazebo and RViz windows
- A GPU with Intel/Mesa OpenGL drivers (for other graphics, see the
  environment variables in `docker/docker-compose.yaml`)

## Set up (once)

Build and start the container:

    git clone git@github.com:Brian-Lim-Tze-Zhen/Turtlebot4-Social-Navigation.git
    cd Turtlebot4-Social-Navigation/docker
    docker compose build
    docker compose up -d

Build the workspace inside it:

    docker exec -it thesis_social_nav bash
    cd /root/thesis_social_navigation_ws
    source /opt/ros/jazzy/setup.bash
    colcon build --symlink-install
    exit

The main `README.md` has more detail on the image and the YOLO weights.

## Run a scenario

Open a shell in the container. If it has been stopped (for example after a
reboot), start it first:

    docker start thesis_social_nav
    docker exec -it thesis_social_nav bash
    source /opt/ros/jazzy/setup.bash

Then launch one scenario:

    ros2 launch /root/thesis_social_navigation_ws/launch/scenario.launch.py scenario:=headon
    ros2 launch /root/thesis_social_navigation_ws/launch/scenario.launch.py scenario:=headon_blocked
    ros2 launch /root/thesis_social_navigation_ws/launch/scenario.launch.py scenario:=conversation_wide
    ros2 launch /root/thesis_social_navigation_ws/launch/scenario.launch.py scenario:=conversation_narrow
    ros2 launch /root/thesis_social_navigation_ws/launch/scenario.launch.py scenario:=queue

Gazebo and RViz open by themselves. You do not need to set an initial pose or
send a goal; the run does both. Wait for the summary at the end before
starting the next one.

The same thing without `ros2 launch`:

    cd /root/thesis_social_navigation_ws
    ./run_scenario.sh headon

### Options

| Launch argument | Default | Effect |
|---|---|---|
| `rviz:=false` | `true` | Do not open RViz |
| `gazebo_gui:=false` | `true` | Do not open the Gazebo window |
| `bag_name:=<name>` | `demo_<scenario>_<date>_<time>` | Name of the recorded bag. A name can be used only once. |

With the script, the same switches are environment variables:
`SHOW_RVIZ=false HEADLESS=true ./run_scenario.sh headon my_bag_name`.
For `headon_blocked`, `STAND_S=<seconds>` sets how long the two people stand
before they walk away (default 20).

## What to look for

**In RViz:** the map, the robot, the planned path (it is replanned five times
a second), and the cost around people. In the conversation scenarios and in
`headon_blocked` a graded zone appears around the standing pair: a small
lethal core at each person, a wider personal-space ring, and the shared space
between them.

**In the summary printed at the end.** These are example values from single
runs on the development machine; yours will differ by a few centimetres.

| Scenario | Goal reached | Closest approach, centre to centre | Stopped time |
|---|---|---|---|
| `headon` | yes | 0.78 m | 0 s |
| `headon_blocked` | yes | 1.12 m | about 12 s, with a few beeps (estimate; measured 21 s and 8 beeps when the pair stood for 30 s) |
| `conversation_wide` | yes | 0.83 to 0.98 m | 0 s |
| `conversation_narrow` | yes | 0.65 m | 0 s |
| `queue` | yes | 1.03 to 1.13 m, around the far end of the line | 0 s |

Robot and person touch at 0.44 m centre to centre (0.189 m robot radius plus
0.25 m body radius), so every value above that is a pass without contact.

**The beep.** The simulator has no speaker. In `headon_blocked` the beeps are
listed in the summary with their times. To watch them live, open a second
shell in the container while the run is going:

    tail -f /root/thesis_social_navigation_ws/logs/<bag_name>/blocked_beep.log

## Where the results go

| What | Where (inside the container) |
|---|---|
| Recorded bag, plus the exact config and code used for the run | `/root/thesis_social_navigation_ws/bags/<bag_name>/` |
| Logs of every process | `/root/thesis_social_navigation_ws/logs/<bag_name>/` |

On the host these are `ros2_ws/bags/` and `ros2_ws/logs/` in the repository.

Full analysis of a run:

    cd /root/thesis_social_navigation_ws
    python3 analysis/analyse_headon_F.py bags/<bag_name>     # head-on scenarios
    python3 analysis/analyse_F_narrow.py bags/<bag_name>     # conversation scenarios
    python3 analysis/queue_run_summary.py bags/<bag_name> 6.0 -3.0   # queue scenario

## If something goes wrong

The simulation does not always start cleanly. If a run fails before the robot
gets its goal, the scenario is started again by itself, up to three times, and
says so (`Attempt 1 of 3 did not produce a result`). The failed attempt's logs
are kept as `<bag_name>_invalid1`. The table below is for when that is not
enough.

| What you see | What to do |
|---|---|
| `ERROR: /map_server not active after 120 s` (or the same for `/amcl`, `/controller_server`) | Nav2 sometimes hangs at start-up (about 1 launch in 4). The run stops by itself after two minutes and is retried automatically. |
| `ERROR: a Gazebo instance is already running` | A previous run is still up. Wait for it to finish, or run `pkill -9 -f "gz sim"` in the container. |
| `ERROR: ... already exists` | That bag name has been used. Pick another, or leave `bag_name` out. |
| No windows appear | Allow the container to use your display: run `xhost +local:root` on the host, then try again. |
| `ERROR: no wide group`, `no narrow group` or `no queue group` | The people were not recognised as a group in time. Run it again. |
| `ERROR: /costmap_filter_info_server not active after 60 s` | The node that passes the social zone to the planner did not start. Run it again. |
| `docker exec` says the container is not running | `docker start thesis_social_nav` |

## What each scenario runs

| Scenario | World | Nav2 config | Trial script |
|---|---|---|---|
| `headon` | `corridor_headon` | `config/social_nav2_headon_F_hwreq_block_blockedhold_sim.yaml` | `run_headon_F_trial.sh` |
| `headon_blocked` | `corridor_two_human` | same | `run_headon_F_trial.sh` |
| `conversation_wide` | `conversation_test` | `config/social_nav2_ablation_F_socialzone_sim.yaml` | `run_conv_F_trial.sh` |
| `conversation_narrow` | `conversation_test_narrow` | same | `run_conv_F_trial.sh` |
| `queue` | `queue_test` | same, with queue detection switched on in the group detector | `run_conv_F_trial.sh` |

`run_scenario.sh` only chooses between these; the two trial scripts take more
options (walker speed and position, a second walker, other configs), listed at
the top of each file.
