#!/usr/bin/env python3
"""
queue_perception.launch.py

THESIS ADDITION - brings up the full queue-scenario perception chain.

-----------------------------------------------------------------------
WHY A LAUNCH FILE, AND WHAT IT PRESERVES
-----------------------------------------------------------------------
Six nodes started by hand across six terminals is six chances to omit
`use_sim_time`, six chances to start them in the wrong order, and - most
importantly for the trial protocol - six processes to remember to restart
between trials. Any node left running across a trial boundary carries its
instance state with it, which is exactly the state-pollution failure that
already invalidated one condition group and prompted
start_lidar_fusion_clean.sh.

With this file, the per-trial reset is Ctrl-C plus one command.

TWO CONSTRAINTS ARE ENCODED HERE, NOT LEFT TO THE OPERATOR:

1. use_sim_time on EVERY node. Without it a node stamps wall-clock while
   Nav2 runs on sim time, and the costmap silently drops every cloud with
   "Message Filter dropping message... earlier than all the data in the
   transform cache". The node still logs "Published cloud" and nothing
   errors - the robot simply fails to react.

2. leg_detector_node starts BEFORE group_formation_detector. The
   detector's queue hold-open uses lidar clusters as position anchors; if
   its lidar_points cache is empty at the first hold attempt, the first
   camera dropout is unprotected.

WHAT IS DELIBERATELY NOT HERE:

- identity_fusion_node. It re-labels camera detections on camera message
  arrival, so it addresses id churn, not blindness - and id churn is
  already handled by position-anchored holding. Its independent value
  (KF velocity continuity, close_since timers) is near zero for a static
  queue. Deferred to the conversation scenario.
- human_kf_predictor's `input_topic` remap. It must NOT be set while
  identity_fusion_node is absent, or the KF subscribes to a topic nobody
  publishes and goes silent with no error.

-----------------------------------------------------------------------
WHY ExecuteProcess RATHER THAN Node
-----------------------------------------------------------------------
These are invoked as `python3 <path>`, matching the verified manual
procedure exactly, rather than through `ros2 run` entry points. The
package is an editable install and the console_scripts entry points are
not the path that has been tested; switching invocation style at the same
time as switching to a launch file would change two things at once.

-----------------------------------------------------------------------
LOG VISIBILITY - READ THIS BEFORE FIRST USE
-----------------------------------------------------------------------
All six nodes now write to ONE terminal, interleaved. The per-node
signals that trial verification depends on - "Queue HELD on 4/4",
"points=111", "4 merged" - are still there but no longer separated.

Each line is prefixed with its node name, so filter as needed:

    ros2 launch ... queue_perception.launch.py 2>&1 | grep -E "HELD|points=|merged"

If a specific node needs close watching, comment it out here and run that
one by hand in its own terminal.

-----------------------------------------------------------------------
USAGE
-----------------------------------------------------------------------
    ros2 launch queue_perception.launch.py

or, if the file is not on the launch path:

    ros2 launch /root/thesis_social_navigation_ws/launch/queue_perception.launch.py

Arguments:
    world_name:=queue_test          world queried for ground-truth poses
    enable_ground_truth:=false      omit the ground-truth publisher
    enable_group_layer:=false       omit social_group_cloud_node - this is
                                    the ablation control for the queue
                                    costmap layer (see note below)

ABLATION NOTE: enable_group_layer:=false disables ONLY the group/o-space
injection. predicted_person_cloud_node is launched separately (via
perception_core.launch.py) and still marks each person individually, so
this argument isolates the group layer, not all social costing.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration


WS = "/root/thesis_social_navigation_ws"
SRC = os.path.join(WS, "src", "social_perception", "social_perception")


def node_process(script, extra_args=None, condition=None):
    """One perception node, always with use_sim_time set.

    use_sim_time is applied here rather than per call site so it cannot be
    forgotten for an individual node - the failure it prevents is silent.
    """
    cmd = ["python3", os.path.join(SRC, script),
           "--ros-args", "-p", "use_sim_time:=true"]
    if extra_args:
        cmd += extra_args
    kwargs = dict(cmd=cmd, output="screen", emulate_tty=True)
    if condition is not None:
        kwargs["condition"] = condition
    return ExecuteProcess(**kwargs)


def generate_launch_description():
    world_name = LaunchConfiguration("world_name")
    enable_ground_truth = LaunchConfiguration("enable_ground_truth")
    enable_group_layer = LaunchConfiguration("enable_group_layer")

    args = [
        DeclareLaunchArgument(
            "world_name", default_value="queue_test",
            description="Gazebo world queried by the ground-truth publisher"),
        DeclareLaunchArgument(
            "enable_ground_truth", default_value="true",
            description="Publish /person_ground_truth for offline evaluation"),
        DeclareLaunchArgument(
            "enable_group_layer", default_value="true",
            description="Inject queue/conversation zones into the costmap"),
    ]

    # --- t=0: lidar first -------------------------------------------
    # group_formation_detector's hold-open anchors on lidar clusters, so
    # this must be publishing before that node's first detect cycle.
    leg_detector = node_process("leg_detector_node.py")

    # --- t=0: camera detection and the KF ---------------------------
    # No input_topic remap on the KF: identity_fusion_node is not running.
    yolo = node_process("yolo_detector.py")
    kf = node_process("human_kf_predictor.py")

    # --- t=2: ground truth ------------------------------------------
    # Queries Gazebo over gz transport at startup. Delayed so the
    # simulator has finished spawning models; querying too early returns
    # "Service call to [/gazebo/worlds] timed out" and the node then
    # publishes nothing at all.
    ground_truth = TimerAction(
        period=2.0,
        actions=[node_process(
            "queue_ground_truth_node.py",
            extra_args=["-p", ["world_name:=", world_name]],
            condition=IfCondition(enable_ground_truth))])

    # --- t=3: group detection ---------------------------------------
    # Loads MobileCLIP-S1, which takes several seconds; the delay is for
    # lidar ordering, not for the model load, which blocks internally.
    group_detector = TimerAction(
        period=3.0,
        actions=[node_process("group_formation_detector.py")])

    # --- t=5: costmap injection -------------------------------------
    # Started last so its first cycle has groups to consume rather than
    # logging an empty cloud while the detector is still loading.
    group_cloud = TimerAction(
        period=5.0,
        actions=[node_process(
            "social_group_cloud_node.py",
            condition=IfCondition(enable_group_layer))])

    return LaunchDescription(args + [
        leg_detector,
        yolo,
        kf,
        ground_truth,
        group_detector,
        group_cloud,
    ])
