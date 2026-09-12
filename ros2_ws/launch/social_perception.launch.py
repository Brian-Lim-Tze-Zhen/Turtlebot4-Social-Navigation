#!/usr/bin/env python3
"""
social_perception.launch.py

THESIS ADDITION - one perception stack for all three scenarios.

=======================================================================
WHY ONE FILE INSTEAD OF THREE
=======================================================================
The head-on, queue and conversation scenarios were each brought up by a
different launch file or by hand. That produced two classes of error
that cost real trials:

  Silent omission. Every node needs `use_sim_time`, or it stamps
  wall-clock while Nav2 runs sim time and the costmap drops every cloud
  with no error logged - the node still prints "Published cloud" and the
  robot simply fails to react. Six or eight hand-typed commands is six
  or eight chances to omit it.

  Silent mis-wiring. The conversation scenario needs
  human_kf_predictor started with
  `-p input_topic:=/person_positions_fused`. Without it the KF reads raw
  ByteTrack ids, identity_fusion_node does nothing at all, and nothing
  anywhere reports a problem. Tying that remap to the same flag that
  starts the fusion node makes the pair impossible to separate.

A deployed robot does not get told which scenario it is entering, so the
default configuration here is the full stack. The per-scenario flags
exist to reproduce the ablations, not because normal operation needs
them.

=======================================================================
WHAT RUNS BY DEFAULT
=======================================================================
  leg_detector_node          lidar leg clusters
  yolo_detector              camera detections
  identity_fusion_node       stable ids from camera+lidar
  human_kf_predictor         reads /person_positions_fused
  group_formation_detector   conversation and queue detection
  social_group_cloud_node    o-space / queue-gap costmap injection
  predicted_person_cloud_node per-person marking, defers group members
  set_pose_bridge            only if a mover needs it
  queue_ground_truth_node    only for static-pedestrian worlds

=======================================================================
ORDERING - THESE DELAYS ARE DEPENDENCIES, NOT GUESSES
=======================================================================
t=0  leg_detector, yolo
     group_formation_detector's queue hold-open anchors on lidar
     clusters. If lidar_points is empty at its first hold attempt, the
     first camera dropout is unprotected.

t=1  identity_fusion_node
     Binds camera to lidar tracks, so both must already be publishing or
     its first association window is empty.

t=2  human_kf_predictor
     Must not start before fusion when the remap is on, or it subscribes
     to a topic with no publisher and stays silent.

t=2  queue_ground_truth_node
     Queries Gazebo over gz transport at startup. Too early returns
     "Service call to [/gazebo/worlds] timed out" and the node then
     publishes nothing for the whole run.

t=3  group_formation_detector
     Loads MobileCLIP-S1. The delay is for lidar/fusion ordering; the
     model load blocks internally.

t=5  social_group_cloud_node, predicted_person_cloud_node
     Started last so their first cycle has groups to consume. The person
     cloud in particular suppresses tracks claimed by /social_groups, so
     starting it before the detector means a window where queue members
     are marked by both layers.

=======================================================================
LAYER RESPONSIBILITY - READ BEFORE SETTING enable_group_layer:=false
=======================================================================
Both cost layers are enabled by default and do NOT overlap:
predicted_person_cloud_node defers any track sitting at a position
claimed by /social_groups, so a queue member is marked by the group
layer only and nobody is marked twice.

That split is not cosmetic. Measured on the 4-person queue, n=3 each:

    no social layer      min_dist 0.528  passes through the queue
    person layer only    navigation fails - spins in place
    group layer only     min_dist 1.129  rounds the head of the queue
    both, naive overlap  min_dist 0.920 +/- 0.356  - unstable, 1 of 3
                         runs passed through
    both, split (default) min_dist 1.116 +/- 0.001, fastest of the four

The naive-overlap standard deviation is the point: marking the same
people twice made the outcome non-reproducible run to run.

=======================================================================
USAGE
=======================================================================
Conversation (default - full stack):
    ros2 launch social_perception.launch.py

Queue (fusion adds nothing for static people; see below):
    ros2 launch social_perception.launch.py \
        enable_fusion:=false enable_ground_truth:=true \
        world_name:=queue_test

Head-on (single pedestrian: no group can form, so the group nodes idle;
disabled anyway to keep the stack minimal, and the mover needs the
set_pose bridge):
    ros2 launch social_perception.launch.py \
        enable_fusion:=false enable_groups:=false \
        enable_set_pose_bridge:=true world_name:=empty_human

Ablation controls:
    enable_group_layer:=false     no o-space / queue-gap injection
    enable_person_layer:=false    no per-person marking

WHY enable_fusion DEFAULTS TRUE BUT IS OFF FOR THE QUEUE
identity_fusion_node re-labels camera detections as they arrive, so it
addresses id churn, not blindness. For a static queue, id churn is
already handled from the other direction by position-anchored holding,
and its independent value - KF velocity continuity, close_since timers -
is near zero when velocity is zero by definition. For conversation,
where members move and sustained-duration timing decides detection, it
matters.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression


WS = "/root/thesis_social_navigation_ws"
SRC = os.path.join(WS, "src", "social_perception", "social_perception")


def node(script, extra_args=None, condition=None, taskset=None):
    """One perception node, always with use_sim_time set.

    use_sim_time is applied here rather than at each call site so it
    cannot be forgotten for an individual node. The failure it prevents
    is silent: the costmap drops every cloud while the node logs
    normally.
    """
    cmd = []
    if taskset:
        cmd += ["taskset", "-c", taskset]
    cmd += ["python3", os.path.join(SRC, script),
            "--ros-args", "-p", "use_sim_time:=true"]
    if extra_args:
        cmd += extra_args
    kwargs = dict(cmd=cmd, output="screen", emulate_tty=True)
    if condition is not None:
        kwargs["condition"] = condition
    return ExecuteProcess(**kwargs)


def generate_launch_description():
    world_name = LaunchConfiguration("world_name")
    enable_fusion = LaunchConfiguration("enable_fusion")
    enable_groups = LaunchConfiguration("enable_groups")
    enable_group_layer = LaunchConfiguration("enable_group_layer")
    enable_person_layer = LaunchConfiguration("enable_person_layer")
    enable_ground_truth = LaunchConfiguration("enable_ground_truth")
    enable_set_pose_bridge = LaunchConfiguration("enable_set_pose_bridge")

    args = [
        DeclareLaunchArgument(
            "world_name", default_value="combined_scenario",
            description="Gazebo world; used by the ground-truth node and "
                        "the set_pose bridge"),
        DeclareLaunchArgument(
            "enable_fusion", default_value="true",
            description="Run identity_fusion_node AND remap the KF to "
                        "/person_positions_fused. These are one switch on "
                        "purpose: the remap without the node leaves the KF "
                        "silent, and the node without the remap does "
                        "nothing. Neither reports an error."),
        DeclareLaunchArgument(
            "enable_groups", default_value="true",
            description="Run group_formation_detector (conversation + queue)"),
        DeclareLaunchArgument(
            "enable_group_layer", default_value="true",
            description="Inject o-space / queue-gap zones. Ablation control."),
        DeclareLaunchArgument(
            "enable_person_layer", default_value="true",
            description="Per-person disks/ellipses. Ablation control."),
        DeclareLaunchArgument(
            "enable_ground_truth", default_value="false",
            description="Publish /person_ground_truth for STATIC "
                        "pedestrians. Movers publish it themselves, so "
                        "leave this off for head-on."),
        DeclareLaunchArgument(
            "enable_set_pose_bridge", default_value="false",
            description="Bridge /world/<world_name>/set_pose. Required by "
                        "the movers, useless without one."),
    ]

    # The KF's input topic follows enable_fusion. Setting one without the
    # other is the mis-wiring this file exists to prevent.
    kf_input = PythonExpression([
        "'/person_positions_fused' if '", enable_fusion,
        "' == 'true' else '/person_positions_map'",
    ])

    # --- t=0 --------------------------------------------------------
    # Lidar before the group detector: its queue hold-open anchors on
    # lidar clusters and an empty cache leaves the first dropout
    # unprotected. yolo is pinned to two cores - it is the CPU
    # bottleneck and starves the Gazebo render loop otherwise.
    leg_detector = node("leg_detector_node.py")
    yolo = node("yolo_detector.py", taskset="0,1")

    # --- t=1 --------------------------------------------------------
    fusion = TimerAction(
        period=1.0,
        actions=[node("identity_fusion_node.py",
                      condition=IfCondition(enable_fusion))])

    # --- t=2 --------------------------------------------------------
    kf = TimerAction(
        period=2.0,
        actions=[node("human_kf_predictor.py",
                      extra_args=["-p", ["input_topic:=", kf_input]])])

    ground_truth = TimerAction(
        period=2.0,
        actions=[node("queue_ground_truth_node.py",
                      extra_args=["-p", ["world_name:=", world_name]],
                      condition=IfCondition(enable_ground_truth))])

    # --- t=3 --------------------------------------------------------
    group_detector = TimerAction(
        period=3.0,
        actions=[node("group_formation_detector.py",
                      condition=IfCondition(enable_groups))])

    # --- t=5 --------------------------------------------------------
    # Person layer last: it defers tracks claimed by /social_groups, so
    # starting it before the detector opens a window where queue members
    # are marked twice.
    group_cloud = TimerAction(
        period=5.0,
        actions=[node("social_group_cloud_node.py",
                      condition=IfCondition(enable_group_layer))])

    person_cloud = TimerAction(
        period=5.0,
        actions=[node("predicted_person_cloud_node.py",
                      condition=IfCondition(enable_person_layer))])

    # --- bridge -----------------------------------------------------
    set_pose_bridge = ExecuteProcess(
        cmd=["ros2", "run", "ros_gz_bridge", "parameter_bridge",
             ["/world/", world_name,
              "/set_pose@ros_gz_interfaces/srv/SetEntityPose"]],
        output="screen",
        emulate_tty=True,
        condition=IfCondition(enable_set_pose_bridge),
    )

    return LaunchDescription(args + [
        leg_detector,
        yolo,
        fusion,
        kf,
        ground_truth,
        group_detector,
        group_cloud,
        person_cloud,
        set_pose_bridge,
    ])
