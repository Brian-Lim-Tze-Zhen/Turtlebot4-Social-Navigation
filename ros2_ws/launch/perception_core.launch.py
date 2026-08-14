#!/usr/bin/env python3
"""
perception_core.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.actions import ExecuteProcess


def generate_launch_description():
    ws_root_arg = DeclareLaunchArgument(
        "ws_root",
        default_value="/root/thesis_social_navigation_ws",
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
    )
    world_arg = DeclareLaunchArgument(
        "world",
        default_value="empty_human",
    )
    enable_predicted_cloud_arg = DeclareLaunchArgument(
        "enable_predicted_cloud",
        default_value="true",
    )

    ws_root = LaunchConfiguration("ws_root")
    use_sim_time = LaunchConfiguration("use_sim_time")
    world = LaunchConfiguration("world")
    enable_predicted_cloud = LaunchConfiguration("enable_predicted_cloud")

    pkg_dir = PathJoinSubstitution(
        [ws_root, "src", "social_perception", "social_perception"])

    yolo_detector = ExecuteProcess(
        cmd=[
            "taskset", "-c", "0,1",
            "python3", PathJoinSubstitution([pkg_dir, "yolo_detector.py"]),
            "--ros-args", "-p", ["use_sim_time:=", use_sim_time],
        ],
        name="yolo_detector",
        output="screen",
        emulate_tty=True,
    )

    human_kf_predictor = ExecuteProcess(
        cmd=[
            "python3", PathJoinSubstitution([pkg_dir, "human_kf_predictor.py"]),
            "--ros-args", "-p", ["use_sim_time:=", use_sim_time],
        ],
        name="human_kf_predictor",
        output="screen",
        emulate_tty=True,
    )

    predicted_person_cloud_node = ExecuteProcess(
        cmd=[
            "python3", PathJoinSubstitution(
                [pkg_dir, "predicted_person_cloud_node.py"]),
            "--ros-args", "-p", ["use_sim_time:=", use_sim_time],
        ],
        name="predicted_person_cloud_node",
        output="screen",
        emulate_tty=True,
        condition=IfCondition(enable_predicted_cloud),
    )

    set_pose_bridge = ExecuteProcess(
        cmd=[
            "ros2", "run", "ros_gz_bridge", "parameter_bridge",
            ["/world/", world, "/set_pose@ros_gz_interfaces/srv/SetEntityPose"],
        ],
        name="set_pose_bridge",
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription([
        ws_root_arg,
        use_sim_time_arg,
        world_arg,
        enable_predicted_cloud_arg,
        LogInfo(msg=[
            "perception_core: starting yolo_detector (taskset 0,1), "
            "human_kf_predictor, set_pose_bridge (world=", world,
            "), predicted_person_cloud_node enabled=", enable_predicted_cloud,
            " — use_sim_time=", use_sim_time,
        ]),
        yolo_detector,
        human_kf_predictor,
        predicted_person_cloud_node,
        set_pose_bridge,
    ])
