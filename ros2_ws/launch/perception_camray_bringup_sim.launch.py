from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration

WS = "/root/thesis_social_navigation_ws"

# SIMULATION version of perception_camray_bringup.launch.py.
# Differences from hardware:
#   - use_sim_time defaults to true
#   - no /turtlebot4 namespace: /scan, /odom, /tf, /tf_static, /map used as-is
#   - *_sim.py for yolo (raw Image), camera_ray_person (sim intrinsics),
#     camera_ray_identity (/scan)
#   - camera_fx / yaw offset = Gazebo camera_info (443.53, 0.0), not hw calibration
# Hardware launch file is untouched.


def generate_launch_description():
    simtime = LaunchConfiguration("use_sim_time")
    coast = LaunchConfiguration("coast_timeout")
    ray_coast = LaunchConfiguration("ray_coast")

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("coast_timeout", default_value="1.5"),
        DeclareLaunchArgument("ray_coast", default_value="false"),
        ExecuteProcess(
            cmd=["python3", f"{WS}/src/social_perception/social_perception/camera_lidar/yolo_leg_detector_lidar_sim.py",
                 "--ros-args", "-p", ["use_sim_time:=", simtime]],
            name="yolo_leg_detector_lidar", output="screen"),
        ExecuteProcess(
            cmd=["python3", f"{WS}/src/social_perception/social_perception/camera_lidar/camera_ray_person_node_sim.py",
                 "--ros-args", "-p", ["use_sim_time:=", simtime],
                 "-p", "scan_topic:=/scan",
                 "-p", "odom_topic:=/odom",
                 "-p", "camera_yaw_offset_deg:=0.0",
                 "-p", "camera_fx:=443.53",
                 "-p", "camera_cx:=320.0",
                 "-p", "ray_half_width_deg:=6.0"],
            name="camera_ray_person_node", output="screen"),
        ExecuteProcess(
            cmd=["python3", f"{WS}/src/social_perception/social_perception/camera_lidar/camera_ray_identity_node_sim.py",
                 "--ros-args", "-p", ["use_sim_time:=", simtime],
                 "-p", ["coast_enable:=", ray_coast],
                 "-p", "scan_topic:=/scan"],
            name="camera_ray_identity_node", output="screen"),
        ExecuteProcess(
            cmd=["python3", f"{WS}/src/social_perception/social_perception/camera_lidar/person_marker_publisher.py",
                 "--ros-args", "-p", ["use_sim_time:=", simtime],
                 "-p", "lidar_topic:=/camera_ray_clusters",
                 "-p", "output_topic:=/person_markers"],
            name="person_marker_publisher", output="screen"),
        ExecuteProcess(
            cmd=["python3", f"{WS}/src/social_perception/social_perception/camera_lidar/human_kf_predictor_lidar.py",
                 "--ros-args", "-p", ["use_sim_time:=", simtime],
                 "-p", ["coast_timeout:=", coast]],
            name="human_kf_predictor_lidar", output="screen"),
        ExecuteProcess(
            cmd=["python3", f"{WS}/src/social_perception/social_perception/camera_lidar/predicted_person_cloud_node_lidar.py",
                 "--ros-args", "-p", ["use_sim_time:=", simtime]],
            name="predicted_person_cloud_node_lidar", output="screen"),
    ])


