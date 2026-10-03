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
    yolo_imgsz = LaunchConfiguration("yolo_imgsz")
    yolo_min_conf = LaunchConfiguration("yolo_min_conf")
    max_range = LaunchConfiguration("max_person_range")
    lane_slope = LaunchConfiguration("ellipse_a_slope")
    lane_max = LaunchConfiguration("ellipse_a_max")
    pass_block = LaunchConfiguration("pass_side_block_width")
    ray_coast_s = LaunchConfiguration("ray_coast_s")
    lane_b = LaunchConfiguration("ellipse_b")
    disk_r = LaunchConfiguration("person_disk_radius")

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("coast_timeout", default_value="1.5"),
        DeclareLaunchArgument("ray_coast", default_value="false"),
        # Defaults = hardware values; raised for the head-on avoidance config.
        DeclareLaunchArgument("yolo_imgsz", default_value="320"),
        DeclareLaunchArgument("yolo_min_conf", default_value="0.45"),
        DeclareLaunchArgument("max_person_range", default_value="8.0"),
        DeclareLaunchArgument("ellipse_a_slope", default_value="0.75"),
        DeclareLaunchArgument("ellipse_a_max", default_value="3.0"),
        DeclareLaunchArgument("pass_side_block_width", default_value="0.0"),
        DeclareLaunchArgument("ray_coast_s", default_value="1.5"),
        DeclareLaunchArgument("ellipse_b", default_value="0.4"),
        DeclareLaunchArgument("person_disk_radius", default_value="0.4"),
        ExecuteProcess(
            cmd=["python3", f"{WS}/src/social_perception/social_perception/camera_lidar/yolo_leg_detector_lidar_sim.py",
                 "--ros-args", "-p", ["use_sim_time:=", simtime],
                 "-p", ["imgsz:=", yolo_imgsz],
                 "-p", ["min_publish_conf:=", yolo_min_conf]],
            name="yolo_leg_detector_lidar", output="screen"),
        ExecuteProcess(
            cmd=["python3", f"{WS}/src/social_perception/social_perception/camera_lidar/camera_ray_person_node_sim.py",
                 "--ros-args", "-p", ["use_sim_time:=", simtime],
                 "-p", "scan_topic:=/scan",
                 "-p", "odom_topic:=/odom",
                 "-p", "camera_yaw_offset_deg:=0.0",
                 "-p", "camera_fx:=443.53",
                 "-p", "camera_cx:=320.0",
                 "-p", "ray_half_width_deg:=6.0",
                 "-p", ["max_person_range:=", max_range]],
            name="camera_ray_person_node", output="screen"),
        ExecuteProcess(
            cmd=["python3", f"{WS}/src/social_perception/social_perception/camera_lidar/camera_ray_identity_node_sim.py",
                 "--ros-args", "-p", ["use_sim_time:=", simtime],
                 "-p", ["coast_enable:=", ray_coast],
                 "-p", ["coast_s:=", ray_coast_s],
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
                 "--ros-args", "-p", ["use_sim_time:=", simtime],
                 "-p", ["ellipse_a_slope:=", lane_slope],
                 "-p", ["ellipse_a_max:=", lane_max],
                 "-p", ["pass_side_block_width:=", pass_block],
                 "-p", ["ellipse_b:=", lane_b],
                 "-p", ["person_disk_radius:=", disk_r]],
            name="predicted_person_cloud_node_lidar", output="screen"),
    ])


