from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# ABLATION F: publishes the CostmapFilterInfo that KeepoutFilter needs
# in order to interpret /social_zone_map. Without this the filter loads,
# logs that it is waiting for filter info, and applies nothing.
#
# The MASK itself comes from social_zone_costmap_node_sim.py (latched,
# TRANSIENT_LOCAL) - NOT from a map_server, unlike the Nav2 keepout
# tutorial, because the zone is generated per episode from perception
# rather than annotated on a map file. Only the info server is needed.
#
# Both nodes are lifecycle-managed: costmap_filter_info_server is a
# lifecycle node and stays unconfigured (publishing nothing) unless a
# lifecycle manager brings it up, hence the manager below.

PARAMS = ("/root/thesis_social_navigation_ws/config/"
          "social_nav2_ablation_F_socialzone_sim.yaml")


def generate_launch_description():
    simtime = LaunchConfiguration("use_sim_time")
    params = LaunchConfiguration("params_file")

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("params_file", default_value=PARAMS),
        Node(
            package="nav2_map_server",
            executable="costmap_filter_info_server",
            name="costmap_filter_info_server",
            output="screen",
            emulate_tty=True,
            parameters=[params, {"use_sim_time": simtime}],
        ),
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_costmap_filters",
            output="screen",
            emulate_tty=True,
            parameters=[{
                "use_sim_time": simtime,
                "autostart": True,
                "node_names": ["costmap_filter_info_server"],
            }],
        ),
    ])
