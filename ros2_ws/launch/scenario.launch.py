from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PythonExpression

WS = "/root/thesis_social_navigation_ws"

# One demo scenario from start to finish:
#   ros2 launch /root/thesis_social_navigation_ws/launch/scenario.launch.py scenario:=headon
#
# scenario:  headon | headon_blocked | conversation_wide | conversation_narrow | queue
# bag_name:  name of the recorded bag (default: demo_<scenario>_<date>_<time>)
# rviz:      true/false, open RViz            (default true)
# gazebo_gui: true/false, open the Gazebo window (default true)
#
# This is a thin wrapper: it runs run_scenario.sh, which does the sequencing
# (Gazebo + Nav2, AMCL pose, undock, perception, bag, goal, teardown, summary).
# A launch file alone cannot do those waits, so they stay in the script.


def generate_launch_description():
    scenario = LaunchConfiguration("scenario")
    bag_name = LaunchConfiguration("bag_name")
    rviz = LaunchConfiguration("rviz")
    gui = LaunchConfiguration("gazebo_gui")

    run = ExecuteProcess(
        cmd=[f"{WS}/run_scenario.sh", scenario, bag_name],
        additional_env={
            "SHOW_RVIZ": rviz,
            "HEADLESS": PythonExpression(["'false' if '", gui, "' == 'true' else 'true'"]),
        },
        output="screen",
        emulate_tty=True,
    )
    return LaunchDescription([
        DeclareLaunchArgument("scenario", description="headon | headon_blocked | "
                              "conversation_wide | conversation_narrow | queue"),
        # Empty = run_scenario.sh makes a name from the scenario and the time.
        DeclareLaunchArgument("bag_name", default_value=""),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("gazebo_gui", default_value="true"),
        run,
        RegisterEventHandler(OnProcessExit(target_action=run, on_exit=[EmitEvent(event=Shutdown())])),
    ])
