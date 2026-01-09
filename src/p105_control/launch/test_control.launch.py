from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    description_pkg_share = get_package_share_directory("p105_description")
    control_pkg_share = get_package_share_directory("p105_control")

    robot_file_path = os.path.join(
        get_package_share_directory("p105_description"),
        "urdf",
        "p105.xacro",
    )

    robot_controllers_path = os.path.join(
        get_package_share_directory("p105_control"),
        "config",
        "ros2_controllers.yaml",
    )

    rviz_config_file_path = os.path.join(
        get_package_share_directory("p105_description"),
        "rviz",
        "default.rviz",
    )
    robot_description = Command(["xacro ", robot_file_path])

    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            {"robot_description": robot_description},
            robot_controllers_path,
        ],
        output="screen",
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description}],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config_file_path],
        output="screen",
    )

    # Node: Spawner for Joint State Broadcaster, publish joint states to /joint_states
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
    )

    # Node: Spawner for Forward Position Controller
    robot_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["forward_command_controller"],
    )

    delay_robot_controller_spawner_after_joint_state_broadcaster_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[robot_controller_spawner],
        )
    )

    return LaunchDescription(
        [
            control_node,
            robot_state_publisher_node,
            rviz_node,
            joint_state_broadcaster_spawner,
            delay_robot_controller_spawner_after_joint_state_broadcaster_spawner,
        ]
    )