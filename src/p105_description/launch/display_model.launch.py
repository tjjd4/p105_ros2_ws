import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command
from launch_ros.actions import Node

def generate_launch_description():

    package_name = 'p105_description'
    robot_file_name = 'p105.xacro'

    robot_file_path = os.path.join(
        get_package_share_directory(package_name),
        "urdf",
        robot_file_name,
    )

    with open(robot_file_path, "r") as infp:
        robot_desc = infp.read()
    # params = {"robot_description": robot_desc}
    params = {"robot_description": Command(["xacro ", robot_file_path])}

    rviz_config_file_path = os.path.join(
        get_package_share_directory(package_name),
        "rviz",
        "default.rviz",
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config_file_path],
        output="screen",
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[params],
    )

    joint_state_publisher_gui_node = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        name="joint_state_publisher_gui",
    )

    return LaunchDescription(
        [
            rviz_node,
            robot_state_publisher_node,
            joint_state_publisher_gui_node,
        ]
    )