from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('p105_gait_controller')
    config_file = os.path.join(pkg_share, 'config', 'gait_controller.yaml')

    gait_controller_node = Node(
        package='p105_gait_controller',
        executable='gait_controller',
        name='gait_controller',
        output='screen',
        parameters=[config_file],
        emulate_tty=True,
    )

    return LaunchDescription([
        gait_controller_node,
    ])
