#!/usr/bin/env python3
"""Launch file for bottle seeker node."""

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    config_dir = get_package_share_directory('hexapod_navigation')
    params_file = os.path.join(config_dir, 'config', 'bottle_seeker_params.yaml')
    
    return LaunchDescription([
        Node(
            package='hexapod_navigation',
            executable='bottle_seeker',
            name='bottle_seeker',
            output='screen',
            parameters=[params_file],
        ),
    ])
