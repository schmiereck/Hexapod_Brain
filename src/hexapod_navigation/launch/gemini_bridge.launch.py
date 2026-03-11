#!/usr/bin/env python3
"""
Launch file for Gemini Bridge node.
Starts LLM-based robot control with embodied reasoning using a YAML config.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """Generate launch description for Gemini Bridge."""
    
    config_dir = get_package_share_directory('hexapod_navigation')
    params_file = os.path.join(config_dir, 'config', 'gemini_bridge_params.yaml')
    
    # Node
    gemini_bridge_node = Node(
        package='hexapod_navigation',
        executable='gemini_bridge',
        name='gemini_bridge',
        output='screen',
        parameters=[params_file]
    )
    
    return LaunchDescription([
        gemini_bridge_node,
    ])
