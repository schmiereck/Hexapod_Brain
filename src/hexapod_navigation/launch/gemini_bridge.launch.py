#!/usr/bin/env python3
"""
Launch file for Gemini Bridge node.
Starts LLM-based robot control with embodied reasoning.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for Gemini Bridge."""
    
    # Declare arguments
    gemini_api_key_arg = DeclareLaunchArgument(
        'gemini_api_key',
        default_value='',
        description='Gemini API key (or set GEMINI_API_KEY env var)'
    )
    
    model_name_arg = DeclareLaunchArgument(
        'model_name',
        default_value='gemini-2.5-flash',
        description='Gemini model (2.5-flash is better for structured output than robotics-er-1.5-preview)'
    )
    
    timeout_arg = DeclareLaunchArgument(
        'timeout',
        default_value='120.0',
        description='Maximum time for goal completion (seconds)'
    )
    
    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/raspclaws/camera/image_raw/compressed',
        description='Compressed image topic to subscribe to'
    )
    
    detection_topic_arg = DeclareLaunchArgument(
        'detection_topic',
        default_value='/hexapod/detections',
        description='Detection topic to subscribe to'
    )
    
    control_loop_hz_arg = DeclareLaunchArgument(
        'control_loop_hz',
        default_value='1.0',
        description='Control loop frequency (Hz). 1 Hz is good for LLM latency.'
    )
    
    max_retries_arg = DeclareLaunchArgument(
        'max_retries',
        default_value='3',
        description='Max retries for Gemini API calls'
    )
    
    retry_delay_arg = DeclareLaunchArgument(
        'retry_delay',
        default_value='2.0',
        description='Delay between retries (seconds)'
    )
    
    # Node
    gemini_bridge_node = Node(
        package='hexapod_navigation',
        executable='gemini_bridge',
        name='gemini_bridge',
        output='screen',
        parameters=[{
            'gemini_api_key': LaunchConfiguration('gemini_api_key'),
            'model_name': LaunchConfiguration('model_name'),
            'timeout': LaunchConfiguration('timeout'),
            'image_topic': LaunchConfiguration('image_topic'),
            'detection_topic': LaunchConfiguration('detection_topic'),
            'control_loop_hz': LaunchConfiguration('control_loop_hz'),
            'max_retries': LaunchConfiguration('max_retries'),
            'retry_delay': LaunchConfiguration('retry_delay'),
        }]
    )
    
    return LaunchDescription([
        gemini_api_key_arg,
        model_name_arg,
        timeout_arg,
        image_topic_arg,
        detection_topic_arg,
        control_loop_hz_arg,
        max_retries_arg,
        retry_delay_arg,
        gemini_bridge_node,
    ])
