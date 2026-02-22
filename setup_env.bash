#!/bin/bash
# Hexapod Brain Environment Setup
# Source this file before using ROS2 commands:
# source ~/Hexapod_Brain/setup_env.bash

# Set ROS Domain ID for communication with raspclaws-1
export ROS_DOMAIN_ID=1

# Source ROS2 Humble
source /opt/ros/humble/setup.bash

# Source local workspace
source ~/Hexapod_Brain/install/setup.bash

echo '✅ Hexapod Brain environment ready!'
echo '   ROS_DOMAIN_ID=1'
echo '   Workspace: ~/Hexapod_Brain'
echo ''
echo 'Available commands:'
echo '  ros2 launch hexapod_vision yolo_detector_tflite.launch.py'
echo '  ros2 launch hexapod_navigation bottle_seeker.launch.py'
