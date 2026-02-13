# Hexapod Brain (Compute Node)

This repository contains the high-level intelligence code for the Adeept RaspClaws robot.
It is designed to run on a separate compute node (e.g., Raspberry Pi 5 "ubuntu1") and communicates with the robot ("raspclaws-1") via ROS2 over the network.

## Architecture
*   **Robot (raspclaws-1)**: Handles hardware abstraction (Servos, Camera) and executes Actions.
*   **Brain (ubuntu1)**: Handles Perception (YOLO), Planning, and Decision Making.

## Structure
*   `src/hexapod_vision`: ROS2 package for Object Detection (YOLOv8)
*   `src/hexapod_navigation`: (Future) Path planning and logic

## Setup on Ubuntu Server (ubuntu1)

1. **Install Micromamba**
   ```bash
   "${SHELL}" <(curl -L micro.mamba.pm/install.sh)
   ```

2. **Create Environment**
   ```bash
   micromamba create -f environment.yml
   micromamba activate hexapod_brain
   ```

3. **Build ROS2 Workspace**
   ```bash
   colcon build
   source install/setup.bash
   ```

4. **Run**
   ```bash
   ros2 launch hexapod_vision yolo_detector.launch.py
   ```
