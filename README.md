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
   colcon build --symlink-install --packages-select hexapod_navigation
   source install/setup.bash
   ```

   4. **Run**
      **old Yolo detector:**
      ```bash
      ros2 launch hexapod_vision yolo_detector.launch.py
      ```
      **new Gemini bridge:**
      ```bash
      cd ~/Hexapod_Brain
      source ~/Hexapod_Brain/setup_env.bash
      ros2 launch hexapod_navigation gemini_bridge.launch.py
      ```

      **Change Parameter:**
      ```bash
      ros2 launch hexapod_navigation gemini_bridge.launch.py --show-args
      
      nano src/hexapod_navigation/config/gemini_bridge_params.yaml
      
      ros2 run hexapod_navigation gemini_bridge --ros-args -p timeout:=900.0
      
           parameter('gemini_api_key', '')
           parameter('model_name', 'models/gemini-robotics-er-1.5-preview')
           parameter('timeout', 600.0)
           parameter('image_topic', '/raspclaws/camera/image_raw/compressed')
           parameter('detection_topic', '/hexapod/detections')
           parameter('control_loop_hz', 1.0)
           parameter('max_retries', 3)
           parameter('retry_delay', 2.0)
           parameter('use_yolo', False)  # Phase 4: Pure vision mode by default
           parameter('max_history_length', 8)  # Phase 5a: Conversation history
      ```
