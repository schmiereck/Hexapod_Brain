# Hexapod Brain

## Project Overview
This repository contains the high-level intelligence code for the Adeept RaspClaws robot. It is designed to run on a dedicated compute node (e.g., a Raspberry Pi 5 named "ubuntu1") and communicates with the robot hardware node ("raspclaws-1") via ROS2 over a local network.

The system acts as the "Brain" of the robot, handling:
*   **Perception:** Object detection using YOLOv8 (TFLite) via the `hexapod_vision` package.
*   **Planning & Decision Making:** High-level reasoning and navigation logic powered by the Google Gemini API via the `hexapod_navigation` package (Gemini Bridge).

### Main Technologies
*   **ROS2 (Humble):** Middleware for communication between the brain and robot hardware.
*   **Python 3.10:** Primary programming language.
*   **PyTorch & Ultralytics:** For YOLO object detection.
*   **Google Gemini API:** For high-level reasoning and decision-making.

## Building and Running

### Environment Setup
The project relies on a Conda/Micromamba environment defined in `environment.yml`. Before running ROS2 commands, you must source the environment setup script:
```bash
source ~/Hexapod_Brain/setup_env.bash
```
This script sets up the local workspace and configures `ROS_DOMAIN_ID=1` for communication with the robot.

### Build Workspace
To build the ROS2 workspace, run:
```bash
cd ~/Hexapod_Brain
colcon build --symlink-install
source install/setup.bash
```

### Running the System
The system is typically split into vision and navigation nodes, run in separate terminals.

**1. Start Vision (YOLO Detector):**
```bash
source ~/Hexapod_Brain/setup_env.bash
ros2 launch hexapod_vision yolo_detector_tflite.launch.py
```

**2. Start Navigation (Gemini Bridge):**
```bash
source ~/Hexapod_Brain/setup_env.bash
ros2 launch hexapod_navigation gemini_bridge.launch.py
```

**3. Send a Goal:**
You can publish string goals to trigger the navigation reasoning loop:
```bash
source ~/Hexapod_Brain/setup_env.bash
ros2 topic pub --once /hexapod/goal std_msgs/String "data: 'Find the bottle'"
```

## Development Conventions

*   **Memory and Context Tracking:** The workspace relies heavily on memory files located in the `memory/` directory. 
    *   `memory/YYYY-MM-DD.md`: Daily logs and context.
    *   `memory/MEMORY.md`: Curated long-term memory and lessons learned.
    *   `memory/TODO.md`: Task tracking.
    *   Agents are expected to read these files to establish context at the beginning of sessions and update them as tasks progress. See `AGENTS.md` for detailed rules.
*   **API Keys:** The Gemini API requires a key. Ensure `GEMINI_API_KEY` is exported in the environment (typically in `~/.bashrc`).
*   **Testing:** New features, particularly those related to the Gemini integration, should be tested empirically. Refer to `TESTING_GUIDE.md` for specific test scenarios (e.g., "Find the bottle", "Find a person").
