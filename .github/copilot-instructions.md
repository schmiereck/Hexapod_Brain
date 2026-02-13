# Copilot Instructions for Hexapod Brain

This repository contains the high-level intelligence code for the Adeept RaspClaws robot, designed to run on a separate compute node (e.g., Raspberry Pi 5 "ubuntu1") using ROS 2 Humble.

## Build, Test, and Lint

### Build
Use `colcon` to build the ROS 2 workspace.
```bash
colcon build --symlink-install
```
After building, source the setup script:
- **Linux/WSL**: `source install/setup.bash`
- **Windows**: `call install/setup.bat`

### Test
Run tests for a specific package:
```bash
colcon test --packages-select hexapod_vision
```
Or run pytest directly on a file:
```bash
pytest src/hexapod_vision/test/test_yolo_detector.py
```

### Lint
Follow PEP 8 standards. Run flake8 on the package:
```bash
ament_flake8 src/hexapod_vision
```

### Environment

#### Development (Local)
This project includes an `environment.yml` for **Micromamba** to create a reproducible development environment, especially on non-native ROS systems.
```bash
micromamba activate hexapod_brain
```

#### Deployment (Ubuntu Server)
The target compute node (`ubuntu1`) has ROS 2 Humble installed natively.
```bash
source /opt/ros/humble/setup.bash
```

## High-Level Architecture

### Distributed System
- **Hardware Node (`raspclaws-1`)**: Runs `Adeept_RaspClaws` code. Handles hardware abstraction (Servos, Camera) and executes Actions.
- **Compute Node (`ubuntu1`)**: This repository (`Hexapod_Brain`). Runs on Raspberry Pi 5. Handles Perception, Logic, and Navigation.

### Key Packages
1.  **`hexapod_vision`**:
    -   **Node**: `yolo_detector`
    -   **Function**: Subscribes to camera stream from `raspclaws-1`, runs YOLOv8 object detection, and publishes detections.
2.  **`hexapod_navigation`** (Planned):
    -   **Node**: `navigation_node`
    -   **Function**: Subscribes to detections and controls robot movement via ROS 2 Actions.

## Key Conventions

### ROS 2 Python
-   **Package Structure**: Follow standard `ament_python` structure.
    -   `src/package_name/package.xml`: Dependency definitions.
    -   `src/package_name/setup.py`: Entry points and installation logic.
    -   `src/package_name/package_name/`: Source code modules.
-   **Launch Files**: Python-based launch files in `src/package_name/launch/`.
    -   Naming convention: `package_name_node.launch.py`.

### Code Style
-   Use **snake_case** for function and variable names.
-   Use **CamelCase** for class names.
-   Type hinting is encouraged for all public methods.

### Communication
-   **Topics**: Use standard message types where possible (`sensor_msgs`, `vision_msgs`).
-   **Actions**: Use Action servers for long-running tasks (e.g., movement sequences).
