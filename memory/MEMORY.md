# MEMORY.md - Hexapod Brain Project

## Projekt Taget
The Taget of the Project is defined in "docu/hexapod_vision_projekt.md".
Edit this file, if the targets are changing.

## Workflows
Allways update projects file locally on this development PC, and push the changes to the remote repository. 
This way, the project file is always up to date and can be used as a reference for the current state of the project.

## Servers
1.  **Hardware Node (`raspclaws-1` @ 192.168.2.126)**:
    *   Running `Adeept_RaspClaws` code
    *   Provides Hardware Abstraction & ROS2 Actions (`LinearMove`, `Rotate`, `HeadPosition`, `ArcMove`)
    *   Publishes Camera Stream via ZMQ → ROS2 bridge
    *   **Services**: `gui_server.service`, `ros_server.service` (systemd)
    *   **Environment**: Micromamba with `ros_env`
    *   **ROS_DOMAIN_ID**: 1 (configured in service)

2.  **Compute Node (`ubuntu1` @ 192.168.2.133)**:
    *   **THIS PROJECT (`Hexapod_Brain`)**
    *   Runs on Raspberry Pi 5 8GB (Ubuntu 22.04 + ROS2 Humble)
    *   Responsible for Perception (YOLO), Logic, and Navigation
    *   **Environment**: Native ROS2 Humble (`/opt/ros/humble/setup.bash`)
    *   **ROS_DOMAIN_ID**: Must be set to 1 manually: `export ROS_DOMAIN_ID=1`
    *   **Workspace**: `~/Hexapod_Brain`

## Available ROS2 Resources (from raspclaws-1)

### Topics
- **Camera**: `/raspclaws/camera/image_raw`, `/raspclaws/camera/image_raw/compressed`, `/raspclaws/camera/camera_info`
- **Control**: `/raspclaws/cmd_vel`, `/raspclaws/head_cmd`
- **Telemetry**: `/raspclaws/battery`, `/raspclaws/cpu_temp`, `/raspclaws/cpu_usage`, `/raspclaws/ram_usage`
- **Sensors**: `/raspclaws/gyro_data`, `/raspclaws/imu`, `/raspclaws/servo_positions`
- **Status**: `/raspclaws/status`

### Actions
- `/raspclaws/arc_move` - Curved movement trajectories
- `/raspclaws/head_position` - Camera/head positioning
- `/raspclaws/linear_move` - Straight-line movement
- `/raspclaws/rotate` - In-place rotation

### Nodes (on raspclaws-1)
- `/raspclaws_node` - Main hardware control node
- `/camera_publisher` - ZMQ-to-ROS2 camera bridge

## 📂 Project Structure (Expected)
*   `src/hexapod_vision`: ROS2 Package for YOLOv8 Object Detection.
*   `src/hexapod_navigation`: (Planned) Logic & Navigation Node.
*   `Docu/hexapod_vision_projekt.md`: **MASTER PLAN** for this project (Copied from previous repo).

## ⚠️ Known Issues & Solutions

### PyTorch on Raspberry Pi 5 (Ubuntu 22.04)
**Problem**: PyTorch shows "Illegal instruction" error on Raspberry Pi 5 with Ubuntu 22.04
- Tested versions: PyTorch 2.0.0, 2.4.0, 2.10.0 (all fail with SIGILL)
- System PyTorch (apt): 1.8.1 (too old for Ultralytics >= 8.x)
- Root cause: CPU instruction incompatibility (likely missing AVX/NEON extensions)

**Attempted Solutions (Failed)**:
1. ❌ pip install from PyTorch CPU repo
2. ❌ Downgrade to PyTorch 2.4.0
3. ❌ System packages (python3-torch 1.8.1 too old)

**Working Solution**: Use **TensorFlow Lite** instead of PyTorch for YOLOv8
- TFLite has better ARM64 optimization
- No "Illegal instruction" issues
- Good performance on Raspberry Pi 5

### Dependency Compatibility (ubuntu1)
- **NumPy**: Must use <2.0 for ROS2 Humble cv_bridge compatibility
- **OpenCV**: Use <4.10 to match NumPy 1.x
- Install: `pip3 install 'numpy<2' 'opencv-python<4.10'`

## 🔗 Important References
*   See `Docu/hexapod_vision_projekt.md` for detailed roadmap.
*   See `BRAIN_README.md` (or `README.md`) for setup instructions.
