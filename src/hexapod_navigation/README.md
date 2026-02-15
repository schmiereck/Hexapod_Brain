# Hexapod Navigation Package

Vision-based navigation and behavior logic for the Hexapod robot.

## Nodes

### `bottle_seeker`
Proof-of-concept node that searches for a bottle using head scanning and navigates towards it.

**Subscriptions:**
- `/hexapod/detections` (vision_msgs/Detection2DArray) - Object detections from YOLO detector

**Action Clients:**
- `/raspclaws/head_position` - Control camera/head position
- `/raspclaws/rotate` - Rotate robot in place
- `/raspclaws/linear_move` - Move robot forward/backward

**Publishers:**
- `/hexapod/navigation/status` (std_msgs/String) - Current state (SEARCHING, CENTERING, APPROACHING, ARRIVED)
- `/hexapod/navigation/target` (geometry_msgs/Point) - Target bottle position in image

## Behavior States

### 1. SEARCHING
Scans for bottle by panning the head left-right across predefined positions. If bottle not found after full head scan, rotates the robot 30° and repeats.

### 2. CENTERING
Centers the detected bottle in the camera frame by rotating the robot left or right.

### 3. APPROACHING
Moves forward towards the bottle while maintaining centering. Uses bounding box width as distance proxy.

### 4. ARRIVED
Mission complete. Celebrates with head wiggle and stops.

## Safety Features
- **Detection Loss**: Returns to SEARCHING if bottle not seen for 3 seconds
- **Timeout**: Aborts after 60 seconds of searching
- **Confidence Threshold**: Only trusts detections > 0.5 confidence
- **Small Movements**: 10cm forward steps, 15° rotation steps

## Usage

### Prerequisites
1. Start services on raspclaws-1:
   ```bash
   sudo systemctl start gui_server ros_server
   ```

2. Enable camera on ubuntu1:
   ```bash
   export ROS_DOMAIN_ID=1
   source /opt/ros/humble/setup.bash
   ros2 service call /raspclaws/set_camera_pause std_srvs/srv/SetBool "{data: false}"
   ```

3. Start YOLO detector:
   ```bash
   source ~/Hexapod_Brain/install/setup.bash
   ros2 launch hexapod_vision yolo_detector_tflite.launch.py
   ```

### Run Bottle Seeker
```bash
source ~/Hexapod_Brain/install/setup.bash
ros2 launch hexapod_navigation bottle_seeker.launch.py
```

### Monitor Status
```bash
# Watch state transitions
ros2 topic echo /hexapod/navigation/status

# Watch target position
ros2 topic echo /hexapod/navigation/target

# Watch detections
ros2 topic echo /hexapod/detections
```

## Parameters
See `config/bottle_seeker_params.yaml` for all configurable parameters:
- Detection settings (target class, confidence threshold)
- Head scan positions and timing
- Centering tolerance and rotation steps
- Approach distance and goal thresholds
- Timeouts

## Development

Build package:
```bash
cd ~/Hexapod_Brain
colcon build --packages-select hexapod_navigation --symlink-install
```

Test with linter:
```bash
ament_flake8 src/hexapod_navigation
```

## Future Enhancements
- Multi-object avoidance
- Path planning around obstacles
- World model / SLAM
- Reinforcement learning for optimal movement
