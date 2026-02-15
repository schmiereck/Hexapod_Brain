# TODO.md - Hexapod Brain Project

## ✅ Completed (Phase 1)
*   [x] Basic `hexapod_vision` package structure created
*   [x] `yolo_detector.py` implemented (PyTorch version - **has issues on Pi**)
*   [x] `yolo_detector_tflite.py` implemented (TFLite version - **recommended**)
*   [x] `environment.yml` for Micromamba created
*   [x] TFLite runtime installed on ubuntu1
*   [x] Package builds successfully on ubuntu1
*   [x] ROS2 services verified on raspclaws-1 (gui_server, ros_server)
*   [x] ROS2 topics and actions verified on ubuntu1
*   [x] **YOLOv8n TFLite model downloaded and uploaded to ubuntu1**
*   [x] **TFLite detector successfully starts and loads model**

## 📝 Current TODOs (Phase 1 - Testing & Validation)
1.  **Test with Live Camera Stream**:
    - Ensure raspclaws-1 gui_server is publishing camera frames
    - Run detector and verify detections: `ros2 topic echo /hexapod/detections`
    - Check FPS and latency

2.  **Optimize if Needed**:
    - If FPS is low (<5), consider lowering input_size parameter
    - Test with compressed image stream instead of raw

3.  **Documentation Update**:
    - Document successful TFLite setup in README
    - Add model download instructions

## 📋 Next Phase (Phase 2 - Navigation)
1.  **Create hexapod_navigation Package**:
    - Navigation node that subscribes to `/hexapod/detections`
    - Implements behavior logic (e.g., "approach detected person")
    - Calls ROS2 Actions on raspclaws-1 (`/raspclaws/linear_move`, `/raspclaws/rotate`)

2.  **Implement Simple Behaviors**:
    - Object tracking (rotate camera to keep object centered)
    - Approach behavior (move towards detected object)
    - Collision avoidance (stop if too close)

3.  **Integration Testing**:
    - End-to-end test: Camera → Detection → Navigation → Movement
