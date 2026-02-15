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

## 📝 Current TODOs (Phase 1 - Final Step)
1.  **Create YOLOv8 TFLite Model**:
    - **Option A** (Preferred): Convert on Windows PC with working PyTorch
      - Fix ONNX dependencies issue in `scripts/export_yolo_tflite.py`
      - Or use online conversion tool
    - **Option B**: Download pre-converted YOLOv8n TFLite model from community
    - See `src/hexapod_vision/README_TFLITE.md` for details

2.  **Upload Model to ubuntu1**:
    ```bash
    scp yolov8n_float32.tflite ubuntu@192.168.2.133:~/
    ```

3.  **Test YOLOv8 TFLite Detector**:
    ```bash
    # On ubuntu1
    export ROS_DOMAIN_ID=1
    source /opt/ros/humble/setup.bash
    cd ~/Hexapod_Brain
    source install/setup.bash
    ros2 launch hexapod_vision yolo_detector_tflite.launch.py
    ```

4.  **Verify Detections**:
    ```bash
    ros2 topic echo /hexapod/detections
    ```

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
