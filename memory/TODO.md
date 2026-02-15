# TODO.md

## ✅ Current Status (Phase 1)
*   [x] Basic `hexapod_vision` package structure created.
*   [x] `yolo_detector.py` implemented (Initial version).
*   [x] `environment.yml` for Micromamba created.

## 📝 TODOs (Immediate)
1.  **Verify Setup**: Check if `colcon build` works and dependencies are met.
2.  **Test YOLO**: Run `yolo_detector.launch.py` and verify detections from `raspclaws-1` stream.
3.  **Phase 2 Start**: Implement `NavigationNode` that subscribes to `/hexapod/detections` and calls Actions on `raspclaws-1`.
