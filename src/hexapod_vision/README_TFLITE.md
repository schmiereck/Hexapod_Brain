# YOLOv8 TensorFlow Lite Setup

## Prerequisites

Install TensorFlow Lite Runtime on ubuntu1:

```bash
pip3 install tflite-runtime
```

If tflite-runtime is not available for your platform, install full TensorFlow:

```bash
pip3 install tensorflow
```

## Convert YOLOv8 to TensorFlow Lite

### Option 1: Convert using Ultralytics (requires PyTorch temporarily)

On a development machine with PyTorch working:

```bash
pip install ultralytics
yolo export model=yolov8n.pt format=tflite imgsz=640
```

This creates `yolov8n_float32.tflite` or `yolov8n_saved_model/yolov8n_float32.tflite`.

### Option 2: Download Pre-converted Model

Download pre-converted YOLOv8n TFLite model:

```bash
# From Ultralytics or community sources
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n_float32.tflite
```

### Option 3: Export via Python Script

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')

# Export to TFLite
model.export(format='tflite', imgsz=640)
```

## Model Placement

Place the `.tflite` model file in the home directory or specify the full path:

```bash
# Place model in home directory
cp yolov8n_float32.tflite ~/

# Or specify full path in launch file
ros2 launch hexapod_vision yolo_detector_tflite.launch.py model_path:=/path/to/yolov8n_float32.tflite
```

## Build and Run

```bash
cd ~/Hexapod_Brain
export ROS_DOMAIN_ID=1
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash

# Run TFLite detector
ros2 launch hexapod_vision yolo_detector_tflite.launch.py
```

## Testing

Check detections:

```bash
export ROS_DOMAIN_ID=1
source /opt/ros/humble/setup.bash

# List topics
ros2 topic list | grep hexapod

# Echo detections
ros2 topic echo /hexapod/detections

# View annotated image (requires image viewer)
ros2 run rqt_image_view rqt_image_view /hexapod/detections/image
```

## Performance Notes

- **Expected FPS**: 5-15 FPS on Raspberry Pi 5
- **Input size**: 640x640 (configurable via parameter)
- **Model variants**: yolov8n (nano), yolov8s (small), yolov8m (medium)
  - Nano is recommended for Raspberry Pi 5

## Troubleshooting

### "No module named 'tflite_runtime'"

Install: `pip3 install tflite-runtime` or `pip3 install tensorflow`

### "Model file not found"

Check model path parameter matches actual file location.

### Low FPS

- Use yolov8n (nano) instead of larger models
- Reduce input_size parameter (e.g., 320 instead of 640)
- Consider quantized INT8 model for better performance

### No detections

- Check confidence_threshold (try lowering to 0.3)
- Verify camera stream: `ros2 topic hz /raspclaws/camera/image_raw`
- Check logs: `ros2 run hexapod_vision yolo_detector_tflite`
