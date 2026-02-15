#!/usr/bin/env python3
"""
Export YOLOv8 model to TensorFlow Lite format.
Run this on a machine with working PyTorch and Ultralytics.
"""

import sys
import os

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed")
    print("Install with: pip install ultralytics")
    sys.exit(1)

def export_yolo_to_tflite(model_name='yolov8n', imgsz=640):
    """Export YOLO model to TFLite format."""
    print(f"Exporting {model_name} to TensorFlow Lite...")
    
    # Load model
    model = YOLO(f'{model_name}.pt')
    
    # Export to TFLite
    success = model.export(format='tflite', imgsz=imgsz)
    
    if success:
        tflite_file = f'{model_name}_saved_model/{model_name}_float32.tflite'
        if os.path.exists(tflite_file):
            print(f"\n✅ Export successful!")
            print(f"📁 TFLite model saved to: {tflite_file}")
            print(f"\nTo use on Raspberry Pi:")
            print(f"  scp {tflite_file} ubuntu@192.168.2.133:~/yolov8n_float32.tflite")
        else:
            print(f"⚠️ Export completed but file not found at expected location")
            print(f"Check current directory for .tflite files")
    else:
        print("❌ Export failed")
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = 'yolov8n'
    
    export_yolo_to_tflite(model_name)
