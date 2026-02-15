from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='hexapod_vision',
            executable='yolo_detector_tflite',
            name='yolo_detector_tflite',
            parameters=[{
                'model_path': 'yolov8n_float32.tflite',
                'confidence_threshold': 0.5,
                'iou_threshold': 0.45,
                'image_topic': '/raspclaws/camera/image_raw',
                'input_size': 640  # Must match TFLite model input size
            }],
            output='screen'
        )
    ])
