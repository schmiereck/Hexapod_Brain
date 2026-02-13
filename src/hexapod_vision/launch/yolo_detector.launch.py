from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='hexapod_vision',
            executable='yolo_detector',
            name='yolo_detector',
            parameters=[{
                'model_path': 'yolov8n.pt',
                'confidence_threshold': 0.5,
                'image_topic': '/raspclaws/camera/image_raw'
            }],
            output='screen'
        )
    ])
