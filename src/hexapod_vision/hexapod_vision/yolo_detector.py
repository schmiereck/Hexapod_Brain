import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np

class YOLODetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        
        # Parameters
        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('image_topic', '/raspclaws/camera/image_raw')
        
        model_path = self.get_parameter('model_path').value
        self.conf_threshold = self.get_parameter('confidence_threshold').value
        image_topic = self.get_parameter('image_topic').value
        
        self.get_logger().info(f"Loading YOLO model: {model_path}...")
        try:
            self.model = YOLO(model_path)
            self.get_logger().info("Model loaded successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            return

        self.bridge = CvBridge()
        
        # Subscriber & Publisher
        self.image_sub = self.create_subscription(
            Image, image_topic, self.image_callback, 10)
        
        self.detection_pub = self.create_publisher(
            Detection2DArray, '/hexapod/detections', 10)
        
        self.annotated_pub = self.create_publisher(
            Image, '/hexapod/detections/image', 10)
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = self.get_clock().now()
        
        self.get_logger().info(f'YOLO Detector Node started. Listening on {image_topic}')
    
    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return
        
        # YOLO Inference
        # verbose=False to reduce console noise
        results = self.model(cv_image, conf=self.conf_threshold, verbose=False)
        
        # Create Detection2DArray
        detection_array = Detection2DArray()
        detection_array.header = msg.header
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                detection = Detection2D()
                
                # Bounding Box (xyxy format from YOLO to center/size for VisionMsgs)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w = x2 - x1
                h = y2 - y1
                cx = x1 + w / 2
                cy = y1 + h / 2
                
                detection.bbox.center.position.x = float(cx)
                detection.bbox.center.position.y = float(cy)
                detection.bbox.size_x = float(w)
                detection.bbox.size_y = float(h)
                
                # Class and Confidence
                hypothesis = ObjectHypothesisWithPose()
                class_id = int(box.cls[0])
                hypothesis.hypothesis.class_id = str(class_id)
                # Try to get class name if available
                if self.model.names and class_id in self.model.names:
                    # Storing name in class_id might break strict typing if expecting int ID, 
                    # but vision_msgs uses string.
                    hypothesis.hypothesis.class_id = self.model.names[class_id]
                
                hypothesis.hypothesis.score = float(box.conf[0])
                detection.results.append(hypothesis)
                
                detection_array.detections.append(detection)
        
        # Publish detections
        self.detection_pub.publish(detection_array)
        
        # Publish annotated image (for debugging)
        if self.annotated_pub.get_subscription_count() > 0:
            annotated = results[0].plot()
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated, "bgr8")
            annotated_msg.header = msg.header
            self.annotated_pub.publish(annotated_msg)
        
        # Log FPS occasionally
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            now = self.get_clock().now()
            elapsed = (now - self.start_time).nanoseconds / 1e9
            fps = self.frame_count / elapsed
            self.get_logger().info(f'FPS: {fps:.2f}, Detections: {len(detection_array.detections)}')

def main(args=None):
    rclpy.init(args=args)
    node = YOLODetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
