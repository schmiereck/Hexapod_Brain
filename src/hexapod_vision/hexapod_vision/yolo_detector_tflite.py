import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import cv2
import numpy as np
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite


class YOLODetectorTFLite(Node):
    """YOLO Object Detector using TensorFlow Lite for Raspberry Pi optimization."""
    
    # COCO class names (80 classes)
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    def __init__(self):
        super().__init__('yolo_detector_tflite')
        
        # Parameters
        self.declare_parameter('model_path', 'yolov8n_float32.tflite')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('image_topic', '/raspclaws/camera/image_raw')
        self.declare_parameter('use_compressed', True)
        self.declare_parameter('input_size', 640)
        
        model_path = self.get_parameter('model_path').value
        self.conf_threshold = self.get_parameter('confidence_threshold').value
        self.iou_threshold = self.get_parameter('iou_threshold').value
        image_topic = self.get_parameter('image_topic').value
        use_compressed = self.get_parameter('use_compressed').value
        self.input_size = self.get_parameter('input_size').value
        
        self.get_logger().info(f"Loading TFLite YOLO model: {model_path}...")
        try:
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            self.get_logger().info(f"Model loaded successfully.")
            self.get_logger().info(f"Input shape: {self.input_details[0]['shape']}")
            self.get_logger().info(f"Output shape: {self.output_details[0]['shape']}")
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            raise

        self.bridge = CvBridge()
        
        # Subscriber & Publisher
        if use_compressed:
            self.image_sub = self.create_subscription(
                CompressedImage, image_topic + '/compressed', self.compressed_image_callback, 10)
            self.get_logger().info(f"YOLO TFLite Detector Node started. Listening on {image_topic}/compressed")
        else:
            self.image_sub = self.create_subscription(
                Image, image_topic, self.image_callback, 10)
            self.get_logger().info(f"YOLO TFLite Detector Node started. Listening on {image_topic}")
        
        self.detection_pub = self.create_publisher(
            Detection2DArray, '/hexapod/detections', 10)
        
        self.annotated_pub = self.create_publisher(
            Image, '/hexapod/detections/image', 10)
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = self.get_clock().now()
    
    def preprocess_image(self, image):
        """Preprocess image for YOLO input."""
        # Resize and pad to maintain aspect ratio
        input_h, input_w = self.input_size, self.input_size
        img_h, img_w = image.shape[:2]
        
        # Calculate scaling factor
        scale = min(input_w / img_w, input_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create padded image
        padded = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
        pad_x = (input_w - new_w) // 2
        pad_y = (input_h - new_h) // 2
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        
        # Normalize to [0, 1] and add batch dimension
        input_data = padded.astype(np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)
        
        return input_data, scale, pad_x, pad_y
    
    def postprocess_detections(self, output, original_shape, scale, pad_x, pad_y):
        """Convert YOLOv8 model output to detections.
        
        YOLOv8 TFLite format: [1, 84, 8400]
        - 84 channels: [x_center, y_center, width, height, class_0_logit, ..., class_79_logit]
        - 8400 predictions
        - Coordinates are in model input space (640x640), NOT normalized [0,1]
        - Class scores are logits and need sigmoid activation
        """
        detections = []
        
        # Transpose from [1, 84, 8400] to [8400, 84]
        predictions = output[0].T  # Now shape is [8400, 84]
        
        max_conf_seen = 0.0  # Debug: track max confidence
        for pred in predictions:
            # Extract box coordinates (first 4 values) - already in 640x640 pixel space
            x_center, y_center, width, height = pred[:4]
            
            # YOLOv8 has NO objectness score, only class scores (as logits)
            class_logits = pred[4:]  # 80 class logits
            
            # Apply sigmoid to convert logits to probabilities
            class_scores = 1 / (1 + np.exp(-class_logits))
            
            # Get class with highest score
            class_id = np.argmax(class_scores)
            class_conf = class_scores[class_id]
            
            # Track max confidence for debugging
            if class_conf > max_conf_seen:
                max_conf_seen = class_conf
            
            # Use class confidence as final confidence
            if class_conf < self.conf_threshold:
                continue
            
            # Convert from padded 640x640 space to original image space
            # Remove padding offset first
            x_center = (x_center - pad_x) / scale
            y_center = (y_center - pad_y) / scale
            width = width / scale
            height = height / scale
            
            # Convert to corner coordinates
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(class_conf),
                'class_id': int(class_id),
                'class_name': self.COCO_CLASSES[class_id] if class_id < len(self.COCO_CLASSES) else f"class_{class_id}"
            })
        
        # Debug log
        if len(detections) == 0:
            self.get_logger().debug(f"No detections above threshold {self.conf_threshold}. Max confidence seen: {max_conf_seen:.4f}")
        
        # Apply NMS (Non-Maximum Suppression)
        detections = self.non_max_suppression(detections)
        
        return detections
    
    def non_max_suppression(self, detections):
        """Apply Non-Maximum Suppression to remove overlapping boxes."""
        if len(detections) == 0:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        kept = []
        while len(detections) > 0:
            best = detections.pop(0)
            kept.append(best)
            
            # Remove boxes with high IoU with the best box
            detections = [
                d for d in detections 
                if self.iou(best['bbox'], d['bbox']) < self.iou_threshold
            ]
        
        return kept
    
    def iou(self, box1, box2):
        """Calculate Intersection over Union."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)
        
        if xi_max < xi_min or yi_max < yi_min:
            return 0.0
        
        intersection = (xi_max - xi_min) * (yi_max - yi_min)
        
        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def compressed_image_callback(self, msg):
        try:
            # Decode compressed image
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_image is None:
                self.get_logger().error("Failed to decode compressed image")
                return
            original_shape = cv_image.shape[:2]
        except Exception as e:
            self.get_logger().error(f"Compressed image decode error: {e}")
            return
        
        # Preprocess
        input_data, scale, pad_x, pad_y = self.preprocess_image(cv_image)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Postprocess
        detections = self.postprocess_detections(output, original_shape, scale, pad_x, pad_y)
        
        # Create Detection2DArray
        detection_array = Detection2DArray()
        detection_array.header = msg.header
        
        for det in detections:
            detection = Detection2D()
            
            # Bounding Box
            x1, y1, x2, y2 = det['bbox']
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
            hypothesis.hypothesis.class_id = det['class_name']
            hypothesis.hypothesis.score = float(det['confidence'])
            detection.results.append(hypothesis)
            
            detection_array.detections.append(detection)
        
        # Publish detections
        self.detection_pub.publish(detection_array)
        
        # Publish annotated image (for debugging)
        if self.annotated_pub.get_subscription_count() > 0:
            annotated = cv_image.copy()
            for det in detections:
                x1, y1, x2, y2 = [int(v) for v in det['bbox']]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{det['class_name']}: {det['confidence']:.2f}"
                cv2.putText(annotated, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated, "bgr8")
            annotated_msg.header = msg.header
            self.annotated_pub.publish(annotated_msg)
        
        # Log FPS occasionally
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            now = self.get_clock().now()
            elapsed = (now - self.start_time).nanoseconds / 1e9
            fps = self.frame_count / elapsed
            self.get_logger().info(f'FPS: {fps:.2f}, Detections: {len(detections)}')
    
    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            original_shape = cv_image.shape[:2]
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return
        
        # Preprocess
        input_data, scale, pad_x, pad_y = self.preprocess_image(cv_image)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Postprocess
        detections = self.postprocess_detections(output, original_shape, scale, pad_x, pad_y)
        
        # Create Detection2DArray
        detection_array = Detection2DArray()
        detection_array.header = msg.header
        
        for det in detections:
            detection = Detection2D()
            
            # Bounding Box
            x1, y1, x2, y2 = det['bbox']
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
            hypothesis.hypothesis.class_id = det['class_name']
            hypothesis.hypothesis.score = float(det['confidence'])
            detection.results.append(hypothesis)
            
            detection_array.detections.append(detection)
        
        # Publish detections
        self.detection_pub.publish(detection_array)
        
        # Publish annotated image (for debugging)
        if self.annotated_pub.get_subscription_count() > 0:
            annotated = cv_image.copy()
            for det in detections:
                x1, y1, x2, y2 = [int(v) for v in det['bbox']]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{det['class_name']}: {det['confidence']:.2f}"
                cv2.putText(annotated, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated, "bgr8")
            annotated_msg.header = msg.header
            self.annotated_pub.publish(annotated_msg)
        
        # Log FPS occasionally
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            now = self.get_clock().now()
            elapsed = (now - self.start_time).nanoseconds / 1e9
            fps = self.frame_count / elapsed
            self.get_logger().info(f'FPS: {fps:.2f}, Detections: {len(detections)}')


def main(args=None):
    rclpy.init(args=args)
    node = YOLODetectorTFLite()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
