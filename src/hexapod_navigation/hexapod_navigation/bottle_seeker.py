#!/usr/bin/env python3
"""
Bottle Seeker Node - Phase 2.1 Proof-of-Concept
Searches for a bottle using head scanning and navigates towards it.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from enum import Enum
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import Point
from std_msgs.msg import String
from raspclaws_interfaces.action import HeadPosition, Rotate, LinearMove
import time


class State(Enum):
    SEARCHING = 1
    CENTERING = 2
    APPROACHING = 3
    ARRIVED = 4


class BottleSeeker(Node):
    def __init__(self):
        super().__init__('bottle_seeker')
        
        # Declare parameters
        self.declare_parameter('target_classes', ['bottle', 'person'])
        self.declare_parameter('min_confidence', 0.5)
        self.declare_parameter('head_scan_positions', [-60.0, -30.0, 0.0, 30.0, 60.0])
        self.declare_parameter('head_tilt_angle', -60.0)
        self.declare_parameter('head_scan_delay', 2.0)
        self.declare_parameter('image_width', 640)
        self.declare_parameter('center_tolerance', 0.2)
        self.declare_parameter('rotation_step', 15.0)
        self.declare_parameter('approach_distance', 0.1)
        self.declare_parameter('min_bottle_width', 30.0)
        self.declare_parameter('goal_bottle_width', 120.0)
        self.declare_parameter('search_timeout', 60.0)
        self.declare_parameter('detection_lost_timeout', 3.0)
        
        # Get parameters
        self.target_classes = self.get_parameter('target_classes').value
        self.min_confidence = self.get_parameter('min_confidence').value
        self.head_scan_positions = self.get_parameter('head_scan_positions').value
        self.head_tilt_angle = self.get_parameter('head_tilt_angle').value
        self.head_scan_delay = self.get_parameter('head_scan_delay').value
        self.image_width = self.get_parameter('image_width').value
        self.center_tolerance = self.get_parameter('center_tolerance').value
        self.rotation_step = self.get_parameter('rotation_step').value
        self.approach_distance = self.get_parameter('approach_distance').value
        self.min_bottle_width = self.get_parameter('min_bottle_width').value
        self.goal_bottle_width = self.get_parameter('goal_bottle_width').value
        self.search_timeout = self.get_parameter('search_timeout').value
        self.detection_lost_timeout = self.get_parameter('detection_lost_timeout').value
        
        # State machine
        self.state = State.SEARCHING
        self.current_scan_index = 0
        self.last_detection_time = None
        self.start_time = time.time()
        
        # Current bottle info
        self.bottle_bbox = None
        
        # Subscribers
        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/hexapod/detections',
            self.detection_callback,
            10
        )
        
        # Publishers
        self.status_pub = self.create_publisher(String, '/hexapod/navigation/status', 10)
        self.target_pub = self.create_publisher(Point, '/hexapod/navigation/target', 10)
        
        # Action clients
        self.head_client = ActionClient(self, HeadPosition, '/raspclaws/head_position')
        self.rotate_client = ActionClient(self, Rotate, '/raspclaws/rotate')
        self.linear_client = ActionClient(self, LinearMove, '/raspclaws/linear_move')
        
        # Wait for action servers
        self.get_logger().info('Waiting for action servers...')
        self.head_client.wait_for_server()
        self.rotate_client.wait_for_server()
        self.linear_client.wait_for_server()
        self.get_logger().info('All action servers ready!')
        
        # Main control loop
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info(f'Bottle Seeker initialized. Searching for: {", ".join(self.target_classes)}')
    
    def detection_callback(self, msg: Detection2DArray):
        """Process incoming detections."""
        self.bottle_bbox = None
        detected_class = None
        
        # Look for target objects
        for detection in msg.detections:
            for result in detection.results:
                if result.hypothesis.class_id in self.target_classes:
                    if result.hypothesis.score >= self.min_confidence:
                        self.bottle_bbox = detection.bbox
                        detected_class = result.hypothesis.class_id
                        self.last_detection_time = time.time()
                        
                        # Publish target position
                        target = Point()
                        target.x = detection.bbox.center.position.x
                        target.y = detection.bbox.center.position.y
                        self.target_pub.publish(target)
                        
                        if detected_class:
                            self.get_logger().info(f'Detected: {detected_class} (score={result.hypothesis.score:.2f})', throttle_duration_sec=2.0)
                        break
            if self.bottle_bbox:
                break
    
    def control_loop(self):
        """Main state machine control loop."""
        # Check for timeout
        elapsed = time.time() - self.start_time
        if elapsed > self.search_timeout:
            self.get_logger().error('Search timeout reached! Shutting down.')
            self.publish_status('TIMEOUT')
            self.timer.cancel()
            try:
                rclpy.shutdown()
            except RuntimeError:
                pass  # Already shutdown
            return
        
        # Check for detection loss (except in SEARCHING state)
        if self.state != State.SEARCHING:
            if self.last_detection_time is None or \
               (time.time() - self.last_detection_time) > self.detection_lost_timeout:
                self.get_logger().warn('Detection lost! Returning to SEARCHING.')
                self.transition_to(State.SEARCHING)
        
        # Execute state behavior
        if self.state == State.SEARCHING:
            self.state_searching()
        elif self.state == State.CENTERING:
            self.state_centering()
        elif self.state == State.APPROACHING:
            self.state_approaching()
        elif self.state == State.ARRIVED:
            self.state_arrived()
    
    def state_searching(self):
        """Scan for bottle with head movement."""
        self.publish_status('SEARCHING')
        
        # Check if target object found
        if self.bottle_bbox is not None:
            self.get_logger().info('TARGET OBJECT DETECTED!')
            self.transition_to(State.CENTERING)
            return
        
        # Continue head scanning
        if self.current_scan_index < len(self.head_scan_positions):
            pan_angle = self.head_scan_positions[self.current_scan_index]
            self.get_logger().info(f'Scanning head position {self.current_scan_index + 1}/{len(self.head_scan_positions)}: pan={pan_angle}°')
            
            # Send head position goal
            goal = HeadPosition.Goal()
            goal.pan_degrees = pan_angle
            goal.tilt_degrees = self.head_tilt_angle
            goal.smooth = True
            
            self.head_client.send_goal_async(goal)
            
            # Wait for next scan position
            time.sleep(self.head_scan_delay)
            self.current_scan_index += 1
        else:
            # Completed full head scan, rotate robot and reset head
            self.get_logger().info('Full head scan complete. Centering head and rotating robot 30°...')
            
            # First, center the head
            goal = HeadPosition.Goal()
            goal.pan_degrees = 0.0
            goal.tilt_degrees = self.head_tilt_angle
            goal.smooth = True
            self.head_client.send_goal_async(goal)
            time.sleep(1.0)  # Wait for head to center
            
            # Then rotate robot
            self.current_scan_index = 0
            goal = Rotate.Goal()
            goal.angle_degrees = 30.0
            goal.speed = 40.0
            goal.step_size_deg = 5.0
            goal.use_imu = True
            self.rotate_client.send_goal_async(goal)
            time.sleep(2.0)  # Wait for rotation
    
    def state_centering(self):
        """Center bottle in camera frame."""
        self.publish_status('CENTERING')
        
        if self.bottle_bbox is None:
            return
        
        bottle_x = self.bottle_bbox.center.position.x
        left_threshold = self.image_width * (0.5 - self.center_tolerance)
        right_threshold = self.image_width * (0.5 + self.center_tolerance)
        
        if bottle_x < left_threshold:
            # Bottle is left, rotate left
            self.get_logger().info(f'Bottle at x={bottle_x:.0f} (LEFT). Rotating {-self.rotation_step}°')
            goal = Rotate.Goal()
            goal.angle_degrees = -self.rotation_step
            goal.speed = 40.0
            goal.step_size_deg = 5.0
            goal.use_imu = True
            self.rotate_client.send_goal_async(goal)
            time.sleep(1.5)
        elif bottle_x > right_threshold:
            # Bottle is right, rotate right
            self.get_logger().info(f'Bottle at x={bottle_x:.0f} (RIGHT). Rotating {self.rotation_step}°')
            goal = Rotate.Goal()
            goal.angle_degrees = self.rotation_step
            goal.speed = 40.0
            goal.step_size_deg = 5.0
            goal.use_imu = True
            self.rotate_client.send_goal_async(goal)
            time.sleep(1.5)
        else:
            # Bottle is centered
            self.get_logger().info(f'Bottle CENTERED at x={bottle_x:.0f}')
            self.transition_to(State.APPROACHING)
    
    def state_approaching(self):
        """Approach bottle using bbox size as distance proxy."""
        self.publish_status('APPROACHING')
        
        if self.bottle_bbox is None:
            return
        
        bottle_width = self.bottle_bbox.size_x
        
        # Check if arrived
        if bottle_width >= self.goal_bottle_width:
            self.get_logger().info(f'Bottle close enough! width={bottle_width:.0f}px')
            self.transition_to(State.ARRIVED)
            return
        
        # Check centering while approaching
        bottle_x = self.bottle_bbox.center.position.x
        left_threshold = self.image_width * (0.5 - self.center_tolerance)
        right_threshold = self.image_width * (0.5 + self.center_tolerance)
        
        if bottle_x < left_threshold or bottle_x > right_threshold:
            # Lost centering, go back to centering state
            self.get_logger().info('Lost centering during approach')
            self.transition_to(State.CENTERING)
            return
        
        # Move forward
        distance_cm = self.approach_distance * 100.0  # Convert meters to cm
        self.get_logger().info(f'Moving forward {distance_cm:.0f}cm (bottle width={bottle_width:.0f}px)')
        goal = LinearMove.Goal()
        goal.distance_cm = distance_cm
        goal.speed = 40.0
        goal.step_size_cm = 2.0
        self.linear_client.send_goal_async(goal)
        time.sleep(2.0)  # Wait for movement
    
    def state_arrived(self):
        """Mission complete."""
        self.publish_status('ARRIVED')
        self.get_logger().info('🎉 MISSION COMPLETE! Bottle reached!')
        
        # Celebrate with head wiggle (optional)
        for _ in range(3):
            goal = HeadPosition.Goal()
            goal.pan_degrees = 30.0
            goal.tilt_degrees = 0.0
            goal.smooth = True
            self.head_client.send_goal_async(goal)
            time.sleep(0.5)
            
            goal.pan_degrees = -30.0
            goal.tilt_degrees = 0.0
            goal.smooth = True
            self.head_client.send_goal_async(goal)
            time.sleep(0.5)
        
        # Center head
        goal = HeadPosition.Goal()
        goal.pan_degrees = 0.0
        goal.tilt_degrees = 0.0
        goal.smooth = True
        self.head_client.send_goal_async(goal)
        
        # Stop the node
        rclpy.shutdown()
    
    def transition_to(self, new_state: State):
        """Transition to new state."""
        self.get_logger().info(f'State transition: {self.state.name} → {new_state.name}')
        self.state = new_state
        
        # Reset scan index when entering SEARCHING
        if new_state == State.SEARCHING:
            self.current_scan_index = 0
    
    def publish_status(self, status: str):
        """Publish current status."""
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    
    node = BottleSeeker()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
