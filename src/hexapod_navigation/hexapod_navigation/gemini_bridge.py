#!/usr/bin/env python3
"""
Gemini Bridge Node - Phase 3 Embodied Reasoning
Uses Google Gemini API for LLM-based robot control with SayCan principle.
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from enum import Enum
import json
import base64
import io
import os
import time
from typing import Optional, Dict, Any

# ROS2 messages
from sensor_msgs.msg import CompressedImage
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String
from raspclaws_interfaces.action import HeadPosition, Rotate, LinearMove, ArcMove

# Gemini API
from google import genai
from PIL import Image as PILImage

# Local imports
from .gemini_prompts import SYSTEM_INSTRUCTION, RESPONSE_SCHEMA


class State(Enum):
    IDLE = 1
    SENSING = 2
    REASONING = 3
    ACTING = 4
    EVALUATING = 5


class GeminiBridge(Node):
    def __init__(self):
        super().__init__('gemini_bridge')
        
        # Parameters
        self.declare_parameter('gemini_api_key', '')
        self.declare_parameter('model_name', 'models/gemini-robotics-er-1.5-preview')
        self.declare_parameter('timeout', 120.0)
        self.declare_parameter('image_topic', '/raspclaws/camera/image_raw/compressed')
        self.declare_parameter('detection_topic', '/hexapod/detections')
        self.declare_parameter('control_loop_hz', 1.0)
        self.declare_parameter('max_retries', 3)
        self.declare_parameter('retry_delay', 2.0)
        self.declare_parameter('use_yolo', False)  # Phase 4: Pure vision mode by default
        self.declare_parameter('max_history_length', 8)  # Phase 5a: Conversation history
        
        # Get parameters
        api_key = self.get_parameter('gemini_api_key').value
        if not api_key:
            api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            self.get_logger().error('GEMINI_API_KEY not set! Set parameter or environment variable.')
            raise ValueError('GEMINI_API_KEY required')
        
        self.model_name = self.get_parameter('model_name').value
        self.timeout = self.get_parameter('timeout').value
        self.image_topic = self.get_parameter('image_topic').value
        self.detection_topic = self.get_parameter('detection_topic').value
        self.control_loop_hz = self.get_parameter('control_loop_hz').value
        self.max_retries = self.get_parameter('max_retries').value
        self.retry_delay = self.get_parameter('retry_delay').value
        self.use_yolo = self.get_parameter('use_yolo').value
        self.max_history_length = self.get_parameter('max_history_length').value
        
        # Configure Gemini API
        self.client = genai.Client(api_key=api_key)
        self.model_name_full = self.model_name
        self.generation_config = {
            "response_mime_type": "application/json",
            "response_schema": RESPONSE_SCHEMA,
            "system_instruction": SYSTEM_INSTRUCTION
        }
        mode_str = "YOLO+Gemini (Phase 3)" if self.use_yolo else "Pure Vision (Phase 4)"
        self.get_logger().info(f'Gemini client configured: {self.model_name} - Mode: {mode_str}')
        
        # State
        self.state = State.IDLE
        self.current_goal = None
        self.start_time = None
        self.action_in_progress = False
        self.last_action_time = 0.0
        
        # Perception data
        self.latest_image: Optional[CompressedImage] = None
        self.latest_detections: Optional[Detection2DArray] = None
        self.image_width = 640
        self.image_height = 480
        
        # Gemini response
        self.last_gemini_response: Optional[Dict[str, Any]] = None
        
        # Phase 5a: Conversation history
        self.conversation_history = []  # List of dicts: {timestamp, goal, observation, action, outcome}
        
        # Subscribers
        self.image_sub = self.create_subscription(
            CompressedImage,
            self.image_topic,
            self.image_callback,
            10
        )
        # Conditional YOLO subscription (Phase 4)
        if self.use_yolo:
            self.detection_sub = self.create_subscription(
                Detection2DArray,
                self.detection_topic,
                self.detection_callback,
                10
            )
            self.get_logger().info(f'📊 YOLO mode enabled: Subscribing to {self.detection_topic}')
        else:
            self.detection_sub = None
            self.get_logger().info('👁️ Pure vision mode: YOLO disabled')
        
        self.goal_sub = self.create_subscription(
            String,
            '/hexapod/goal',
            self.goal_callback,
            10
        )
        
        # Publishers
        self.status_pub = self.create_publisher(String, '/hexapod/navigation/status', 10)
        self.reasoning_pub = self.create_publisher(String, '/hexapod/reasoning', 10)
        
        # Action clients
        self.head_client = ActionClient(self, HeadPosition, '/raspclaws/head_position')
        self.rotate_client = ActionClient(self, Rotate, '/raspclaws/rotate')
        self.linear_client = ActionClient(self, LinearMove, '/raspclaws/linear_move')
        self.arc_client = ActionClient(self, ArcMove, '/raspclaws/arc_move')
        
        # Wait for action servers
        self.get_logger().info('Waiting for action servers...')
        self.head_client.wait_for_server(timeout_sec=10.0)
        self.rotate_client.wait_for_server(timeout_sec=10.0)
        self.linear_client.wait_for_server(timeout_sec=10.0)
        self.arc_client.wait_for_server(timeout_sec=10.0)
        self.get_logger().info('All action servers ready!')
        
        # Control loop
        self.timer = self.create_timer(1.0 / self.control_loop_hz, self.control_loop)
        
        self.get_logger().info('Gemini Bridge initialized. Waiting for goal on /hexapod/goal')
    
    def image_callback(self, msg: CompressedImage):
        """Store latest image."""
        self.latest_image = msg
    
    def detection_callback(self, msg: Detection2DArray):
        """Store latest detections."""
        self.latest_detections = msg
    
    def goal_callback(self, msg: String):
        """Receive new goal from user."""
        new_goal = msg.data
        
        # Phase 5a: Detect goal changes (even during execution)
        if self.current_goal is not None and self.current_goal != new_goal:
            self.get_logger().warn(f'⚠️ GOAL CHANGED during execution!')
            self.get_logger().info(f'   Old: "{self.current_goal}"')
            self.get_logger().info(f'   New: "{new_goal}"')
            # History preserved for spatial memory, but plan is obsolete
            
        if self.state != State.IDLE:
            self.get_logger().warn(f'Goal received while in state {self.state.name}, ignoring')
            return
        
        self.current_goal = new_goal
        self.start_time = time.time()
        self.get_logger().info(f'🎯 New goal received: "{self.current_goal}"')
        self.transition_to(State.SENSING)
    
    def control_loop(self):
        """Main state machine control loop."""
        # Timeout check
        if self.state != State.IDLE and self.start_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed > self.timeout:
                self.get_logger().error('⏱️ Timeout reached!')
                self.publish_status('TIMEOUT')
                self.transition_to(State.IDLE)
                return
        
        # Execute state behavior
        if self.state == State.IDLE:
            self.state_idle()
        elif self.state == State.SENSING:
            self.state_sensing()
        elif self.state == State.REASONING:
            self.state_reasoning()
        elif self.state == State.ACTING:
            self.state_acting()
        elif self.state == State.EVALUATING:
            self.state_evaluating()
    
    def state_idle(self):
        """Waiting for goal."""
        self.publish_status('IDLE')
    
    def state_sensing(self):
        """Collect perception data."""
        self.publish_status('SENSING')
        
        # Check if we have image (required in all modes)
        if self.latest_image is None:
            self.get_logger().warn('No image received yet', throttle_duration_sec=5.0)
            return
        
        # In YOLO mode, also wait for detections
        if self.use_yolo and self.latest_detections is None:
            self.get_logger().warn('No detections received yet', throttle_duration_sec=5.0)
            return
        
        self.get_logger().info('✅ Perception data collected')
        self.transition_to(State.REASONING)
    
    def state_reasoning(self):
        """Call Gemini API for decision."""
        self.publish_status('REASONING')
        
        # Prepare input
        image_pil = self._compressed_to_pil(self.latest_image)
        
        # Phase 5a: Format conversation history
        history_section = self._format_history_for_prompt()
        
        # Build prompt (conditional based on use_yolo)
        if self.use_yolo:
            # Phase 3: YOLO + Gemini mode
            detections_json = self._detections_to_json(self.latest_detections)
            prompt = f"""
{history_section}

**Current Goal**: {self.current_goal}

**Image Dimensions**: {self.image_width} x {self.image_height} pixels

**Detected Objects**:
{json.dumps(detections_json, indent=2)}

Based on this input, what action should I take next?
"""
        else:
            # Phase 4: Pure vision mode (no YOLO)
            prompt = f"""
{history_section}

**Current Goal**: {self.current_goal}

**Image Dimensions**: {self.image_width} x {self.image_height} pixels

Analyze the camera image directly and decide what action to take next.
Look for objects, obstacles, spatial relationships, and anything relevant to the goal.
"""
        
        # Call Gemini with retry logic
        for attempt in range(self.max_retries):
            try:
                self.get_logger().info(f'🤖 Calling Gemini API (attempt {attempt + 1}/{self.max_retries})...')
                
                # New API: client.models.generate_content
                response = self.client.models.generate_content(
                    model=self.model_name_full,
                    contents=[prompt, image_pil],
                    config=self.generation_config
                )
                
                # Parse JSON response
                response_json = json.loads(response.text)
                self.last_gemini_response = response_json
                
                # LOG COMPLETE JSON FOR DEBUGGING
                self.get_logger().info('📋 COMPLETE Gemini response JSON:')
                self.get_logger().info(json.dumps(response_json, indent=2))
                
                # Publish reasoning
                reasoning_msg = String()
                reasoning_msg.data = json.dumps(response_json, indent=2)
                self.reasoning_pub.publish(reasoning_msg)
                
                # Log reasoning (summary)
                self.get_logger().info(f'💡 Gemini reasoning:')
                self.get_logger().info(f"   Observation: {response_json['reasoning']['observation']}")
                self.get_logger().info(f"   Goal status: {response_json['reasoning']['goal_status']}")
                self.get_logger().info(f"   Safety: {response_json['reasoning']['safety_rating']}")
                self.get_logger().info(f"   Action: {response_json['action']['type']}")
                self.get_logger().info(f"   Explanation: {response_json['explanation']}")
                
                # Log semantic memory
                if 'semantic_memory' in response_json:
                    mem = response_json['semantic_memory']
                    self.get_logger().info(f'🧠 Semantic Memory:')
                    self.get_logger().info(f"   Spatial: {mem.get('spatial_knowledge', 'N/A')}")
                    self.get_logger().info(f"   Insights: {mem.get('learned_insights', 'N/A')}")
                    self.get_logger().info(f"   Environment: {mem.get('environment_notes', 'N/A')}")
                
                # Phase 5a: Add to conversation history with semantic memory
                semantic_memory = response_json.get('semantic_memory', {
                    'spatial_knowledge': response_json['reasoning']['observation'][:100],
                    'learned_insights': 'No insights recorded',
                    'environment_notes': 'No environment notes'
                })
                action_data = response_json['action']
                self._add_to_history(
                    goal=self.current_goal,
                    semantic_memory=semantic_memory,
                    action=action_data,
                    outcome="Pending execution"  # Will be updated in state_evaluating
                )
                
                self.transition_to(State.ACTING)
                return
                
            except Exception as e:
                self.get_logger().error(f'Gemini API error (attempt {attempt + 1}): {e}')
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    self.get_logger().error('❌ Max retries reached, aborting')
                    self.transition_to(State.IDLE)
    
    def state_acting(self):
        """Execute action from Gemini."""
        self.publish_status('ACTING')
        
        if self.action_in_progress:
            return  # Wait for current action to complete
        
        if self.last_gemini_response is None:
            self.get_logger().error('No Gemini response to execute!')
            self.transition_to(State.IDLE)
            return
        
        # Extract action
        action_data = self.last_gemini_response['action']
        action_type = action_data['type']
        params = action_data.get('parameters', {})  # Use .get() with default
        safety_rating = self.last_gemini_response['reasoning']['safety_rating']
        
        # Safety check
        if safety_rating == 'low':
            self.get_logger().warn('⚠️ Low safety rating, but executing action (override if needed)')
            # TODO: Add user confirmation or safety validation
        
        # Log action with parameters
        self.get_logger().info(f'🚀 Executing: {action_type} with params={params}')
        
        if action_type == 'wait':
            # No action needed
            self.get_logger().info('⏸️ Waiting (no action)')
            self.transition_to(State.EVALUATING)
            
        elif action_type == 'linear_move':
            goal = LinearMove.Goal()
            goal.distance_cm = float(params.get('distance_cm', 10.0))
            goal.speed = float(params.get('speed', 40.0))
            goal.step_size_cm = 2.0
            self.action_in_progress = True
            future = self.linear_client.send_goal_async(goal)
            future.add_done_callback(lambda f: self._on_goal_accepted(f, self._on_action_complete))
            
        elif action_type == 'rotate':
            goal = Rotate.Goal()
            goal.angle_degrees = float(params.get('angle_degrees', 30.0))
            goal.speed = float(params.get('speed', 40.0))
            goal.step_size_deg = 5.0
            goal.use_imu = True
            self.action_in_progress = True
            future = self.rotate_client.send_goal_async(goal)
            future.add_done_callback(lambda f: self._on_goal_accepted(f, self._on_action_complete))
            
        elif action_type == 'head_position':
            goal = HeadPosition.Goal()
            goal.pan_degrees = float(params.get('pan_degrees', 0.0))
            goal.tilt_degrees = float(params.get('tilt_degrees', -75.0))
            goal.smooth = True
            self.action_in_progress = True
            future = self.head_client.send_goal_async(goal)
            future.add_done_callback(lambda f: self._on_goal_accepted(f, self._on_action_complete))
            
        elif action_type == 'arc_move':
            goal = ArcMove.Goal()
            goal.radius_cm = float(params.get('radius_cm', 50.0))
            goal.angle_degrees = float(params.get('angle_degrees', 45.0))
            goal.speed = float(params.get('speed', 40.0))
            self.action_in_progress = True
            future = self.arc_client.send_goal_async(goal)
            future.add_done_callback(lambda f: self._on_goal_accepted(f, self._on_action_complete))
            
        else:
            self.get_logger().error(f'Unknown action type: {action_type}')
            self.transition_to(State.IDLE)
    
    def state_evaluating(self):
        """Check if goal is achieved."""
        self.publish_status('EVALUATING')
        
        if self.last_gemini_response is None:
            self.transition_to(State.IDLE)
            return
        
        goal_status = self.last_gemini_response['reasoning']['goal_status']
        
        if 'achieved' in goal_status.lower():
            self.get_logger().info('🎉 Goal achieved!')
            self.publish_status('GOAL_ACHIEVED')
            self.transition_to(State.IDLE)
        else:
            # Continue cycle: sense again
            self.get_logger().info('🔄 Goal not yet achieved, continuing...')
            self.transition_to(State.SENSING)
    
    def _on_goal_accepted(self, future, completion_callback):
        """Callback when goal is accepted."""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('❌ Goal rejected!')
            self.action_in_progress = False
            self.transition_to(State.IDLE)
            return
        
        self.get_logger().debug('Goal accepted, waiting for result...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(completion_callback)
    
    def _on_action_complete(self, future):
        """Callback when action completes."""
        result = future.result()
        self.action_in_progress = False
        self.last_action_time = time.time()
        success = (result.status == 4)  # GoalStatus.STATUS_SUCCEEDED
        self.get_logger().info(f'✅ Action complete (success={success})')
        
        # Phase 5a: Update last history entry outcome
        if self.conversation_history:
            outcome = "Action completed successfully" if success else "Action failed"
            self.conversation_history[-1]['outcome'] = outcome
        
        # Move to evaluation
        self.transition_to(State.EVALUATING)
    
    def _compressed_to_pil(self, msg: CompressedImage) -> PILImage.Image:
        """Convert CompressedImage to PIL Image."""
        import numpy as np
        np_arr = np.frombuffer(msg.data, np.uint8)
        import cv2
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb_image)
        return pil_image
    
    def _detections_to_json(self, msg: Detection2DArray) -> list:
        """Convert Detection2DArray to JSON-serializable list."""
        detections = []
        for detection in msg.detections:
            for result in detection.results:
                detections.append({
                    'class_id': result.hypothesis.class_id,
                    'score': float(result.hypothesis.score),
                    'bbox': {
                        'center': {
                            'x': float(detection.bbox.center.position.x),
                            'y': float(detection.bbox.center.position.y)
                        },
                        'size_x': float(detection.bbox.size_x),
                        'size_y': float(detection.bbox.size_y)
                    }
                })
        return detections
    
    def transition_to(self, new_state: State):
        """Transition to new state."""
        if self.state != new_state:
            self.get_logger().info(f'State: {self.state.name} → {new_state.name}')
            self.state = new_state
    
    def publish_status(self, status: str):
        """Publish current status."""
        msg = String()
        msg.data = f'{self.state.name}:{status}'
        self.status_pub.publish(msg)
    
    # Phase 5a: History management methods
    
    def _add_to_history(self, goal: str, semantic_memory: Dict[str, str], action: Dict[str, Any], outcome: str):
        """Add entry to conversation history with semantic memory and maintain max length."""
        entry = {
            'timestamp': time.time(),
            'goal': goal,
            'semantic_memory': semantic_memory,  # Store semantic memory instead of raw observation
            'action': action,
            'outcome': outcome
        }
        self.conversation_history.append(entry)
        
        # Trim if exceeds max length
        if len(self.conversation_history) > self.max_history_length:
            removed = self.conversation_history.pop(0)  # Remove oldest
            self.get_logger().debug(f'History trimmed: removed entry from {time.ctime(removed["timestamp"])}')
        
        self.get_logger().info(f'📝 History updated: {len(self.conversation_history)}/{self.max_history_length} entries')
    
    def _format_history_for_prompt(self) -> str:
        """Format conversation history as string for Gemini prompt."""
        if not self.conversation_history:
            return "**Conversation History**: (Empty - this is the first interaction)"
        
        history_str = f"**Conversation History** (Last {len(self.conversation_history)} exchanges):\n"
        for i, entry in enumerate(self.conversation_history, 1):
            timestamp_str = time.strftime('%H:%M:%S', time.localtime(entry['timestamp']))
            action_type = entry['action'].get('type', 'unknown')
            action_params = entry['action'].get('parameters', {})
            
            # Use semantic memory if available, fallback to observation for backward compatibility
            if 'semantic_memory' in entry:
                memory = entry['semantic_memory']
                history_str += f"\n[{i}] Time: {timestamp_str}\n"
                history_str += f"    Goal: \"{entry['goal']}\"\n"
                history_str += f"    → Semantic Memory: {memory.get('spatial_knowledge', 'N/A')}\n"
                history_str += f"    → Action: {action_type}({action_params})\n"
                history_str += f"    → Outcome: {entry['outcome']}\n"
            else:
                # Backward compatibility for old format
                history_str += f"\n[{i}] Time: {timestamp_str}\n"
                history_str += f"    Goal: \"{entry['goal']}\"\n"
                history_str += f"    → Observation: {entry.get('observation', 'N/A')[:100]}...\n"
                history_str += f"    → Action: {action_type}({action_params})\n"
                history_str += f"    → Outcome: {entry['outcome']}\n"
        
        return history_str
    
    def clear_history(self):
        """Clear conversation history (e.g., on explicit user reset)."""
        self.conversation_history.clear()
        self.get_logger().info('🗑️ Conversation history cleared')


def main(args=None):
    rclpy.init(args=args)
    
    node = GeminiBridge()
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
