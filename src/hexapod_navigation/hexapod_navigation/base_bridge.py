#!/usr/bin/env python3
"""
Base Bridge Node - Shared logic for LLM-based hexapod control.
Implements the SayCan state machine; subclasses provide _call_llm().
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from enum import Enum
import json
import os
import time
from typing import Optional, Dict, Any

# ROS2 messages
from sensor_msgs.msg import CompressedImage
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String
from raspclaws_interfaces.action import HeadPosition, Rotate, LinearMove, ArcMove

from PIL import Image as PILImage


class State(Enum):
    IDLE = 1
    SENSING = 2
    REASONING = 3
    ACTING = 4
    EVALUATING = 5


class BaseBridge(Node):
    """
    Abstract base class for LLM bridge nodes.
    Subclasses must implement _call_llm(prompt, image_pil) -> dict.
    Subclasses must call _start() at the end of their __init__.
    """

    def __init__(self, node_name: str):
        super().__init__(node_name)

        # Common parameters
        self.declare_parameter('model_name', '')
        self.declare_parameter('timeout', 600.0)
        self.declare_parameter('image_topic', '/raspclaws/camera/image_raw/compressed')
        self.declare_parameter('detection_topic', '/hexapod/detections')
        self.declare_parameter('control_loop_hz', 1.0)
        self.declare_parameter('max_retries', 3)
        self.declare_parameter('retry_delay', 2.0)
        self.declare_parameter('use_yolo', False)
        self.declare_parameter('max_history_length', 8)

        self.model_name = self.get_parameter('model_name').value
        self.timeout = self.get_parameter('timeout').value
        self.image_topic = self.get_parameter('image_topic').value
        self.detection_topic = self.get_parameter('detection_topic').value
        self.control_loop_hz = self.get_parameter('control_loop_hz').value
        self.max_retries = self.get_parameter('max_retries').value
        self.retry_delay = self.get_parameter('retry_delay').value
        self.use_yolo = self.get_parameter('use_yolo').value
        self.max_history_length = self.get_parameter('max_history_length').value

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

        # Last LLM response
        self.last_llm_response: Optional[Dict[str, Any]] = None

        # Conversation history
        self.conversation_history = []

        # Action queue
        self.action_queue = []
        self.current_plan_goal = None
        self.current_plan_type = None

        # Subscribers
        self.image_sub = self.create_subscription(
            CompressedImage,
            self.image_topic,
            self.image_callback,
            10
        )
        if self.use_yolo:
            self.detection_sub = self.create_subscription(
                Detection2DArray,
                self.detection_topic,
                self.detection_callback,
                10
            )
            self.get_logger().info(f'YOLO mode enabled: Subscribing to {self.detection_topic}')
        else:
            self.detection_sub = None
            self.get_logger().info('Pure vision mode: YOLO disabled')

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

        self.get_logger().info('Waiting for action servers...')
        self.head_client.wait_for_server(timeout_sec=10.0)
        self.rotate_client.wait_for_server(timeout_sec=10.0)
        self.linear_client.wait_for_server(timeout_sec=10.0)
        self.arc_client.wait_for_server(timeout_sec=10.0)
        self.get_logger().info('All action servers ready!')

        # Timer is NOT started here - subclass must call _start() after API setup

    def _start(self):
        """Start the control loop. Call this at the end of the subclass __init__."""
        self.timer = self.create_timer(1.0 / self.control_loop_hz, self.control_loop)
        self.get_logger().info(f'{self.get_name()} initialized. Waiting for goal on /hexapod/goal')

    # -------------------------------------------------------------------------
    # Abstract method - must be implemented by subclass
    # -------------------------------------------------------------------------

    def _call_llm(self, prompt: str, image_pil: PILImage.Image) -> dict:
        """
        Call the LLM API and return a parsed JSON dict.
        Raise an exception on failure (retry logic is in state_reasoning).
        """
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # ROS2 callbacks
    # -------------------------------------------------------------------------

    def image_callback(self, msg: CompressedImage):
        self.latest_image = msg

    def detection_callback(self, msg: Detection2DArray):
        self.latest_detections = msg

    def goal_callback(self, msg: String):
        """Receive new goal; supports interrupting ongoing execution."""
        new_goal = msg.data

        if self.current_goal is not None and self.current_goal != new_goal:
            self.get_logger().warn('GOAL CHANGED during execution!')
            self.get_logger().info(f'  Old: "{self.current_goal}"')
            self.get_logger().info(f'  New: "{new_goal}"')
            self.clear_action_queue()
            if self.action_in_progress:
                self.get_logger().warn('Interrupting current action (goal changed)')
                self.action_in_progress = False
            self.state = State.IDLE

        if self.state != State.IDLE:
            self.get_logger().warn(f'Goal received while in state {self.state.name}, ignoring')
            return

        self.current_goal = new_goal
        self.start_time = time.time()
        self.get_logger().info(f'New goal received: "{self.current_goal}"')
        self.transition_to(State.SENSING)

    # -------------------------------------------------------------------------
    # State machine
    # -------------------------------------------------------------------------

    def control_loop(self):
        if self.state != State.IDLE and self.start_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed > self.timeout:
                self.get_logger().error('Timeout reached!')
                self.publish_status('TIMEOUT')
                self.transition_to(State.IDLE)
                return

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
        self.publish_status('IDLE')

    def state_sensing(self):
        self.publish_status('SENSING')
        if self.latest_image is None:
            self.get_logger().warn('No image received yet', throttle_duration_sec=5.0)
            return
        if self.use_yolo and self.latest_detections is None:
            self.get_logger().warn('No detections received yet', throttle_duration_sec=5.0)
            return
        self.get_logger().info('Perception data collected')
        self.transition_to(State.REASONING)

    def state_reasoning(self):
        """Build prompt, call LLM via _call_llm(), process response."""
        self.publish_status('REASONING')

        image_pil = self._compressed_to_pil(self.latest_image)
        history_section = self._format_history_for_prompt()

        if self.use_yolo:
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
            prompt = f"""
{history_section}

**Current Goal**: {self.current_goal}

**Image Dimensions**: {self.image_width} x {self.image_height} pixels

Analyze the camera image directly and decide what action to take next.
Look for objects, obstacles, spatial relationships, and anything relevant to the goal.
"""

        for attempt in range(self.max_retries):
            try:
                self.get_logger().info(f'Calling LLM API (attempt {attempt + 1}/{self.max_retries})...')

                response_json = self._call_llm(prompt, image_pil)
                self.last_llm_response = response_json

                self.get_logger().info('Complete LLM response JSON:')
                self.get_logger().info(json.dumps(response_json, indent=2))

                reasoning_msg = String()
                reasoning_msg.data = json.dumps(response_json, indent=2)
                self.reasoning_pub.publish(reasoning_msg)

                self.get_logger().info('LLM reasoning:')
                self.get_logger().info(f"  Observation: {response_json['reasoning']['observation']}")
                self.get_logger().info(f"  Goal status: {response_json['reasoning']['goal_status']}")
                self.get_logger().info(f"  Safety: {response_json['reasoning']['safety_rating']}")
                self.get_logger().info(f"  Explanation: {response_json['explanation']}")

                if 'semantic_memory' in response_json:
                    mem = response_json['semantic_memory']
                    self.get_logger().info('Semantic Memory:')
                    self.get_logger().info(f"  Spatial: {mem.get('spatial_knowledge', 'N/A')}")
                    self.get_logger().info(f"  Insights: {mem.get('learned_insights', 'N/A')}")
                    self.get_logger().info(f"  Environment: {mem.get('environment_notes', 'N/A')}")

                if 'plan' in response_json:
                    plan = response_json['plan']
                    self.get_logger().info(
                        f"Plan ({plan.get('plan_type', 'unknown')}, "
                        f"confidence: {plan.get('plan_confidence', 0.0):.2f}):"
                    )
                    for i, action in enumerate(plan.get('actions', []), 1):
                        self.get_logger().info(f'  [{i}] {action["type"]}: {action.get("description", "N/A")}')

                if 'plan' in response_json and 'actions' in response_json['plan']:
                    plan = response_json['plan']
                    self._enqueue_actions(
                        actions=plan['actions'],
                        goal=self.current_goal,
                        plan_type=plan.get('plan_type', 'unknown')
                    )
                    action_data = plan['actions'][0] if plan['actions'] else {'type': 'wait', 'parameters': {}}
                else:
                    action_data = response_json.get('action', {'type': 'wait', 'parameters': {}})
                    self._enqueue_actions(
                        actions=[action_data],
                        goal=self.current_goal,
                        plan_type='new'
                    )

                semantic_memory = response_json.get('semantic_memory', {
                    'spatial_knowledge': response_json['reasoning']['observation'][:100],
                    'learned_insights': 'No insights recorded',
                    'environment_notes': 'No environment notes'
                })
                self._add_to_history(
                    goal=self.current_goal,
                    semantic_memory=semantic_memory,
                    action=action_data,
                    outcome="Pending execution"
                )

                self.transition_to(State.ACTING)
                return

            except Exception as e:
                self.get_logger().error(f'LLM API error (attempt {attempt + 1}): {e}')
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    self.get_logger().error('Max retries reached, aborting')
                    self.transition_to(State.IDLE)

    def state_acting(self):
        self.publish_status('ACTING')
        if self.action_in_progress:
            return
        if self._execute_next_action():
            return
        self.get_logger().info('All queued actions complete')
        self.transition_to(State.EVALUATING)

    def state_evaluating(self):
        self.publish_status('EVALUATING')
        if self.last_llm_response is None:
            self.transition_to(State.IDLE)
            return

        goal_status = self.last_llm_response['reasoning']['goal_status']
        if 'achieved' in goal_status.lower():
            self.get_logger().info('Goal achieved!')
            self.publish_status('GOAL_ACHIEVED')
            self.transition_to(State.IDLE)
        else:
            self.get_logger().info('Goal not yet achieved, continuing...')
            self.transition_to(State.SENSING)

    # -------------------------------------------------------------------------
    # Action execution
    # -------------------------------------------------------------------------

    def _on_goal_accepted(self, future, completion_callback):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected!')
            self.action_in_progress = False
            self.transition_to(State.IDLE)
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(completion_callback)

    def _on_action_complete(self, future):
        result = future.result()
        self.action_in_progress = False
        self.last_action_time = time.time()
        success = (result.status == 4)  # GoalStatus.STATUS_SUCCEEDED
        self.get_logger().info(f'Action complete (success={success})')
        if self.conversation_history:
            outcome = "Action completed successfully" if success else "Action failed"
            self.conversation_history[-1]['outcome'] = outcome

    def _execute_next_action(self) -> bool:
        if not self.action_queue:
            return False
        if self.action_in_progress:
            return False

        next_action = self.action_queue.pop(0)
        action_type = next_action['type']
        params = next_action.get('parameters', {})
        description = next_action.get('description', 'N/A')

        self.get_logger().info(f'Executing queued action: {action_type} ({description})')
        self.get_logger().info(f'  Remaining in queue: {len(self.action_queue)}')
        return self._execute_single_action(action_type, params)

    def _execute_single_action(self, action_type: str, params: Dict[str, Any]) -> bool:
        if action_type == 'wait':
            self.get_logger().info('Waiting (no action)')
            self.action_in_progress = False
            return True

        elif action_type == 'linear_move':
            goal = LinearMove.Goal()
            goal.distance_cm = float(params.get('distance_cm', 10.0))
            goal.speed = float(params.get('speed', 40.0))
            goal.step_size_cm = 2.0
            self.action_in_progress = True
            future = self.linear_client.send_goal_async(goal)
            future.add_done_callback(lambda f: self._on_goal_accepted(f, self._on_action_complete))
            return True

        elif action_type == 'rotate':
            goal = Rotate.Goal()
            goal.angle_degrees = float(params.get('angle_degrees', 30.0))
            goal.speed = float(params.get('speed', 40.0))
            self.action_in_progress = True
            future = self.rotate_client.send_goal_async(goal)
            future.add_done_callback(lambda f: self._on_goal_accepted(f, self._on_action_complete))
            return True

        elif action_type == 'arc_move':
            goal = ArcMove.Goal()
            goal.distance_cm = float(params.get('distance_cm', 20.0))
            goal.arc_factor = float(params.get('arc_factor', 0.5))
            goal.speed = float(params.get('speed', 40.0))
            self.action_in_progress = True
            future = self.arc_client.send_goal_async(goal)
            future.add_done_callback(lambda f: self._on_goal_accepted(f, self._on_action_complete))
            return True

        elif action_type == 'head_position':
            goal = HeadPosition.Goal()
            goal.pan_degrees = float(params.get('pan_degrees', 0.0))
            goal.tilt_degrees = float(params.get('tilt_degrees', 0.0))
            self.action_in_progress = True
            future = self.head_client.send_goal_async(goal)
            future.add_done_callback(lambda f: self._on_goal_accepted(f, self._on_action_complete))
            return True

        else:
            self.get_logger().error(f'Unknown action type: {action_type}')
            return False

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def transition_to(self, new_state: State):
        if self.state != new_state:
            self.get_logger().info(f'State: {self.state.name} -> {new_state.name}')
            self.state = new_state

    def publish_status(self, status: str):
        msg = String()
        msg.data = f'{self.state.name}:{status}'
        self.status_pub.publish(msg)

    def _compressed_to_pil(self, msg: CompressedImage) -> PILImage.Image:
        import numpy as np
        import cv2
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        return PILImage.fromarray(rgb_image)

    def _detections_to_json(self, msg: Detection2DArray) -> list:
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

    # -------------------------------------------------------------------------
    # History management
    # -------------------------------------------------------------------------

    def _add_to_history(self, goal: str, semantic_memory: Dict[str, str],
                        action: Dict[str, Any], outcome: str):
        entry = {
            'timestamp': time.time(),
            'goal': goal,
            'semantic_memory': semantic_memory,
            'action': action,
            'outcome': outcome
        }
        self.conversation_history.append(entry)
        if len(self.conversation_history) > self.max_history_length:
            removed = self.conversation_history.pop(0)
            self.get_logger().debug(
                f'History trimmed: removed entry from {time.ctime(removed["timestamp"])}'
            )
        self.get_logger().info(
            f'History updated: {len(self.conversation_history)}/{self.max_history_length} entries'
        )

    def _format_history_for_prompt(self) -> str:
        if not self.conversation_history:
            return "**Conversation History**: (Empty - this is the first interaction)"

        history_str = f"**Conversation History** (Last {len(self.conversation_history)} exchanges):\n"
        for i, entry in enumerate(self.conversation_history, 1):
            timestamp_str = time.strftime('%H:%M:%S', time.localtime(entry['timestamp']))
            action_type = entry['action'].get('type', 'unknown')
            action_params = entry['action'].get('parameters', {})

            if 'semantic_memory' in entry:
                memory = entry['semantic_memory']
                history_str += f"\n[{i}] Time: {timestamp_str}\n"
                history_str += f"    Goal: \"{entry['goal']}\"\n"
                history_str += f"    -> Semantic Memory: {memory.get('spatial_knowledge', 'N/A')}\n"
                history_str += f"    -> Action: {action_type}({action_params})\n"
                history_str += f"    -> Outcome: {entry['outcome']}\n"
            else:
                history_str += f"\n[{i}] Time: {timestamp_str}\n"
                history_str += f"    Goal: \"{entry['goal']}\"\n"
                history_str += f"    -> Observation: {entry.get('observation', 'N/A')[:100]}...\n"
                history_str += f"    -> Action: {action_type}({action_params})\n"
                history_str += f"    -> Outcome: {entry['outcome']}\n"

        return history_str

    def clear_history(self):
        self.conversation_history.clear()
        self.get_logger().info('Conversation history cleared')

    # -------------------------------------------------------------------------
    # Action queue management
    # -------------------------------------------------------------------------

    def _enqueue_actions(self, actions: list, goal: str, plan_type: str):
        sorted_actions = sorted(actions, key=lambda x: x.get('priority', 999))
        self.action_queue = sorted_actions.copy()
        self.current_plan_goal = goal
        self.current_plan_type = plan_type
        self.get_logger().info(f'Action queue updated: {len(self.action_queue)} actions')
        for i, action in enumerate(self.action_queue, 1):
            self.get_logger().info(f'  [{i}] {action["type"]}: {action.get("description", "N/A")}')

    def clear_action_queue(self):
        cleared_count = len(self.action_queue)
        self.action_queue.clear()
        self.current_plan_goal = None
        self.current_plan_type = None
        if cleared_count > 0:
            self.get_logger().warn(f'Action queue cleared ({cleared_count} actions discarded)')
