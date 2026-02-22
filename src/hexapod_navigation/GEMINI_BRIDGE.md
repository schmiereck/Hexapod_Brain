# Gemini Bridge - LLM-Based Robot Control

## Overview
`gemini_bridge.py` implements Phase 3 of the Hexapod Brain project: **Embodied Reasoning with Gemini AI**.

Unlike the fixed state machine in `bottle_seeker.py`, the Gemini Bridge uses a Large Language Model (Google Gemini 2.0 Flash) to decide robot actions based on visual input, object detections, and user-defined goals.

## Architecture

### Core Concept: SayCan
The robot implements Google's SayCan principle:
1. **Say** (Goal): What is the user's objective? (e.g., "Find and approach the bottle")
2. **Can** (Skills): What actions can the robot physically perform? (linear_move, rotate, head_position)
3. **See** (Perception): What does the camera see? (objects, positions, environment)
4. **Choose**: LLM decides the best action based on Say + Can + See

### State Machine
```
IDLE → SENSING → REASONING → ACTING → EVALUATING → (loop or IDLE)
```

- **IDLE**: Waiting for goal on `/hexapod/goal` topic
- **SENSING**: Collect image + detections
- **REASONING**: Call Gemini API for decision
- **ACTING**: Execute chosen action via ROS2 Action
- **EVALUATING**: Check if goal achieved, repeat or finish

### Input/Output

**Inputs**:
- `/hexapod/goal` (std_msgs/String): User's high-level goal
- `/raspclaws/camera/image_raw/compressed`: Camera stream
- `/hexapod/detections`: YOLO object detections

**Outputs**:
- `/hexapod/navigation/status` (std_msgs/String): Current state
- `/hexapod/reasoning` (std_msgs/String): Gemini's reasoning (JSON)
- ROS2 Action Goals: `/raspclaws/linear_move`, `/raspclaws/rotate`, `/raspclaws/head_position`

## Prerequisites

### 1. Gemini API Setup
```bash
# On ubuntu1
cd ~/Hexapod_Brain
bash scripts/setup_gemini_ubuntu1.sh

# Set API key in ~/.bashrc
export GEMINI_API_KEY='your_api_key_here'
source ~/.bashrc

# Test API
python3 scripts/test_gemini_api.py
```

See `scripts/GEMINI_SETUP.md` for detailed instructions.

### 2. Dependencies
```bash
pip3 install google-generativeai pillow
```

### 3. ROS2 Workspace
```bash
cd ~/Hexapod_Brain
source setup_env.bash  # Sets ROS_DOMAIN_ID=1, sources workspace
```

## Usage

### Basic Launch
```bash
# Terminal 1: Start YOLO detector
ros2 launch hexapod_vision yolo_detector_tflite.launch.py

# Terminal 2: Start Gemini Bridge
ros2 launch hexapod_navigation gemini_bridge.launch.py

# Terminal 3: Send goal
ros2 topic pub --once /hexapod/goal std_msgs/String "data: 'Find and approach the bottle'"
```

### Monitor Status
```bash
# Watch status
ros2 topic echo /hexapod/navigation/status

# Watch Gemini's reasoning
ros2 topic echo /hexapod/reasoning
```

### Launch with Custom Parameters
```bash
# Use fallback model if robotics-er not available
ros2 launch hexapod_navigation gemini_bridge.launch.py \
  model_name:='gemini-1.5-pro' \
  timeout:=180.0 \
  control_loop_hz:=0.5
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gemini_api_key` | `''` | API key (or use GEMINI_API_KEY env var) |
| `model_name` | `gemini-robotics-er-1.5-preview` | Gemini model (robotics-specialized) |
| `timeout` | `120.0` | Max time for goal completion (seconds) |
| `image_topic` | `/raspclaws/camera/image_raw/compressed` | Camera topic |
| `detection_topic` | `/hexapod/detections` | YOLO detections topic |
| `control_loop_hz` | `1.0` | Loop frequency (1 Hz recommended for LLM) |
| `max_retries` | `3` | Retries for failed API calls |
| `retry_delay` | `2.0` | Delay between retries (seconds) |

## Example Session

```bash
# Terminal 1: Launch system
ros2 launch hexapod_navigation gemini_bridge.launch.py

# Terminal 2: Send goal
ros2 topic pub --once /hexapod/goal std_msgs/String "data: 'Find the bottle'"

# Expected behavior:
# 1. Robot scans environment (head movement or rotation)
# 2. Detects bottle
# 3. Centers bottle in view (rotation)
# 4. Approaches bottle (forward movement)
# 5. Stops when close enough
# 6. Returns to IDLE

# Terminal 3: Monitor reasoning
ros2 topic echo /hexapod/reasoning

# Example output:
# {
#   "reasoning": {
#     "observation": "Bottle detected at x=520 (right side). Off-center.",
#     "goal_status": "in progress - bottle found but not centered",
#     "affordance_check": "Rotation safe. Will center bottle.",
#     "safety_rating": "high"
#   },
#   "action": {
#     "type": "rotate",
#     "parameters": {"angle_degrees": 20.0, "speed": 40}
#   },
#   "explanation": "Rotate right to center bottle in view"
# }
```

## Comparison: bottle_seeker vs gemini_bridge

| Feature | bottle_seeker.py | gemini_bridge.py |
|---------|------------------|------------------|
| **Decision Logic** | Fixed state machine | LLM reasoning |
| **Flexibility** | Single task (find bottle) | Any goal via prompt |
| **Adaptability** | Hardcoded behavior | Learns from examples |
| **Speed** | 10 Hz loop | 1 Hz (LLM latency) |
| **Cost** | Free | ~$0.01 per decision |
| **Interpretability** | Code inspection | Explicit reasoning output |
| **Edge Cases** | Requires code changes | Can reason through novel situations |

## Troubleshooting

### "GEMINI_API_KEY not set"
```bash
export GEMINI_API_KEY='your_key_here'
# Or add to ~/.bashrc for persistence
```

### "No image received yet"
```bash
# Check camera stream
ros2 topic hz /raspclaws/camera/image_raw/compressed

# Enable camera (if paused)
ros2 service call /raspclaws/set_camera_pause std_srvs/srv/SetBool "{data: false}"
```

### "No detections received yet"
```bash
# Check detector is running
ros2 node list | grep yolo

# Check detections topic
ros2 topic hz /hexapod/detections

# Launch detector if needed
ros2 launch hexapod_vision yolo_detector_tflite.launch.py
```

### "Gemini API error: 403 Forbidden"
- Check API key is valid at https://makersuite.google.com/app/apikey
- Verify API is enabled in Google Cloud Console
- Check billing is enabled (Gemini API requires it)

### "Robot not moving / Action rejected"
```bash
# Check action servers are running
ros2 action list

# Expected:
# /raspclaws/linear_move
# /raspclaws/rotate
# /raspclaws/head_position

# Check raspclaws-1 services are running
ssh pi@192.168.2.126
systemctl status gui_server.service
systemctl status ros_server.service
```

## Cost Optimization

### During Development
- Use `control_loop_hz: 0.5` (slower loop = fewer API calls)
- Test with local mock responses before live API
- Use cheaper model (gemini-1.5-flash) for basic testing

### Production
- Cache repeated scenarios
- Implement local safety checks to avoid unnecessary API calls
- Use batch processing for multiple decisions
- Monitor costs at: https://console.cloud.google.com/billing

## Safety Features

1. **Low Safety Rating Warning**: Logs warning if Gemini rates action as "low" safety
2. **Timeout Protection**: Aborts after 120s (configurable)
3. **Parameter Validation**: Checks action parameters before execution
4. **Two-Stage Callbacks**: Waits for action completion (not just acceptance) to prevent "zappeln"
5. **Retry Logic**: Retries failed API calls (max 3 attempts)

## Next Steps

### Phase 3 Enhancements
- [ ] Add hypervisor camera (bird's eye view)
- [ ] Multi-step planning (3 actions ahead)
- [ ] Experience storage (learn from successes)
- [ ] Natural language interaction ("Why did you rotate?")

### Phase 4 (Future)
- Multi-task learning
- Goal embeddings
- Meta-learning from multiple tasks
- Sim-to-real transfer

## References
- System Instruction: `hexapod_navigation/gemini_prompts.py`
- Gemini API Docs: https://ai.google.dev/docs
- SayCan Paper: https://say-can.github.io/
