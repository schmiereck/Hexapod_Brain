# TODO.md - Hexapod Brain Project

## ✅ Completed (Phase 1)
*   [x] Basic `hexapod_vision` package structure created
*   [x] `yolo_detector.py` implemented (PyTorch version - **has issues on Pi**)
*   [x] `yolo_detector_tflite.py` implemented (TFLite version - **recommended**)
*   [x] `environment.yml` for Micromamba created
*   [x] TFLite runtime installed on ubuntu1
*   [x] Package builds successfully on ubuntu1
*   [x] ROS2 services verified on raspclaws-1 (gui_server, ros_server)
*   [x] ROS2 topics and actions verified on ubuntu1
*   [x] **YOLOv8n TFLite model downloaded and uploaded to ubuntu1**
*   [x] **TFLite detector successfully starts and loads model**

## ✅ Completed (Phase 2 - Navigation)
*   [x] Created hexapod_navigation package
*   [x] Implemented bottle_seeker.py node with state machine (SEARCHING, CENTERING, APPROACHING, ARRIVED)
*   [x] Subscribed to `/hexapod/detections` for object detection
*   [x] Implemented ROS2 Action clients for movement (linear_move, rotate, head_position)
*   [x] Fixed "zappeln" bug with two-stage callback pattern (goal acceptance + result)
*   [x] Added stabilization delays and action_in_progress flags
*   [x] Tested successfully with live robot - autonomous bottle finding and approach ✅
*   [x] Created setup_env.bash for easy workspace setup

**Lessons Learned**:
- ROS2 Actions need two-stage callbacks (acceptance ≠ completion)
- State machines need careful action tracking to prevent command spam
- Stabilization delays critical for smooth movement (0.25s between actions)

## ✅ Completed (Phase 3 - Gemini Robotics ER)
1.  **Create hexapod_navigation Package**:
    - Navigation node that subscribes to `/hexapod/detections`
    - Implements behavior logic (e.g., "approach detected person")
    - Calls ROS2 Actions on raspclaws-1 (`/raspclaws/linear_move`, `/raspclaws/rotate`)

2.  **Implement Simple Behaviors**:
    - Object tracking (rotate camera to keep object centered)
    - Approach behavior (move towards detected object)
    - Collision avoidance (stop if too close)

3.  **Integration Testing**:
    - End-to-end test: Camera → Detection → Navigation → Movement

### Goal
Replace fixed state machine logic (bottle_seeker.py) with **LLM-based reasoning** using Google Gemini 2.0 Flash. The robot will use multimodal AI to decide actions based on:
- Visual input (camera images)
- Object detections (YOLO)
- Available skills (ROS2 Actions)
- User-defined goals

### Architecture: Embodied Bridge
New node `gemini_bridge.py` that implements:
1. **SayCan Principle**: "Say" (goal) + "Can" (skills) → LLM decides action
2. **Multimodal Input**: Image + Detection JSON + Skill Registry
3. **Structured Output**: JSON schema with reasoning + action + parameters
4. **Safety Validation**: Local checks before action execution

### Implementation Tasks (Phase 3)

**Status**: ✅ Complete

**Results**:
- Gemini Bridge node successfully implemented (440 lines)
- 5-state machine: IDLE → SENSING → REASONING → ACTING → EVALUATING
- Multimodal AI reasoning (camera + detections → actions)
- SayCan principle: Goal + Skills → LLM decides next action
- JSON schema enforcement with explicit parameter properties
- Successfully tested: Robot finds bottle, centers it via rotation, approaches
- Two critical bugs found and fixed:
  1. Parameter generation (needed explicit schema properties)
  2. Rotation logic (was inverted left/right)

**Lessons Learned**:
- google.generativeai schema requires explicit `properties` definition, not just descriptions
- LLM understands spatial reasoning (x coordinates vs. center) very well
- Prompt engineering crucial: Examples must be 100% correct
- Head position not used yet (only body rotation) - could improve precision
- System works but YOLO limits semantic understanding

**Known Limitations**:
- YOLO only detects 80 COCO classes (bottle, person, chair, etc.)
- Cannot understand arbitrary objects ("red cube", "toy car", "door")
- Head position skill underutilized (Gemini prefers body rotation)

#### Core Components:
1. ✅ **Skill Registry**: Define all ROS2 Actions with descriptions for LLM
2. ✅ **Perception Aggregator**: Subscribe to detections + images, convert to Gemini format
3. ✅ **Gemini API Client**: Multimodal input, JSON schema enforcement
4. ✅ **Action Dispatcher**: Parse JSON, execute actions with safety checks
5. ✅ **State Machine**: IDLE → SENSING → REASONING → ACTING → EVALUATING

**Detailed implementation concept**:

---

### 1. System-Architektur: Die "Embodied Bridge"

Wir nutzen einen zentralen ROS 2 Node (Python), der als Vermittler fungiert. Er abstrahiert die Komplexität der Cloud-Anbindung für den restlichen Roboter.

#### Die Komponenten:

* **Perception-Aggregator:** Sammelt FPV- und Hypervisor-Bilder synchron.
* **Skill-Registry (SayCan):** Eine Liste aller verfügbaren ROS 2 Actions mit Beschreibungen, was sie bewirken und wann sie scheitern könnten.
* **Action-Dispatcher (VLA):** Übersetzt die Gemini-Ausgabe in tatsächliche ROS 2 Action-Goals.

---

### 2. Das Phasen-Konzept

#### A. Sense (Wahrnehmung)

Der Node abonniert die Image-Topics. Um Bandbreite zu sparen und die Latenz zu optimieren, triggern wir den "Sense"-Vorgang nur, wenn der Roboter bereit für den nächsten Schritt ist (Idle-State).

* **Input:** `Image_FPV` (640x480) + `Image_Hypervisor` (HD für Kontext).
* **Preprocessing:** Normalisierung der Bilder und Konvertierung in Base64 für die Gemini-API.

#### B. Plan (Reasoning via SayCan & VLA)

Wir senden einen **Multimodalen Prompt** an Gemini. Dieser enthält:

1. **Den visuellen Kontext:** Beide Bilder.
2. **Die "Can"-Liste (Skills):** Eine strukturierte Liste deiner ROS 2 Actions.
3. **Die Aufgabe (Say):** "Bringe das Spielzeug in die Box."

**Der Prompt-Aufbau (System-Instruction):**

> "Du bist das Embodied-AI-Gehirn eines Laufroboters. Analysiere Bild 1 (deine Sicht) und Bild 2 (Sicht von oben).
> Entscheide basierend auf dem SayCan-Prinzip:
> 1. **Say:** Was ist der nächste logische Schritt zur Zielerreichung?
> 2. **Can:** Ist die Aktion 'WalkTo' im aktuellen Bild-Kontext sicher (keine Abgründe)?
     > Gib die Aktion im VLA-Format (Vision-Language-Action) zurück."
>
>

#### C. Act (Ausführung)

Gemini antwortet mit einem strikten JSON-Schema. Dein Node parst dieses und startet den entsprechenden ROS 2 Action-Client.

---

### 3. Implementierung des JSON-Schemas (VLA-Output)

Damit dein Roboter nicht "halluziniert", erzwingen wir ein Response-Schema. Dieses Schema ist der Kern deiner **Vision-Language-Action** Schnittstelle:

```json
{
  "thought_process": {
    "observation": "FPV zeigt einen Absatz, Hypervisor zeigt freien Weg rechts davon.",
    "affordance_check": "WalkTo ist möglich, wenn Kurs um 15 Grad korrigiert wird.",
    "safety_rating": "high"
  },
  "action": {
    "type": "ROS2_ACTION",
    "interface": "my_robot_msgs/action/Navigate",
    "goal": {
      "target_pose": { "x": 1.2, "y": -0.3, "theta": 0.26 },
      "speed_limit": 0.5
    }
  },
  "hypervisor_notes": "Achte auf die Stabilität des Standbeins beim Start."
}

```

---

### 4. Python-Node Struktur (Pseudo-Code)

Hier ist das Skelett für deine ROS 2 "Embodied Bridge":

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from my_robot_interfaces.action import WalkTo
from rclpy.action import ActionClient

class GeminiEmbodiedBridge(Node):
    def __init__(self):
        super().__init__('gemini_bridge')
        # Skill Registry (SayCan Definition)
        self.skills = {
            "WalkTo": "Bewege den Roboter zu x,y Koordinaten relativ zur aktuellen Position.",
            "Rotate": "Drehe den Roboter um einen Winkel in Radiant."
        }
        self.fpv_sub = self.create_subscription(Image, '/camera/fpv', self.save_fpv, 10)
        self.hyper_sub = self.create_subscription(Image, '/camera/hypervisor', self.save_hyper, 10)
        self.action_client = ActionClient(self, WalkTo, 'walk_to')

    def run_planning_cycle(self, user_goal):
        # 1. SENSE: Bilder holen
        img1, img2 = self.latest_fpv, self.latest_hyper
        
        # 2. PLAN: Gemini API Call (Multi-Modal)
        # Hier wird das SayCan-Prinzip im Prompt angewendet
        gemini_response = self.call_gemini_api(img1, img2, user_goal, self.skills)
        
        # 3. ACT: ROS 2 Action starten
        self.execute_action(gemini_response)

    def execute_action(self, json_data):
        goal_msg = WalkTo.Goal()
        goal_msg.x = json_data['action']['goal']['x']
        # ... Action Call an ROS 2
        self.action_client.send_goal_async(goal_msg)

```

---

### 5. Sicherheit & Hypervisor-Logik

Da du eine externe Kamera hast, implementieren wir ein **"Dual-Guard"-Prinzip**:

1. **Cross-Check:** Gemini vergleicht die Tiefenschätzung der FPV-Kamera mit der Distanzmessung aus der Hypervisor-Perspektive. Weichen diese mehr als 10% ab, verweigert das Modell die Aktion (`affordance_check: failed`).
2. **Fallback:** Wenn der Hypervisor-Agent sieht, dass der Roboter schwankt (Instabilität), überschreibt er den Plan des Navigator-Agenten mit einer `Stop`-Action.

---

## 📋 Current Phase (Phase 4 - Pure Vision AI)

### Goal
Remove YOLO dependency entirely. Use Gemini's native vision understanding for:
- Arbitrary object recognition ("Find the red cube", "Go to the door")
- Scene understanding beyond 80 COCO classes
- Semantic spatial reasoning without structured detections
- True multimodal AI control

### Why Remove YOLO?
**Current limitations:**
- YOLO: Only 80 classes (bottle, person, chair, etc.)
- Cannot find: "red objects", "doors", "toys", "specific shapes"
- Gemini is limited to reasoning over YOLO's pre-classified detections
- Semantic gap: User says "red thing" → YOLO sees nothing → Gemini blind

**Benefits of Pure Vision:**
- Unlimited semantic understanding (any object, color, shape)
- Natural language goals work directly ("Find something red")
- Simpler architecture (one AI system, not two)
- Lower latency (no YOLO preprocessing)
- More flexible behavior

### Architecture Changes
**Before (Phase 3):**
```
Camera → YOLO → Detections JSON → Gemini → Actions
```

**After (Phase 4):**
```
Camera → Gemini (direct vision) → Actions
```

### Implementation Tasks
1. ⏳ **Test Pure Vision Mode**: Add `use_yolo: false` parameter to gemini_bridge
2. ⏳ **Update System Instruction**: Remove detection-specific prompts, add vision analysis
3. ⏳ **Benchmark Performance**: Compare latency/cost (YOLO+Gemini vs. Pure Gemini)
4. ⏳ **Integration Testing**: Test with non-YOLO objects ("red cup", "wooden block")
5. ⏳ **Head Position Utilization**: Improve prompts to encourage camera scanning

### Notes
- Keep YOLO code for fallback/comparison
- May need to increase Gemini timeout (vision-only might be slower)
- Consider prompt engineering to request structured vision output

