# Phase 3 Testing Guide - Ubuntu1 Integration

## Quick Start (auf ubuntu1)

### 1. System vorbereiten
```bash
ssh ubuntu@192.168.2.133
cd ~/Hexapod_Brain
git pull  # Hole neueste Änderungen
```

### 2. Gemini API Setup
```bash
# Install Dependencies
bash scripts/setup_gemini_ubuntu1.sh

# API Key konfigurieren
nano ~/.bashrc
# Am Ende hinzufügen:
export GEMINI_API_KEY='your_api_key_here'

# Bashrc neu laden
source ~/.bashrc

# API testen
python3 scripts/test_gemini_api.py
```

**Erwartetes Ergebnis**: Alle 6 Tests sollten ✅ sein

### 3. Workspace bauen
```bash
cd ~/Hexapod_Brain
source setup_env.bash  # ROS_DOMAIN_ID=1 + sources workspace
colcon build --symlink-install
source install/setup.bash  # Neues Package laden
```

### 4. Komplettes System testen

**Terminal 1**: YOLO Detector starten
```bash
source ~/Hexapod_Brain/setup_env.bash
ros2 launch hexapod_vision yolo_detector_tflite.launch.py
```

**Terminal 2**: Gemini Bridge starten
```bash
source ~/Hexapod_Brain/setup_env.bash
ros2 launch hexapod_navigation gemini_bridge.launch.py
```

**Terminal 3**: Status monitoren
```bash
source ~/Hexapod_Brain/setup_env.bash
ros2 topic echo /hexapod/navigation/status
```

**Terminal 4**: Reasoning monitoren (optional)
```bash
source ~/Hexapod_Brain/setup_env.bash
ros2 topic echo /hexapod/reasoning
```

**Terminal 5**: Goal senden
```bash
source ~/Hexapod_Brain/setup_env.bash

# Test 1: Flasche finden
ros2 topic pub --once /hexapod/goal std_msgs/String "data: 'Find the bottle'"

# Test 2: Flasche annähern
ros2 topic pub --once /hexapod/goal std_msgs/String "data: 'Find and approach the bottle'"

# Test 3: Person finden
ros2 topic pub --once /hexapod/goal std_msgs/String "data: 'Find a person'"
```

## Erwartetes Verhalten

### Test 1: "Find the bottle"
1. **SENSING**: Sammelt Bild + Detections
2. **REASONING**: Gemini analysiert Szene
3. **ACTING**: 
   - Keine Flasche? → Rotiert, um zu scannen (45°)
   - Flasche gefunden? → Geht zu EVALUATING
4. **EVALUATING**: Prüft, ob Flasche gefunden
5. Wiederholt, bis Flasche im Bild

### Test 2: "Find and approach the bottle"
1. Wie Test 1, ABER:
2. Wenn Flasche gefunden:
   - **Off-center?** → Rotiert zum Zentrieren
   - **Centered?** → Bewegt sich vorwärts
   - **Close enough?** (bbox >= 120px) → GOAL_ACHIEVED
3. Wiederholt SENSING → REASONING → ACTING bis Ziel erreicht

### Test 3: "Find a person"
1. Wie Test 1, aber sucht nach "person" statt "bottle"
2. Demonstriert Flexibilität: Gleicher Code, anderes Ziel

## Logs Interpretation

### SENSING Phase
```
[INFO] State: IDLE → SENSING
[INFO] ✅ Perception data collected
```
→ Bild + Detections erfolgreich gesammelt

### REASONING Phase
```
[INFO] State: SENSING → REASONING
[INFO] 🤖 Calling Gemini API (attempt 1/3)...
[INFO] 💡 Gemini reasoning:
[INFO]    Observation: Bottle detected at x=520 (right side). Off-center.
[INFO]    Goal status: in progress - bottle found but not centered
[INFO]    Safety: high
[INFO]    Action: rotate
[INFO]    Explanation: Rotate right to center bottle in view
```
→ Gemini hat Szene analysiert und Aktion gewählt

### ACTING Phase
```
[INFO] State: REASONING → ACTING
[INFO] 🚀 Executing: rotate with {'angle_degrees': 20.0, 'speed': 40}
[INFO] Goal accepted, waiting for result...
[INFO] ✅ Action complete (success=True)
```
→ Aktion wurde ausgeführt

### EVALUATING Phase
```
[INFO] State: ACTING → EVALUATING
[INFO] 🔄 Goal not yet achieved, continuing...
[INFO] State: EVALUATING → SENSING
```
→ Ziel noch nicht erreicht, nächste Iteration

ODER

```
[INFO] State: ACTING → EVALUATING
[INFO] 🎉 Goal achieved!
[INFO] State: EVALUATING → IDLE
```
→ Ziel erreicht!

## Troubleshooting

### "GEMINI_API_KEY not set"
```bash
# Prüfen
echo $GEMINI_API_KEY

# Setzen (falls leer)
export GEMINI_API_KEY='your_key_here'
# ODER in ~/.bashrc permanent speichern
```

### "No image received yet"
```bash
# Camera aktivieren (falls pausiert)
ros2 service call /raspclaws/set_camera_pause std_srvs/srv/SetBool "{data: false}"

# Camera Topic prüfen
ros2 topic hz /raspclaws/camera/image_raw/compressed
```

### "No detections received yet"
```bash
# Detector starten (falls nicht läuft)
ros2 launch hexapod_vision yolo_detector_tflite.launch.py

# Detections prüfen
ros2 topic hz /hexapod/detections
ros2 topic echo /hexapod/detections
```

### "Action servers not found"
```bash
# Prüfe raspclaws-1 services
ssh pi@192.168.2.126
systemctl status ros_server.service

# ROS_DOMAIN_ID prüfen
echo $ROS_DOMAIN_ID  # Sollte 1 sein!

# Actions listen
ros2 action list
# Sollte zeigen:
# /raspclaws/linear_move
# /raspclaws/rotate
# /raspclaws/head_position
```

### "Gemini API error: 403" oder "Model not found"
- API Key ungültig → https://makersuite.google.com/app/apikey
- Billing nicht aktiviert → https://console.cloud.google.com/billing
- Robotics ER model nicht zugänglich (Preview) → Request access oder use fallback:
  ```bash
  ros2 launch hexapod_navigation gemini_bridge.launch.py model_name:='gemini-1.5-pro'
  ```

### "Robot zappelt"
- Sollte NICHT passieren (two-stage callbacks implementiert)
- Falls doch: Check logs für "Action complete" messages
- Vergleiche mit bottle_seeker.py behavior

## Performance Erwartungen

| Metrik | Erwartung |
|--------|-----------|
| **Gemini API Latenz** | 1-3 Sekunden pro Entscheidung |
| **Control Loop** | 1 Hz (1 Entscheidung/Sekunde) |
| **Gesamt-Latenz** | 3-5 Sekunden pro Aktion |
| **API Cost** | ~$0.01-0.02 pro Entscheidung (robotics-specialized) |
| **Session Cost** | ~$0.10-1.00 (10-50 Entscheidungen) |

**Model**: `gemini-robotics-er-1.5-preview` - spezialisiert für Robotik mit Vision-Language-Action

**Vergleich zu bottle_seeker**:
- bottle_seeker: 10 Hz, ~0.1s pro Iteration, instant decisions
- gemini_bridge: 1 Hz, ~3s pro Iteration, LLM decisions

**Trade-off**: Langsamer, aber viel flexibler!

## Erfolgs-Kriterien

✅ **API Tests bestanden** (alle 6 Tests grün)
✅ **Workspace gebaut** (ohne Fehler)
✅ **Detector läuft** (detections sichtbar)
✅ **Bridge startet** (keine crashes)
✅ **Goal verarbeitet** (state transitions sichtbar)
✅ **Gemini antwortet** (reasoning JSON erscheint)
✅ **Actions ausgeführt** (Roboter bewegt sich)
✅ **Smooth movement** (kein "zappeln")
✅ **Goal erreicht** (status = GOAL_ACHIEVED)

## Nächste Schritte nach erfolgreichem Test

1. **Dokumentation finalisieren**:
   - Update TODO.md: Phase 3 complete
   - Update MEMORY.md: Lessons learned
   - Create 2026-02-22.md entry

2. **Performance optimieren**:
   - Testen verschiedener Gemini models (1.5-flash vs 2.0-flash-exp)
   - Loop frequency anpassen (0.5 Hz vs 1 Hz)
   - Caching für wiederholte Szenen

3. **Phase 4 vorbereiten**:
   - Hypervisor camera integration
   - Multi-step planning
   - Experience storage

## Fragen?

Siehe:
- `scripts/GEMINI_SETUP.md` - API Setup Details
- `src/hexapod_navigation/GEMINI_BRIDGE.md` - Node Documentation
- `src/hexapod_navigation/hexapod_navigation/gemini_prompts.py` - System Instruction
