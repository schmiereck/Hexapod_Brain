# Hexapod Vision-Navigation Projekt

## Vision-basierte autonome Navigation mit ROS2 (Distributed Architecture)

---

## 📋 Projekt-Übersicht

**Ziel:** Hexapod-Roboter (`raspclaws-1`) mit Kamera ausstatten, der autonom Objekte findet und dorthin navigiert, wobei die rechenintensive Bildverarbeitung auf einem dedizierten Compute-Server (`ubuntu1`) läuft.

**Hardware - Distributed Setup:**
*   **Roboter (raspclaws-1)**: Raspberry Pi 3/4 (Raspbian)
    *   Führt Hardware-Steuerung aus (Servos, LEDs, Sensoren)
    *   Publiziert Kamera-Stream (`/raspclaws/camera/image_raw`)
    *   Führt ROS2 Actions aus (`/raspclaws/linear_move`, etc.)
*   **Compute Server (ubuntu1)**: Raspberry Pi 5 8GB (Ubuntu 24.04 + ROS2 Humble)
    *   Empfängt Kamera-Stream
    *   Führt YOLO Object Detection aus
    *   Berechnet Pfadplanung / Navigation
    *   Sendet Action Goals an den Roboter

**Software-Stack:**
- ROS2 Humble (beide Nodes)
- `raspclaws_interfaces` (gemeinsame Definitionen)
- YOLOv8 (auf ubuntu1)
- PyTorch (auf ubuntu1)

---

## 🎯 Phasen-Übersicht

### Phase 1: Basis-Setup (1-2 Wochen) ✅
Setup der Entwicklungsumgebung und erste Objekterkennung

### Phase 2: Hybrid Navigation (2-3 Wochen) 🎯
YOLO + kleines Policy Network für Behavior Cloning

### Phase 3: Latent World Model (1-2 Monate) 🚀
Self-supervised Learning mit Zukunftsvorhersage

### Phase 4: Multi-Task Learning (2-3 Monate) 🌟
Mehrere Aufgaben, Goal Embeddings, Meta-Learning

---

# Phase 1: Basis-Setup und YOLO Integration

## 🎯 Ziele
- ROS2 Entwicklungsumgebung einrichten
- YOLO erfolgreich auf Raspberry Pi 5 laufen lassen
- Erste Objekterkennung in ROS2 integrieren
- Baseline Performance messen

## 💻 Hardware-Anforderungen
**Ausreichend:**
- ✅ Raspberry Pi 5 8GB
- ✅ Kamera
- Entwicklungs-PC für Code-Entwicklung

**Performance Erwartung:**
- YOLO-Nano: 5-10 FPS
- Latenz: 100-200ms
- Ausreichend für Hexapod-Geschwindigkeit

---

## ✅ TODO-Liste Phase 1

### 1.1 Entwicklungsumgebung einrichten

#### [ ] Task: ROS2 Workspace aufsetzen
```bash
# Auf Raspberry Pi 5
mkdir -p ~/hexapod_ws/src
cd ~/hexapod_ws
colcon build
source install/setup.bash
```

**Ergebnis:** Funktionierender ROS2 Workspace

---

#### [ ] Task: Python Dependencies installieren
```bash
# Auf Raspberry Pi 5
pip3 install ultralytics opencv-python torch torchvision
pip3 install cv_bridge  # ROS2 <-> OpenCV
```

**Ergebnis:** Alle Python-Pakete installiert

---

#### [ ] Task: Kamera-Node testen
```bash
# Kamera-Stream testen
ros2 run usb_cam usb_cam_node_exe
# In anderem Terminal
ros2 topic echo /camera/image_raw
```

**Erstelle Test-Script:**
```python
# ~/hexapod_ws/src/test_camera.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraTest(Node):
    def __init__(self):
        super().__init__('camera_test')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.callback, 10)
    
    def callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        cv2.imshow("Camera", cv_image)
        cv2.waitKey(1)
        self.get_logger().info(f"Image shape: {cv_image.shape}")

def main():
    rclpy.init()
    node = CameraTest()
    rclpy.spin(node)
```

**Ergebnis:** Kamera-Stream wird empfangen und angezeigt

---

### 1.2 YOLO Integration

#### [ ] Task: YOLO-Modell herunterladen und testen
```python
# ~/hexapod_ws/src/test_yolo.py
from ultralytics import YOLO
import cv2

# Modell laden (wird automatisch heruntergeladen)
model = YOLO('yolov8n.pt')  # Nano version

# Test-Bild
img = cv2.imread('test.jpg')
results = model(img)

# Detections anzeigen
for r in results:
    boxes = r.boxes
    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"Klasse: {model.names[cls]}, Konfidenz: {conf:.2f}")
```

**Performance-Test durchführen:**
```python
import time

# 10 Durchläufe für Durchschnitts-FPS
start = time.time()
for _ in range(10):
    results = model(img)
end = time.time()

fps = 10 / (end - start)
print(f"FPS auf Pi 5: {fps:.2f}")
```

**Ergebnis:** YOLO läuft, FPS gemessen (erwarte 5-10 FPS)

---

#### [ ] Task: ROS2 YOLO Detection Node erstellen

**Package-Struktur:**
```
hexapod_ws/src/
└── hexapod_vision/
    ├── package.xml
    ├── setup.py
    ├── hexapod_vision/
    │   ├── __init__.py
    │   ├── yolo_detector.py
    │   └── utils.py
    └── launch/
        └── yolo_detector.launch.py
```

**Datei: `yolo_detector.py`**
```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2

class YOLODetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        
        # Parameter
        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('image_topic', '/camera/image_raw')
        
        model_path = self.get_parameter('model_path').value
        self.conf_threshold = self.get_parameter('confidence_threshold').value
        image_topic = self.get_parameter('image_topic').value
        
        # YOLO laden
        self.model = YOLO(model_path)
        self.bridge = CvBridge()
        
        # Subscriber & Publisher
        self.image_sub = self.create_subscription(
            Image, image_topic, self.image_callback, 10)
        
        self.detection_pub = self.create_publisher(
            Detection2DArray, '/detections', 10)
        
        self.annotated_pub = self.create_publisher(
            Image, '/detections/image', 10)
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = self.get_clock().now()
        
        self.get_logger().info('YOLO Detector Node gestartet')
    
    def image_callback(self, msg):
        # Bild konvertieren
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # YOLO Inference
        results = self.model(cv_image, conf=self.conf_threshold, verbose=False)
        
        # Detection2DArray erstellen
        detection_array = Detection2DArray()
        detection_array.header = msg.header
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                detection = Detection2D()
                
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detection.bbox.center.position.x = (x1 + x2) / 2
                detection.bbox.center.position.y = (y1 + y2) / 2
                detection.bbox.size_x = x2 - x1
                detection.bbox.size_y = y2 - y1
                
                # Klasse und Konfidenz
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = str(int(box.cls[0]))
                hypothesis.hypothesis.score = float(box.conf[0])
                detection.results.append(hypothesis)
                
                detection_array.detections.append(detection)
        
        # Publishen
        self.detection_pub.publish(detection_array)
        
        # Annotiertes Bild publishen
        annotated = results[0].plot()
        annotated_msg = self.bridge.cv2_to_imgmsg(annotated, "bgr8")
        annotated_msg.header = msg.header
        self.annotated_pub.publish(annotated_msg)
        
        # FPS loggen
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            elapsed = (self.get_clock().now() - self.start_time).nanoseconds / 1e9
            fps = self.frame_count / elapsed
            self.get_logger().info(f'FPS: {fps:.2f}, Detections: {len(detection_array.detections)}')

def main(args=None):
    rclpy.init(args=args)
    node = YOLODetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

**Launch-File: `yolo_detector.launch.py`**
```python
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
                'image_topic': '/camera/image_raw'
            }],
            output='screen'
        )
    ])
```

**Package erstellen:**
```bash
cd ~/hexapod_ws/src
ros2 pkg create hexapod_vision --build-type ament_python --dependencies rclpy sensor_msgs vision_msgs cv_bridge

# setup.py anpassen für Entry Points
# package.xml anpassen für Dependencies
```

**Ergebnis:** YOLO läuft als ROS2 Node, publisht Detections

---

#### [ ] Task: Visualisierung und Testing
```bash
# Node starten
ros2 launch hexapod_vision yolo_detector.launch.py

# In anderem Terminal: Detections anschauen
ros2 topic echo /detections

# Annotiertes Bild anschauen
ros2 run rqt_image_view rqt_image_view /detections/image
```

**Test-Checklist:**
- [ ] Objekte werden erkannt (Ball, Person, etc.)
- [ ] Bounding Boxes sind korrekt
- [ ] FPS ist >= 5
- [ ] Detections werden published

**Ergebnis:** Funktionierende Objekterkennung in ROS2

---

### 1.3 Performance Baseline

#### [ ] Task: Performance-Messungen dokumentieren

**Test-Script erstellen:**
```python
# ~/hexapod_ws/src/benchmark_yolo.py
import time
import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO('yolov8n.pt')

# Test mit verschiedenen Auflösungen
resolutions = [(320, 320), (416, 416), (640, 640)]

for res in resolutions:
    test_img = np.random.randint(0, 255, (*res, 3), dtype=np.uint8)
    
    times = []
    for _ in range(50):
        start = time.time()
        results = model(test_img, verbose=False)
        times.append(time.time() - start)
    
    avg_time = np.mean(times)
    fps = 1 / avg_time
    
    print(f"Resolution {res}: {fps:.2f} FPS, {avg_time*1000:.1f}ms")
```

**Messwerte dokumentieren:**
```markdown
# Performance Baseline Raspberry Pi 5

## YOLO-Nano (yolov8n.pt)
- 320x320: ~X FPS, ~Xms
- 640x640: ~X FPS, ~Xms

## RAM-Nutzung:
- Idle: ~XMB
- Mit YOLO: ~XMB

## CPU-Last:
- Durchschnitt: ~X%
```

**Ergebnis:** Baseline-Performance dokumentiert

---

## 📊 Phase 1 Abschluss

### Erfolgskriterien:
- [x] ROS2 Workspace funktioniert
- [x] YOLO erkennt Objekte in Echtzeit
- [x] FPS >= 5
- [x] Detections werden als ROS2 Messages published
- [x] Performance-Baseline dokumentiert

### Deliverables:
- Funktionierender YOLO Detection Node
- Launch-Files
- Test-Scripts
- Performance-Dokumentation

---
---

# Phase 2: Hybrid Navigation (YOLO + Policy Network)

## 🎯 Ziele
- Datensammlung für Behavior Cloning
- Kleines Policy Network trainieren
- Erste autonome Navigation implementieren
- Hexapod navigiert zu erkannten Objekten

## 💻 Hardware-Anforderungen

### Für Inference (Raspberry Pi 5):
- ✅ **AUSREICHEND** für Policy Network
- Policy NN: <1ms Inferenz
- YOLO + Policy: ~100-150ms total
- ✅ **Keine zusätzliche Hardware nötig**

### Für Training:
- **Option A: Laptop/Desktop mit GPU** (empfohlen)
  - NVIDIA GPU (auch alte GTX 1060 reicht)
  - Training: 10-30 Minuten für 50-200 Episoden
  - **💰 Kosten: 0€** (vorhandene Hardware)

- **Option B: Cloud Training** (falls kein GPU-PC)
  - Google Colab Free: ✅ Ausreichend!
  - AWS/GCP: ~2-5€ für komplettes Training
  - **💰 Kosten: 0-5€**

- **Option C: Direkt auf Pi 5** (nicht empfohlen)
  - ❌ Sehr langsam (Stunden statt Minuten)
  - Nur für kleine Datensätze (<50 Samples)

**Empfehlung:** Laptop/PC mit GPU für Training, Pi 5 für Deployment

---

## ✅ TODO-Liste Phase 2

### 2.1 Datensammlung vorbereiten

#### [ ] Task: Manuelle Steuerungs-Interface erstellen

**ROS2 Package für Teleop:**
```python
# ~/hexapod_ws/src/hexapod_vision/hexapod_vision/teleop_keyboard.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import sys
import termios
import tty

class TeleopKeyboard(Node):
    def __init__(self):
        super().__init__('teleop_keyboard')
        self.action_pub = self.create_publisher(String, '/manual_action', 10)
        
        self.get_logger().info('''
        Hexapod Teleoperation
        ---------------------
        w: forward
        a: turn left
        d: turn right
        s: stop
        q: quit
        ''')
    
    def get_key(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            key = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return key
    
    def run(self):
        while rclpy.ok():
            key = self.get_key()
            
            action = None
            if key == 'w':
                action = 'forward'
            elif key == 'a':
                action = 'turn_left'
            elif key == 'd':
                action = 'turn_right'
            elif key == 's':
                action = 'stop'
            elif key == 'q':
                break
            
            if action:
                msg = String()
                msg.data = action
                self.action_pub.publish(msg)
                self.get_logger().info(f'Action: {action}')

def main():
    rclpy.init()
    node = TeleopKeyboard()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
```

**Ergebnis:** Keyboard-Steuerung für Hexapod

---

#### [ ] Task: Data Collector Node erstellen

```python
# ~/hexapod_ws/src/hexapod_vision/hexapod_vision/data_collector.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import json
import os
from datetime import datetime

class DataCollector(Node):
    def __init__(self):
        super().__init__('data_collector')
        
        # Parameter
        self.declare_parameter('save_path', '/home/pi/hexapod_data')
        self.save_path = self.get_parameter('save_path').value
        
        # Verzeichnisse erstellen
        self.session_dir = os.path.join(
            self.save_path, 
            f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(os.path.join(self.session_dir, 'images'), exist_ok=True)
        
        self.bridge = CvBridge()
        self.sample_count = 0
        
        # Subscriber
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        
        self.detection_sub = self.create_subscription(
            Detection2DArray, '/detections', self.detection_callback, 10)
        
        self.action_sub = self.create_subscription(
            String, '/manual_action', self.action_callback, 10)
        
        # Speicher für aktuellen Frame
        self.current_image = None
        self.current_detections = None
        
        # Dataset
        self.dataset = []
        
        self.get_logger().info(f'Data Collector gestartet. Speichert in: {self.session_dir}')
    
    def image_callback(self, msg):
        self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
    
    def detection_callback(self, msg):
        self.current_detections = msg
    
    def action_callback(self, msg):
        # Wenn Action kommt, speichere (Image, Detections, Action)
        if self.current_image is None or self.current_detections is None:
            return
        
        # Bild speichern
        img_filename = f"img_{self.sample_count:05d}.jpg"
        img_path = os.path.join(self.session_dir, 'images', img_filename)
        cv2.imwrite(img_path, self.current_image)
        
        # Detections extrahieren (Features für Policy Network)
        features = self.extract_features(self.current_detections)
        
        # Sample speichern
        sample = {
            'id': self.sample_count,
            'image': img_filename,
            'features': features,
            'action': msg.data,
            'timestamp': self.get_clock().now().to_msg()
        }
        
        self.dataset.append(sample)
        self.sample_count += 1
        
        self.get_logger().info(f'Sample {self.sample_count} gespeichert: {msg.data}')
        
        # Periodisch speichern
        if self.sample_count % 10 == 0:
            self.save_dataset()
    
    def extract_features(self, detections):
        """Extrahiere Features aus Detections für Policy Network"""
        if len(detections.detections) == 0:
            # Keine Objekte erkannt
            return {
                'has_object': False,
                'x_center': 0.5,
                'y_center': 0.5,
                'width': 0.0,
                'height': 0.0,
                'confidence': 0.0,
                'class_id': -1
            }
        
        # Nehme größtes Objekt (Proxy für nächstes)
        largest = max(
            detections.detections,
            key=lambda d: d.bbox.size_x * d.bbox.size_y
        )
        
        # Normalisiere auf [0, 1]
        # Annahme: Bild ist 640x480
        img_width, img_height = 640, 480
        
        return {
            'has_object': True,
            'x_center': largest.bbox.center.position.x / img_width,
            'y_center': largest.bbox.center.position.y / img_height,
            'width': largest.bbox.size_x / img_width,
            'height': largest.bbox.size_y / img_height,
            'confidence': largest.results[0].hypothesis.score,
            'class_id': int(largest.results[0].hypothesis.class_id)
        }
    
    def save_dataset(self):
        dataset_path = os.path.join(self.session_dir, 'dataset.json')
        with open(dataset_path, 'w') as f:
            json.dump(self.dataset, f, indent=2, default=str)
        self.get_logger().info(f'Dataset gespeichert: {len(self.dataset)} samples')

def main():
    rclpy.init()
    node = DataCollector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

**Ergebnis:** Daten werden während manueller Steuerung gesammelt

---

#### [ ] Task: Datensammlung durchführen

**Sammlung-Session:**
```bash
# Terminal 1: Kamera + YOLO
ros2 launch hexapod_vision yolo_detector.launch.py

# Terminal 2: Data Collector
ros2 run hexapod_vision data_collector

# Terminal 3: Teleop
ros2 run hexapod_vision teleop_keyboard

# Terminal 4: Hexapod Controller (dein bestehender)
ros2 run hexapod_controller controller_node
```

**Sammlung-Strategie:**
1. **Szenario 1: Ball links**
   - Platziere Ball links vom Hexapod
   - Drücke 'a' (turn_left) mehrmals
   - Dann 'w' (forward)
   - Wiederhole 10x mit verschiedenen Positionen

2. **Szenario 2: Ball rechts**
   - Analog mit 'd' (turn_right)
   - 10x wiederholen

3. **Szenario 3: Ball mittig**
   - Nur 'w' (forward)
   - 10x wiederholen

4. **Szenario 4: Kein Ball**
   - 's' (stop) oder zufällige Rotation
   - 5x wiederholen

**Ziel:** Mindestens **50-100 Samples** sammeln

**Ergebnis:** Dataset mit Bild-Feature-Action Tripeln

---

### 2.2 Policy Network trainieren

#### [ ] Task: Training-Script erstellen (auf PC/Cloud)

**Datei: `train_policy.py`** (auf Trainings-PC)
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from sklearn.model_selection import train_test_split

# Policy Network Definition
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=6, num_actions=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_actions)
        )
    
    def forward(self, x):
        return self.network(x)

# Dataset
class HexapodDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        # Action mapping
        self.action_to_idx = {
            'forward': 0,
            'turn_left': 1,
            'turn_right': 2,
            'stop': 3
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Features extrahieren
        features = sample['features']
        x = torch.tensor([
            features['x_center'],
            features['y_center'],
            features['width'],
            features['height'],
            features['confidence'],
            1.0 if features['has_object'] else 0.0
        ], dtype=torch.float32)
        
        # Action label
        action = sample['action']
        y = torch.tensor(self.action_to_idx[action], dtype=torch.long)
        
        return x, y

# Training
def train_policy(data_path, epochs=50, batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training auf: {device}')
    
    # Dataset laden
    dataset = HexapodDataset(data_path)
    
    # Train/Val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Model
    model = PolicyNetwork().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += y.size(0)
            train_correct += predicted.eq(y).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += y.size(0)
                val_correct += predicted.eq(y).sum().item()
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.2f}%')
        
        # Best model speichern
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'policy_best.pth')
            print(f'  ✓ Neues bestes Modell gespeichert!')
    
    print(f'\nTraining abgeschlossen. Beste Val Accuracy: {best_val_acc:.2f}%')
    
    # Model für Deployment exportieren
    model.load_state_dict(torch.load('policy_best.pth'))
    model.eval()
    
    # TorchScript export
    example_input = torch.randn(1, 6).to(device)
    traced = torch.jit.trace(model, example_input)
    traced.save('policy_deployment.pt')
    print('Deployment-Modell gespeichert: policy_deployment.pt')

if __name__ == '__main__':
    train_policy('dataset.json')
```

**Training durchführen:**
```bash
# Daten vom Pi kopieren
scp -r pi@hexapod:/home/pi/hexapod_data/session_XXX ./

# Training starten (GPU)
python train_policy.py

# Erwartete Dauer: 5-15 Minuten für 50 Epochen
```

**Ergebnis:** Trainiertes Policy Network (`policy_deployment.pt`)

---

#### [ ] Task: Modell zurück auf Pi kopieren
```bash
# Von Trainings-PC
scp policy_deployment.pt pi@hexapod:~/hexapod_ws/src/hexapod_vision/models/
```

**Ergebnis:** Modell auf Raspberry Pi

---

### 2.3 Autonome Navigation implementieren

#### [ ] Task: Policy-basierter Navigator Node

```python
# ~/hexapod_ws/src/hexapod_vision/hexapod_vision/policy_navigator.py
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String
import torch
import numpy as np

class PolicyNavigator(Node):
    def __init__(self):
        super().__init__('policy_navigator')
        
        # Parameter
        self.declare_parameter('model_path', 'policy_deployment.pt')
        model_path = self.get_parameter('model_path').value
        
        # Policy laden
        self.device = torch.device('cpu')  # Pi 5 = CPU
        self.policy = torch.jit.load(model_path, map_location=self.device)
        self.policy.eval()
        
        # Action mapping
        self.actions = ['forward', 'turn_left', 'turn_right', 'stop']
        
        # Subscriber
        self.detection_sub = self.create_subscription(
            Detection2DArray, '/detections', self.detection_callback, 10)
        
        # Publisher (für Visualisierung)
        self.action_pub = self.create_publisher(String, '/policy_action', 10)
        
        # Action Client zu deinem Hexapod Controller
        # Annahme: du hast eine Action "HexapodMove"
        # self.hexapod_client = ActionClient(self, HexapodMove, '/hexapod/move')
        
        self.get_logger().info('Policy Navigator gestartet!')
    
    def detection_callback(self, msg):
        # Features extrahieren (gleiche Logik wie beim Sammeln)
        features = self.extract_features(msg)
        
        # Als Tensor
        x = torch.tensor([
            features['x_center'],
            features['y_center'],
            features['width'],
            features['height'],
            features['confidence'],
            1.0 if features['has_object'] else 0.0
        ], dtype=torch.float32).unsqueeze(0)  # Batch dimension
        
        # Prediction
        with torch.no_grad():
            logits = self.policy(x)
            action_idx = torch.argmax(logits, dim=1).item()
        
        action = self.actions[action_idx]
        
        # Action publishen
        msg = String()
        msg.data = action
        self.action_pub.publish(msg)
        
        # An Hexapod senden
        self.execute_action(action)
        
        self.get_logger().info(f'Policy: {action} (Confidence: {torch.softmax(logits, dim=1).max():.2f})')
    
    def extract_features(self, detections):
        """Gleiche Logik wie DataCollector"""
        if len(detections.detections) == 0:
            return {
                'has_object': False,
                'x_center': 0.5,
                'y_center': 0.5,
                'width': 0.0,
                'height': 0.0,
                'confidence': 0.0
            }
        
        largest = max(
            detections.detections,
            key=lambda d: d.bbox.size_x * d.bbox.size_y
        )
        
        img_width, img_height = 640, 480
        
        return {
            'has_object': True,
            'x_center': largest.bbox.center.position.x / img_width,
            'y_center': largest.bbox.center.position.y / img_height,
            'width': largest.bbox.size_x / img_width,
            'height': largest.bbox.size_y / img_height,
            'confidence': largest.results[0].hypothesis.score
        }
    
    def execute_action(self, action):
        """Führe Action auf Hexapod aus"""
        # HIER: Integration mit deiner bestehenden Hexapod-Kinematik
        # Beispiel mit ROS2 Action:
        # goal = HexapodMove.Goal()
        # goal.action_type = action
        # self.hexapod_client.send_goal_async(goal)
        
        # Oder direkt mit Topic:
        # self.hexapod_cmd_pub.publish(...)
        
        pass  # TODO: An deine Implementierung anpassen

def main():
    rclpy.init()
    node = PolicyNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

**Ergebnis:** Autonomer Navigator mit gelernter Policy

---

#### [ ] Task: Integration mit Hexapod Kinematik

**Anpassen an deine ROS2 Actions:**
```python
# In policy_navigator.py, execute_action() anpassen:

from hexapod_interfaces.action import Move  # Dein Action Interface

class PolicyNavigator(Node):
    def __init__(self):
        # ...
        self.action_client = ActionClient(self, Move, '/hexapod/move')
        self.action_client.wait_for_server()
    
    def execute_action(self, action):
        goal = Move.Goal()
        
        if action == 'forward':
            goal.direction = 'forward'
            goal.distance = 0.1  # 10cm
        elif action == 'turn_left':
            goal.direction = 'rotate'
            goal.angle = 15.0  # 15 Grad links
        elif action == 'turn_right':
            goal.direction = 'rotate'
            goal.angle = -15.0  # 15 Grad rechts
        elif action == 'stop':
            return  # Nichts tun
        
        self.action_client.send_goal_async(goal)
```

**Ergebnis:** Policy steuert echten Hexapod

---

#### [ ] Task: End-to-End Test

**Test-Szenario:**
```bash
# Alles starten
ros2 launch hexapod_vision navigation.launch.py
```

**Launch-File erstellen:**
```python
# navigation.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Kamera (dein Node)
        Node(package='usb_cam', executable='usb_cam_node_exe'),
        
        # YOLO Detector
        Node(
            package='hexapod_vision',
            executable='yolo_detector',
            parameters=[{'model_path': 'yolov8n.pt'}]
        ),
        
        # Policy Navigator
        Node(
            package='hexapod_vision',
            executable='policy_navigator',
            parameters=[{'model_path': 'policy_deployment.pt'}]
        ),
        
        # Hexapod Controller (dein Node)
        # Node(package='hexapod_controller', executable='controller'),
    ])
```

**Test-Checkliste:**
- [ ] Platziere Ball vor Hexapod
- [ ] Hexapod dreht sich zum Ball
- [ ] Hexapod läuft auf Ball zu
- [ ] Stoppt in der Nähe (oder fährt drüber)

**Ergebnis:** Autonome Objektsuche funktioniert!

---

## 📊 Phase 2 Abschluss

### Erfolgskriterien:
- [x] 50-100 Trainings-Samples gesammelt
- [x] Policy Network trainiert (>70% Validation Accuracy)
- [x] Hexapod navigiert autonom zu Objekten
- [x] End-to-End System funktioniert

### Hardware-Entscheidung nach Phase 2:

**✅ Bleibe bei Raspberry Pi 5 wenn:**
- Navigation funktioniert zufriedenstellend (5-10 FPS)
- Latenz <200ms akzeptabel ist
- Budget begrenzt ist

**🤔 Erwäge Jetson Orin Nano (~250€) wenn:**
- Du >15 FPS brauchst
- Größere YOLO-Modelle (yolov8s/m) testen willst
- Multi-Objekt-Tracking wichtig wird
- Phase 3 (World Model) angehen willst

---
---

# Phase 3: Latent World Model

## 🎯 Ziele
- Self-supervised Learning einführen
- Modell lernt Zukunfts-Vorhersage (in Latent Space)
- Intrinsic Reward durch Prediction Error
- Robustere, adaptivere Navigation

## 💻 Hardware-Anforderungen

### ⚠️ **KRITISCHER PUNKT: Training**

**Raspberry Pi 5:**
- ❌ **NICHT ausreichend** für World Model Training
- World Model benötigt GPU
- Inference: möglich, aber grenzwertig

**Trainings-Optionen (in Reihenfolge der Empfehlung):**

1. **Desktop/Laptop mit NVIDIA GPU** 
   - GTX 1060 oder besser
   - 💰 **Kosten: 0€** (vorhandene Hardware)
   - ⏱️ Training: 1-4 Stunden
   - ✅ **BESTE Option**

2. **Google Colab Pro** (~10€/Monat)
   - Tesla T4 GPU
   - 💰 **Kosten: ~10€** für einen Monat
   - ⏱️ Training: 2-6 Stunden
   - ✅ Gut für Experimente

3. **Cloud (AWS/GCP)**
   - g4dn.xlarge (~0.50€/Stunde)
   - 💰 **Kosten: ~5-10€** für komplettes Training
   - ⏱️ Training: 2-4 Stunden
   - ✅ Pay-as-you-go

4. **Jetson Orin Nano** (~250€)
   - Kann sowohl trainieren als auch deployen
   - 💰 **Kosten: 250€** einmalig
   - ⏱️ Training: 4-8 Stunden (langsamer als Desktop GPU)
   - ✅ **Beste Langzeit-Investition** wenn du viel experimentieren willst

**Empfehlung für Phase 3:**
- **Wenn GPU-PC vorhanden:** Nutze diesen! ✅
- **Wenn kein GPU-PC:** 
  - Für Experimente: Google Colab (10€)
  - Für ernsthaftes Projekt: Jetson Orin Nano (250€)

### Deployment (Inference auf Roboter):

**Raspberry Pi 5:**
- ⚠️ **GRENZWERTIG** für World Model
- Latent World Model: ~30-50ms (machbar)
- Full Image Prediction: ❌ Zu langsam

**Jetson Orin Nano:**
- ✅ **OPTIMAL** für World Model
- Latent World Model: ~5-10ms
- Könnte auch Full Image Prediction (~50ms)

---

## 🔀 Hardware-Entscheidungsbaum

```
Hast du einen PC/Laptop mit NVIDIA GPU?
│
├─ JA → ✅ Nutze PC für Training, Pi 5 für Deployment
│        (Teste erstmal, ob Pi 5 Performance reicht)
│
└─ NEIN → Willst du langfristig mit ML experimentieren?
          │
          ├─ JA → 🚀 Kaufe Jetson Orin Nano (250€)
          │        - Training + Deployment auf einem Gerät
          │        - Beste Langzeit-Investition
          │
          └─ NEIN → 💻 Nutze Google Colab für Training
                    - 10€/Monat
                    - Deployment auf Pi 5 (teste Performance)
```

---

## ✅ TODO-Liste Phase 3

### 3.1 Trainings-Setup vorbereiten

#### [ ] Task: Trainings-Hardware entscheiden

**Mache Performance-Test auf Pi 5:**
```python
# test_world_model_inference.py
import torch
import time
import numpy as np

# Dummy World Model
class DummyWorldModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, 2, 1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1)
        )
        self.dynamics = torch.nn.Sequential(
            torch.nn.Linear(64 + 4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64)
        )

model = DummyWorldModel()
model.eval()

# Test
img = torch.randn(1, 3, 224, 224)
action = torch.randn(1, 4)

times = []
for _ in range(50):
    start = time.time()
    with torch.no_grad():
        z = model.encoder(img).flatten(1)
        z_next = model.dynamics(torch.cat([z, action], dim=1))
    times.append(time.time() - start)

avg_time = np.mean(times)
fps = 1 / avg_time
print(f"World Model Inference: {avg_time*1000:.1f}ms, {fps:.1f} FPS")
```

**Entscheidung basierend auf Ergebnis:**
- **<50ms:** ✅ Pi 5 ausreichend, bleibe dabei
- **50-100ms:** ⚠️ Grenzwertig, erwäge Jetson
- **>100ms:** ❌ Jetson Orin Nano kaufen

**Ergebnis:** Hardware-Entscheidung getroffen

---

#### [ ] Task: Trainings-Environment einrichten

**Auf Trainings-PC/Colab:**
```bash
# GPU check
python -c "import torch; print(torch.cuda.is_available())"

# Dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tensorboard wandb matplotlib
```

**Colab Notebook Template:**
```python
# World_Model_Training.ipynb

# GPU check
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Mount Google Drive (für Daten)
from google.colab import drive
drive.mount('/content/drive')

# Upload Dataset
# !unzip /content/drive/MyDrive/hexapod_data.zip
```

**Ergebnis:** Trainings-Environment bereit

---

### 3.2 Latent World Model implementieren

#### [ ] Task: World Model Architektur

```python
# world_model.py (auf Trainings-PC)
import torch
import torch.nn as nn
import torch.nn.functional as F

class VisualEncoder(nn.Module):
    """Encode images to latent vectors"""
    def __init__(self, latent_dim=256):
        super().__init__()
        # Nutze vortrainiertes MobileNetV2 als Basis
        mobilenet = torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=True)
        self.features = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(1280, latent_dim)
    
    def forward(self, x):
        # x: (B, 3, 224, 224)
        features = self.features(x)  # (B, 1280, 7, 7)
        pooled = self.avgpool(features).flatten(1)  # (B, 1280)
        latent = self.projection(pooled)  # (B, latent_dim)
        return latent

class DynamicsModel(nn.Module):
    """Predict next latent state given current latent + action"""
    def __init__(self, latent_dim=256, action_dim=4, hidden_dim=512):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, latent, action):
        # latent: (B, latent_dim)
        # action: (B, action_dim)
        x = torch.cat([latent, action], dim=1)
        next_latent = self.network(x)
        return next_latent

class RewardPredictor(nn.Module):
    """Predict reward from latent state"""
    def __init__(self, latent_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, latent):
        return self.network(latent)

class PolicyNetwork(nn.Module):
    """Select actions from latent state"""
    def __init__(self, latent_dim=256, action_dim=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, latent):
        return self.network(latent)

class WorldModel(nn.Module):
    """Complete World Model"""
    def __init__(self, latent_dim=256, action_dim=4):
        super().__init__()
        self.encoder = VisualEncoder(latent_dim)
        self.dynamics = DynamicsModel(latent_dim, action_dim)
        self.reward_predictor = RewardPredictor(latent_dim)
        self.policy = PolicyNetwork(latent_dim, action_dim)
    
    def forward(self, image, action):
        # Encode current image
        z_current = self.encoder(image)
        
        # Predict next latent
        z_next_pred = self.dynamics(z_current, action)
        
        # Predict reward
        reward_pred = self.reward_predictor(z_next_pred)
        
        # Get policy action
        action_logits = self.policy(z_current)
        
        return z_current, z_next_pred, reward_pred, action_logits
    
    def imagine(self, z_current, action_sequence):
        """Imagine future trajectory"""
        trajectory = [z_current]
        z = z_current
        
        for action in action_sequence:
            z = self.dynamics(z, action)
            trajectory.append(z)
        
        return trajectory
```

**Ergebnis:** World Model Architektur definiert

---

#### [ ] Task: Datensammlung für World Model

**Neue Daten sammeln mit Transitions:**
```python
# data_collector_transitions.py (auf Pi)
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import json
import os
from datetime import datetime

class TransitionCollector(Node):
    def __init__(self):
        super().__init__('transition_collector')
        
        self.declare_parameter('save_path', '/home/pi/hexapod_data')
        self.save_path = self.get_parameter('save_path').value
        
        self.session_dir = os.path.join(
            self.save_path,
            f"transitions_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.session_dir, exist_ok=True)
        os.makedirs(os.path.join(self.session_dir, 'images'), exist_ok=True)
        
        self.bridge = CvBridge()
        self.frame_count = 0
        
        # Buffer für (state, action, next_state, reward)
        self.prev_image = None
        self.current_action = None
        self.transitions = []
        
        # Subscriber
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        
        self.action_sub = self.create_subscription(
            String, '/policy_action', self.action_callback, 10)
        
        self.get_logger().info('Transition Collector gestartet')
    
    def image_callback(self, msg):
        current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # Wenn wir prev_image und action haben, speichere Transition
        if self.prev_image is not None and self.current_action is not None:
            self.save_transition(self.prev_image, self.current_action, current_image)
        
        self.prev_image = current_image
        self.current_action = None  # Reset
    
    def action_callback(self, msg):
        self.current_action = msg.data
    
    def save_transition(self, img_t, action, img_t1):
        # Bilder speichern
        img_t_path = f"img_{self.frame_count:05d}_t.jpg"
        img_t1_path = f"img_{self.frame_count:05d}_t1.jpg"
        
        cv2.imwrite(os.path.join(self.session_dir, 'images', img_t_path), img_t)
        cv2.imwrite(os.path.join(self.session_dir, 'images', img_t1_path), img_t1)
        
        # Reward berechnen (später von Supervisor)
        reward = 0.0  # TODO: Von Supervisor holen
        
        transition = {
            'id': self.frame_count,
            'image_t': img_t_path,
            'action': action,
            'image_t1': img_t1_path,
            'reward': reward
        }
        
        self.transitions.append(transition)
        self.frame_count += 1
        
        if self.frame_count % 10 == 0:
            self.save_dataset()
            self.get_logger().info(f'{self.frame_count} transitions gespeichert')
    
    def save_dataset(self):
        dataset_path = os.path.join(self.session_dir, 'transitions.json')
        with open(dataset_path, 'w') as f:
            json.dump(self.transitions, f, indent=2)

def main():
    rclpy.init()
    node = TransitionCollector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

**Sammle ~500-1000 Transitions:**
```bash
# Lasse Hexapod mit Policy Navigator herumlaufen
ros2 launch hexapod_vision navigation.launch.py

# Starte Transition Collector
ros2 run hexapod_vision transition_collector

# Laufe 10-15 Minuten
```

**Ergebnis:** Dataset mit (s_t, a_t, s_t+1, r_t) Transitionen

---

#### [ ] Task: World Model Training

```python
# train_world_model.py (auf Trainings-PC/Colab)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json
import cv2
import os
from world_model import WorldModel

class TransitionDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        with open(os.path.join(data_dir, 'transitions.json'), 'r') as f:
            self.transitions = json.load(f)
        
        self.data_dir = data_dir
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.action_to_idx = {
            'forward': [1, 0, 0, 0],
            'turn_left': [0, 1, 0, 0],
            'turn_right': [0, 0, 1, 0],
            'stop': [0, 0, 0, 1]
        }
    
    def __len__(self):
        return len(self.transitions)
    
    def __getitem__(self, idx):
        trans = self.transitions[idx]
        
        # Load images
        img_t = cv2.imread(os.path.join(self.data_dir, 'images', trans['image_t']))
        img_t1 = cv2.imread(os.path.join(self.data_dir, 'images', trans['image_t1']))
        
        img_t = cv2.cvtColor(img_t, cv2.COLOR_BGR2RGB)
        img_t1 = cv2.cvtColor(img_t1, cv2.COLOR_BGR2RGB)
        
        # Transform
        img_t = self.transform(img_t)
        img_t1 = self.transform(img_t1)
        
        # Action
        action = torch.tensor(self.action_to_idx[trans['action']], dtype=torch.float32)
        
        # Reward
        reward = torch.tensor([trans['reward']], dtype=torch.float32)
        
        return img_t, action, img_t1, reward

def train_world_model(data_dir, epochs=100, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training auf: {device}')
    
    # Dataset
    dataset = TransitionDataset(data_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Model
    model = WorldModel(latent_dim=256, action_dim=4).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for img_t, action, img_t1, reward in train_loader:
            img_t = img_t.to(device)
            action = action.to(device)
            img_t1 = img_t1.to(device)
            reward = reward.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            z_current, z_next_pred, reward_pred, _ = model(img_t, action)
            
            # Get actual next latent (target)
            with torch.no_grad():
                z_next_actual = model.encoder(img_t1)
            
            # Losses
            latent_loss = F.mse_loss(z_next_pred, z_next_actual)
            reward_loss = F.mse_loss(reward_pred, reward)
            
            loss = latent_loss + 0.1 * reward_loss  # Gewichtung
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for img_t, action, img_t1, reward in val_loader:
                img_t = img_t.to(device)
                action = action.to(device)
                img_t1 = img_t1.to(device)
                reward = reward.to(device)
                
                z_current, z_next_pred, reward_pred, _ = model(img_t, action)
                z_next_actual = model.encoder(img_t1)
                
                latent_loss = F.mse_loss(z_next_pred, z_next_actual)
                reward_loss = F.mse_loss(reward_pred, reward)
                loss = latent_loss + 0.1 * reward_loss
                
                val_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'  Val Loss: {val_loss/len(val_loader):.4f}')
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'world_model_epoch_{epoch+1}.pth')
    
    # Export für Deployment
    model.eval()
    traced = torch.jit.trace(model, (
        torch.randn(1, 3, 224, 224).to(device),
        torch.randn(1, 4).to(device)
    ))
    traced.save('world_model_deployment.pt')
    print('Deployment-Modell gespeichert!')

if __name__ == '__main__':
    train_world_model('./transitions_data')
```

**Training starten:**
```bash
# Auf GPU-PC/Colab
python train_world_model.py

# Erwartete Dauer: 2-4 Stunden für 100 Epochen
```

**Ergebnis:** Trainiertes World Model

---

### 3.3 Task Supervisor (Heiß/Kalt System)

#### [ ] Task: Supervisor Node implementieren

```python
# task_supervisor.py (auf Pi)
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import Float32, String

class TaskSupervisor(Node):
    def __init__(self):
        super().__init__('task_supervisor')
        
        # Parameter
        self.declare_parameter('target_class', 'sports ball')  # COCO class für Ball
        
        self.target_class = self.get_parameter('target_class').value
        
        # Subscriber
        self.detection_sub = self.create_subscription(
            Detection2DArray, '/detections', self.detection_callback, 10)
        
        # Publisher
        self.reward_pub = self.create_publisher(Float32, '/task_reward', 10)
        self.status_pub = self.create_publisher(String, '/task_status', 10)
        
        # State
        self.prev_ball_size = 0.0
        
        # YOLO COCO classes (Klasse 32 = sports ball)
        self.target_class_id = 32
        
        self.get_logger().info(f'Task Supervisor gestartet. Ziel: {self.target_class}')
    
    def detection_callback(self, msg):
        # Finde Ziel-Objekt
        target_detections = [
            d for d in msg.detections
            if int(d.results[0].hypothesis.class_id) == self.target_class_id
        ]
        
        if len(target_detections) == 0:
            # Kein Ball sichtbar
            reward = -0.1  # Kleine Strafe
            status = "COLD - Kein Ball sichtbar"
        else:
            # Ball gefunden - berechne Größe (Proxy für Distanz)
            ball = max(target_detections, key=lambda d: d.bbox.size_x * d.bbox.size_y)
            ball_size = ball.bbox.size_x * ball.bbox.size_y
            
            # Normalisiere auf Bildgröße
            img_size = 640 * 480
            ball_size_norm = ball_size / img_size
            
            # Reward basierend auf Größen-Änderung
            if ball_size_norm > self.prev_ball_size:
                reward = 1.0  # HOT - Ball wird größer (näher)
                status = f"HOT! Ball Größe: {ball_size_norm:.3f}"
            else:
                reward = -0.5  # COLD - Ball wird kleiner (weiter weg)
                status = f"Cold. Ball Größe: {ball_size_norm:.3f}"
            
            self.prev_ball_size = ball_size_norm
            
            # Bonus wenn Ball sehr nah
            if ball_size_norm > 0.3:  # >30% des Bildes
                reward += 2.0
                status = "SUCCESS! Ball erreicht!"
        
        # Publish reward
        reward_msg = Float32()
        reward_msg.data = float(reward)
        self.reward_pub.publish(reward_msg)
        
        status_msg = String()
        status_msg.data = status
        self.status_pub.publish(status_msg)
        
        self.get_logger().info(status)

def main():
    rclpy.init()
    node = TaskSupervisor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

**Ergebnis:** Supervisor gibt "heiß/kalt" Feedback

---

#### [ ] Task: World Model + Supervisor Integration

**Navigator mit World Model:**
```python
# world_model_navigator.py (auf Pi)
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import torch
import torchvision.transforms as transforms

class WorldModelNavigator(Node):
    def __init__(self):
        super().__init__('world_model_navigator')
        
        # World Model laden
        self.device = torch.device('cpu')
        self.model = torch.jit.load('world_model_deployment.pt', map_location=self.device)
        self.model.eval()
        
        self.bridge = CvBridge()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Subscriber
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        
        self.reward_sub = self.create_subscription(
            Float32, '/task_reward', self.reward_callback, 10)
        
        # Action mapping
        self.actions = [
            torch.tensor([[1, 0, 0, 0]], dtype=torch.float32),  # forward
            torch.tensor([[0, 1, 0, 0]], dtype=torch.float32),  # left
            torch.tensor([[0, 0, 1, 0]], dtype=torch.float32),  # right
            torch.tensor([[0, 0, 0, 1]], dtype=torch.float32),  # stop
        ]
        self.action_names = ['forward', 'turn_left', 'turn_right', 'stop']
        
        # State
        self.last_reward = 0.0
        
        self.get_logger().info('World Model Navigator gestartet!')
    
    def reward_callback(self, msg):
        self.last_reward = msg.data
    
    def image_callback(self, msg):
        # Bild vorbereiten
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(cv_image).unsqueeze(0)
        
        # Latent state
        with torch.no_grad():
            z_current = self.model.encoder(img_tensor)
        
        # Imagine future mit verschiedenen Actions
        best_action_idx = 0
        best_predicted_reward = -float('inf')
        
        for i, action in enumerate(self.actions):
            with torch.no_grad():
                # Predict next latent
                z_next = self.model.dynamics(z_current, action)
                
                # Predict reward
                predicted_reward = self.model.reward_predictor(z_next).item()
                
                if predicted_reward > best_predicted_reward:
                    best_predicted_reward = predicted_reward
                    best_action_idx = i
        
        # Führe beste Action aus
        chosen_action = self.action_names[best_action_idx]
        self.execute_action(chosen_action)
        
        self.get_logger().info(
            f'Action: {chosen_action}, Predicted Reward: {best_predicted_reward:.3f}'
        )
    
    def execute_action(self, action):
        # TODO: Sende zu Hexapod Controller
        pass

def main():
    rclpy.init()
    node = WorldModelNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

**Ergebnis:** Navigator nutzt World Model für Planung

---

## 📊 Phase 3 Abschluss

### Erfolgskriterien:
- [x] World Model trainiert
- [x] Supervisor gibt sinnvolles Feedback
- [x] Navigator plant mit World Model
- [x] Hexapod zeigt verbessertes Verhalten

### Hardware-Finale Entscheidung:

**Nach Phase 3 Test:**
```bash
# Performance messen auf Pi 5
ros2 topic hz /policy_action
# Erwarte: 5-15 Hz

# Latenz messen
ros2 topic delay /camera/image_raw /policy_action
# Erwarte: <200ms
```

**Entscheidung:**
- **Performance OK?** → ✅ Bleibe bei Pi 5
- **Performance grenzwertig?** → 🚀 Upgrade zu Jetson Orin Nano (250€)
  - 3-5x schnellere Inferenz
  - Kann größere Modelle
  - Kann on-device Continual Learning

---
---

# Phase 4: Multi-Task Learning (Optional)

## 🎯 Ziele
- Mehrere Tasks ("Ball finden", "Person folgen", "Explorieren")
- Goal Embeddings
- Task-agnostisches Lernen
- Meta-Learning für schnelle Adaption

## 💻 Hardware-Empfehlung

**Für Phase 4 DRINGEND empfohlen:**
- **Jetson Orin Nano** (250€) oder besser
- Grund: Multi-Task benötigt mehr Rechenleistung
- Alternative: Starker Desktop GPU für Training + Pi 5 für Simple Deployment

---

## ✅ TODO-Liste Phase 4

### 4.1 Goal Embeddings

#### [ ] Task: Goal Encoder implementieren
```python
class GoalEncoder(nn.Module):
    def __init__(self, num_tasks=5, embedding_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(num_tasks, embedding_dim)
        
        # Task definitions
        self.tasks = {
            'find_ball': 0,
            'find_person': 1,
            'explore': 2,
            'go_home': 3,
            'avoid_obstacles': 4
        }
    
    def forward(self, task_name):
        task_id = self.tasks[task_name]
        return self.embedding(torch.tensor([task_id]))
```

#### [ ] Task: Multi-Task World Model
```python
class MultiTaskWorldModel(nn.Module):
    def __init__(self, latent_dim=256, action_dim=4, goal_dim=128):
        super().__init__()
        self.encoder = VisualEncoder(latent_dim)
        self.goal_encoder = GoalEncoder(embedding_dim=goal_dim)
        
        # Dynamics konditioniert auf Goal
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim + goal_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
        
        # Task-spezifische Reward Predictor
        self.reward_predictor = nn.ModuleDict({
            'find_ball': RewardPredictor(latent_dim),
            'find_person': RewardPredictor(latent_dim),
            # ...
        })
```

---

### 4.2 Continual Learning

#### [ ] Task: On-Device Learning
```python
class ContinualLearner(Node):
    """Lernt kontinuierlich während Hexapod läuft"""
    def __init__(self):
        super().__init__('continual_learner')
        
        self.model = WorldModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5)
        
        # Experience Replay Buffer
        self.buffer = []
        self.buffer_size = 1000
    
    def update_callback(self, transition):
        # Füge Transition zu Buffer hinzu
        self.buffer.append(transition)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        
        # Mini-batch update
        if len(self.buffer) >= 32:
            batch = random.sample(self.buffer, 32)
            loss = self.compute_loss(batch)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
```

---

## 📊 Phase 4 Abschluss

**Erfolgskriterien:**
- [x] Hexapod kann zwischen Tasks wechseln
- [x] Lernt neue Tasks schneller (Transfer Learning)
- [x] Kontinuierliches On-Device Learning funktioniert

---
---

# 📈 Zusammenfassung & Hardware-Roadmap

## Hardware-Investitions-Strategie

### **Minimaler Start** (0€)
```
✅ Phase 1-2: Raspberry Pi 5
- Objekterkennung ✅
- Einfache Navigation ✅
- Training auf vorhandenem PC/Colab
```

### **Empfohlener Aufbau** (250€)
```
✅ Phase 1-2: Raspberry Pi 5
🚀 Phase 3+: Jetson Orin Nano
- Bessere Performance
- On-Device Training
- Langzeit-Investition für ML
```

### **Maximaler Aufbau** (400-500€)
```
✅ Phase 1-2: Raspberry Pi 5
🚀 Phase 3-4: Jetson Orin NX / Xavier NX
- Deutlich mehr Power
- Größere Modelle
- Multi-Task Learning
```

---

## Erwartete Zeitplanung

| Phase | Dauer | Kumulativ |
|-------|-------|-----------|
| Phase 1: Basis | 1-2 Wochen | 2 Wochen |
| Phase 2: Hybrid Nav | 2-3 Wochen | 5 Wochen |
| Phase 3: World Model | 1-2 Monate | 3 Monate |
| Phase 4: Multi-Task | 2-3 Monate | 6 Monate |

**Realistisches Projekt-Ende:** 3-6 Monate

---

## Hardware-Entscheidungs-Checkpoints

### **Checkpoint 1: Nach Phase 1**
- ✅ Funktioniert YOLO? → Weiter mit Pi 5
- ❌ FPS <3? → Erwäge Jetson schon jetzt

### **Checkpoint 2: Nach Phase 2**
- ✅ Navigation zufriedenstellend? → Pi 5 OK
- ⚠️ Latenz nervt? → Plane Jetson für Phase 3

### **Checkpoint 3: Vor Phase 3**
- **Hast du GPU-PC?** → Trainiere dort, teste Pi 5 Deployment
- **Kein GPU?** → **JETZT Jetson kaufen!**

### **Checkpoint 4: Vor Phase 4**
- Phase 4 ohne Jetson = sehr schwierig
- **Entscheidung:** Jetson oder Phase 4 skippen

---

## Projekt-Tracking

### Aktueller Status
- [ ] Phase 1 abgeschlossen
- [ ] Phase 2 abgeschlossen
- [ ] Phase 3 abgeschlossen
- [ ] Phase 4 abgeschlossen

### Hardware Status
- [x] Raspberry Pi 5 8GB
- [ ] Jetson Orin Nano
- [ ] GPU-Trainings-PC verfügbar: [ ] Ja [ ] Nein

### Performance Metriken
```markdown
| Metrik | Ziel | Ist | Status |
|--------|------|-----|--------|
| YOLO FPS | >=5 | ___ | ⏳ |
| Total Latenz | <200ms | ___ | ⏳ |
| Navigation Erfolgsrate | >80% | ___ | ⏳ |
| World Model FPS | >=10 | ___ | ⏳ |
```

---

## 📚 Weitere Ressourcen

### Papers zum Lesen
1. **World Models** - Ha & Schmidhuber (2018)
2. **Dream to Control: Learning Behaviors by Latent Imagination** - Hafner et al. (2020)
3. **PlaNet: A Deep Planning Network for Reinforcement Learning** (2019)

### Code-Repositories
- **YOLOv8:** https://github.com/ultralytics/ultralytics
- **DreamerV3:** https://github.com/danijar/dreamerv3
- **ROS2 Examples:** https://github.com/ros2/examples

### Communities
- ROS Discourse: https://discourse.ros.org
- Jetson Forums: https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems

---

## 🎯 Nächste Schritte

1. [ ] Starte mit Phase 1 Task 1.1
2. [ ] Richte GitHub Repository ein für Code
3. [ ] Dokumentiere Progress in diesem Markdown
4. [ ] Bei Fragen: Community fragen oder zurück zu mir!

**Viel Erfolg mit deinem Hexapod-Projekt! 🚀🤖**
