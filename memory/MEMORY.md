# MEMORY.md - Hexapod Brain Project

## Projekt Taget
The Taget of the Project is defined in "docu/hexapod_vision_projekt.md".
Edit this file, if the targets are changing.

## Servers
1.  **Hardware Node (`raspclaws-1`)**:
    *   Running `Adeept_RaspClaws` code.
    *   Provides Hardware Abstraction & ROS2 Actions (`LinearMove`, `Rotate`, `HeadPosition`).
    *   Publishes Camera Stream.

2.  **Compute Node (`ubuntu1`)**:
    *   **THIS PROJECT (`Hexapod_Brain`)**.
    *   Runs on Raspberry Pi 5 (Ubuntu + ROS2 Humble).
    *   Responsible for Perception (YOLO), Logic, and Navigation.

## 📂 Project Structure (Expected)
*   `src/hexapod_vision`: ROS2 Package for YOLOv8 Object Detection.
*   `src/hexapod_navigation`: (Planned) Logic & Navigation Node.
*   `Docu/hexapod_vision_projekt.md`: **MASTER PLAN** for this project (Copied from previous repo).

## 🔗 Important References
*   See `Docu/hexapod_vision_projekt.md` for detailed roadmap.
*   See `BRAIN_README.md` (or `README.md`) for setup instructions.
