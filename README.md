# Classroom Monitoring System

Real-time classroom monitoring with three pipelines running on one laptop:

- Identity: OpenCV LBPH face recognizer
- Behavior: MediaPipe Face Landmarker + EAR + head pose
- Counting: YOLOv8n person detection with frame skipping for CPU stability

## What This Version Fixes

- Drowsy status now requires sustained bad behavior duration (not single-frame trigger)
- Tracker uses one-to-one assignment per frame to reduce ID collisions
- Runtime failures in training/inference are logged instead of silently ignored
- Portable launcher script (`run.bat`) works across different machines

## Project Structure

```
CFRS/
├── main.py
├── run.bat
├── README.md
├── requirements.txt
├── face_landmarker.task
├── yolov8n.pt
└── known_faces/
```

## Requirements

- Python 3.10+ recommended
- Webcam
- Windows/Linux/macOS

Install dependencies:

```bash
pip install -r requirements.txt
```

Important:

- This project uses `cv2.face.LBPHFaceRecognizer_create`, so you must install `opencv-contrib-python`.
- If you only install `opencv-python`, identity recognition will be disabled.

## Quick Start

1. Add student images to `known_faces/`.
2. Use file naming format `NAME_xxx.jpg` or `NAME_xxx.png`.
3. Run:

```bash
run.bat
```

or

```bash
python main.py
```

Press `q` to quit.

## Core Runtime Logic

### 1) Behavior Classification

- EAR from eye landmarks checks eye closure.
- Head pose (pitch/yaw/roll) checks looking-away posture.
- Track is marked drowsy only after sustained duration (`DROWSY_HOLD_TIME`).

### 2) Identity

- LBPH is trained from `known_faces/` at startup.
- If LBPH is unavailable or not trained, system keeps running in detection-only mode.

### 3) People Counting

- YOLOv8n runs every N frames (`yolo_skip_frames`) to reduce CPU usage.
- Last detection result is reused between YOLO runs for smooth display.

## Main Tunables (in `main.py`)

- `CONFIRMATION_TIME`: time before name is marked confirmed
- `TIMEOUT`: tracker remove timeout
- `MAX_DISTANCE`: centroid matching threshold
- `DROWSY_HOLD_TIME`: sustained bad behavior duration before drowsy
- `EAR_THRESH`: eye-closure sensitivity
- `yolo_skip_frames`: YOLO inference interval

## Troubleshooting

- `AttributeError: module 'cv2' has no attribute 'face'`
  - Install: `pip install opencv-contrib-python`

- `face_landmarker.task` missing
  - System auto-downloads on first run if internet is available.

- Low FPS
  - Increase `yolo_skip_frames`
  - Reduce camera resolution in your webcam settings

## Notes

- Thai text is rendered via Pillow for reliable on-frame Thai display.
- This is a real-time heuristic system; tune thresholds for your classroom camera angle.
