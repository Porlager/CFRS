---
name: cfrs-operations
description: 'Run, validate, and troubleshoot the CFRS classroom monitoring system. Use for setup, quick start, webcam vs IP Webcam decisions, low-FPS tuning, LBPH identity issues, and backend connectivity checks.'
argument-hint: 'goal=<quick-start|preflight|troubleshoot|tune> source=<usb|ip-webcam> mode=<main|stream>'
---

# CFRS Operations

## What This Skill Produces
- A repeatable runbook to start CFRS from a clean machine.
- A decision-based troubleshooting path for camera, model, identity, and backend problems.
- A stable runtime configuration with explicit completion checks.

## When to Use
- You need to run CFRS on a new laptop.
- `main.py` or `stream_main.py` fails to start.
- Detection works but identity or behavior output is unstable.
- FPS is low, lag is high, or stream drops occur.

## Inputs To Confirm First
- Preferred runtime mode: `main.py` (single-loop) or `stream_main.py` (threaded).
- Camera source: local USB webcam or IP Webcam on phone.
- Whether backend posting is required right now.

## Procedure

### 1. Baseline Setup
1. Confirm Python 3.10+ is available.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Prefer `opencv-contrib-python` for LBPH identity support.
4. Confirm assets exist in repo root:
   - `yolov8n.pt`
   - `face_landmarker.task` (auto-download is supported if internet is available)

### 2. Prepare Identity Data
1. Put face images in `known_faces/`.
2. Use naming pattern `NAME_xxx.jpg` or `NAME_xxx.png`.
3. Keep images reasonably sharp and front-facing.

### 3. Choose Runtime Path
1. If you want the simplest launch path, run:
   - `run.bat` or `python main.py`
2. If you want threaded pipeline + optional backend sending, run:
   - `python stream_main.py`

### 4. Branch By Camera Source
1. USB webcam path:
   - Use camera index in `stream_main.py` when needed.
2. IP Webcam path:
   - Set `USE_IP_WEBCAM = True` and correct `IPWEBCAM_URL` in `stream_main.py`.
   - Run `python network_check.py` before live session.

### 5. Validate Runtime Health
1. Confirm video window opens and updates smoothly.
2. Confirm person count appears and changes with scene.
3. Confirm behavior labels update (`attentive` vs drowsy states).
4. Confirm known faces become confirmed after hold time.
5. Quit cleanly with `q`.

### 6. Troubleshooting Decision Points
1. If `cv2.face` is missing:
   - Install `opencv-contrib-python`, then retry.
2. If camera cannot open:
   - Verify source mode and URL/index.
   - For IP Webcam, ensure same network and app server started.
3. If identity stays unknown:
   - Verify `known_faces/` naming format and image quality.
   - Ensure LBPH training had valid crops at startup.
4. If FPS is low:
   - Increase `yolo_skip_frames` in `main.py`.
   - Reduce input resolution in camera source.
5. If backend fails:
   - Toggle `BACKEND_ENABLED` based on environment.
   - Validate backend health endpoint.

### 7. Tune Safely (One Knob At A Time)
1. Start with `yolo_skip_frames` for CPU relief.
2. Adjust `EAR_THRESH` and drowsy hold timing only after visual checks.
3. Adjust confirmation and timeout values last.
4. Re-test with at least one known face and one unknown subject.

## Completion Checks
- App runs for 2+ minutes without crash.
- Detection and labeling are stable on moving subjects.
- Known face confirmation appears consistently.
- If backend enabled, payload posts without repeated failures.

## References
- [Decision Matrix](./references/decision-matrix.md)
