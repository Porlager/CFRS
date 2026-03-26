# CFRS Decision Matrix

## Runtime Selection
- Use `main.py` for quick local testing with minimal moving parts.
- Use `stream_main.py` for threaded processing and backend integration.

## Camera Selection
- Use USB webcam when network reliability is unknown.
- Use IP Webcam when camera positioning flexibility is required.

## Symptom To Action
- Start failure with `cv2.face` error:
  - Install `opencv-contrib-python`.
- Model startup delay or missing landmarker file:
  - Allow first-run download of `face_landmarker.task`.
- High latency or drops from phone stream:
  - Lower camera resolution and quality.
  - Prefer hotspot/local network path.
- Many unknown identities:
  - Improve face image quality in `known_faces/` and restart to retrain.

## Stability Thresholds (Practical)
- Stream check healthy target:
  - drop rate near 0%
  - average frame pull latency under ~200 ms
- Demo-safe backend behavior:
  - backend can be offline if capture-only mode is acceptable
