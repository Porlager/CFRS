---
name: cfrs-presentation-preflight
description: 'Prepare CFRS for a live presentation. Use for network preflight, IP Webcam readiness, backend health checks, and final go/no-go validation before demo.'
argument-hint: 'minutes-before-demo=<30|10|5> camera=<ip-webcam|usb> backend=<on|off>'
---

# CFRS Presentation Preflight

## What This Skill Produces
- A timed checklist for a reliable live demo.
- A go/no-go decision based on measured latency and drop rate.
- A fallback plan if backend or network is unstable.

## When to Use
- Before classroom demo, advisor review, or project presentation.
- Any time the camera source changes or Wi-Fi environment changes.

## Procedure

### 1. T-30 Minutes: Environment Lock-In
1. Pick camera source and do not switch again unless required.
2. If using IP Webcam:
   - Ensure phone and laptop are on same hotspot/network.
   - Start IP Webcam server.
3. Close unnecessary CPU-heavy applications.

### 2. T-10 Minutes: Network And Stream Verification
1. Run `python network_check.py`.
2. Evaluate output:
   - If stream cannot open, fix URL/network first.
   - If drop rate is high, lower camera resolution/quality.
   - If latency is high, move devices closer or change network.

### 3. T-5 Minutes: Full Pipeline Smoke Test
1. Run `python stream_main.py` (or `python main.py` if using local mode only).
2. Confirm:
   - Window renders continuously.
   - Person count changes with movement.
   - Behavior labels appear.
   - Known face confirms within expected delay.
3. Press `q` and restart once to verify clean recovery.

### 4. Go/No-Go Rules
1. GO if:
   - Stream opens reliably.
   - Low drop behavior is observed.
   - No recurring runtime errors in first 60 seconds.
2. NO-GO if:
   - Camera repeatedly disconnects.
   - Severe lag persists after one resolution reduction.
   - App cannot run stably for 1 minute.

### 5. Fallback Strategy
1. If IP Webcam is unstable, switch to USB webcam path.
2. If backend is unstable, continue capture-only demo with backend disabled.
3. Keep one known face sample ready for quick confirmation test.

## Completion Checks
- One successful 60-second run without critical warnings.
- Known identity and unknown identity both handled as expected.
- Operator can start/stop quickly with predictable behavior.

## References
- [Preflight Checklist](./references/preflight-checklist.md)
