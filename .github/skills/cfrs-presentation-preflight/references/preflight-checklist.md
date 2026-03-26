# Preflight Checklist

## Camera And Network
- Phone and laptop on same network when using IP Webcam.
- IP Webcam server started and reachable.
- Resolution around 640x480 and moderate quality for stability.

## Application Health
- `network_check.py` reports acceptable latency/drop behavior.
- `stream_main.py` or `main.py` opens and renders frame updates.
- No repeated crash loops in startup phase.

## Demo Readiness
- At least one known face recognized and confirmed.
- Unknown face is clearly labeled as unknown.
- Exit via `q` works and restart succeeds.

## If Something Breaks
- Switch to USB webcam if network stream is unstable.
- Disable backend posting if backend is unavailable.
- Prioritize stable visualization over full feature set.
