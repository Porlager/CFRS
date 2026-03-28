#!/usr/bin/env python3
import argparse
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent


def _parse_bounded_int(value: str, default: int, min_value: int, max_value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(min_value, min(parsed, max_value))


def _is_port_available(host: str, port: int) -> bool:
    bind_host = host if host not in ("0.0.0.0", "") else "127.0.0.1"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((bind_host, port))
            return True
        except OSError:
            return False


def _resolve_port(host: str, preferred_port: int, scan_range: int) -> tuple[int, bool]:
    if _is_port_available(host, preferred_port):
        return preferred_port, False

    for offset in range(1, scan_range + 1):
        candidate = preferred_port + offset
        if candidate > 65535:
            break
        if _is_port_available(host, candidate):
            return candidate, True

    raise RuntimeError(
        f"Port {preferred_port} is busy and no free port found in +{scan_range} range. "
        "Please set --port explicitly."
    )


def _wait_for_health(base_url: str, timeout_sec: float, dashboard_proc: subprocess.Popen) -> bool:
    deadline = time.time() + timeout_sec
    health_url = f"{base_url}/health"
    while time.time() < deadline:
        if dashboard_proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(health_url, timeout=1.8) as resp:
                if resp.status == 200:
                    payload = json.loads(resp.read().decode("utf-8"))
                    if payload.get("ok") is True:
                        return True
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
            pass
        time.sleep(0.35)
    return False


def _terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _check_required_modules(modules: list[str]) -> list[str]:
    missing = []
    for module_name in modules:
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-platform CFRS runner (Linux/macOS/Windows)")
    parser.add_argument("--dashboard-only", action="store_true", help="Run dashboard API only")
    parser.add_argument(
        "--port",
        type=int,
        default=_parse_bounded_int(os.getenv("CFRS_PORT", "5000"), 5000, 1, 65535),
        help="Preferred dashboard port",
    )
    parser.add_argument(
        "--port-fallback-range",
        type=int,
        default=_parse_bounded_int(os.getenv("CFRS_PORT_FALLBACK_RANGE", "20"), 20, 1, 200),
        help="Range to scan for next free port if preferred port is busy",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("CFRS_HOST", "0.0.0.0"),
        help="Dashboard host binding",
    )
    parser.add_argument(
        "--health-timeout",
        type=float,
        default=8.0,
        help="Seconds to wait for dashboard health check in full-stack mode",
    )
    args = parser.parse_args()

    required = ["flask", "requests"] if args.dashboard_only else ["flask", "requests", "cv2", "face_recognition"]
    missing_modules = _check_required_modules(required)
    if missing_modules:
        print(
            "[CFRS] Missing dependencies: "
            + ", ".join(missing_modules)
            + f"\n[CFRS] Install with: {sys.executable} -m pip install -r requirements.txt"
        )
        return 1

    try:
        dashboard_port, used_fallback = _resolve_port(args.host, args.port, args.port_fallback_range)
    except RuntimeError as exc:
        print(f"[CFRS] ERROR: {exc}")
        return 1

    env_dashboard = os.environ.copy()
    env_dashboard["CFRS_HOST"] = args.host
    env_dashboard["CFRS_PORT"] = str(dashboard_port)
    env_dashboard["CFRS_PORT_AUTO_FALLBACK"] = "0"

    if used_fallback:
        print(f"[CFRS] WARN: Port {args.port} busy -> using {dashboard_port}")

    dashboard_url = f"http://127.0.0.1:{dashboard_port}"
    print(f"[CFRS] Dashboard URL: {dashboard_url}")
    print(f"[CFRS] Register URL: {dashboard_url}/register")

    dashboard_proc = subprocess.Popen(
        [sys.executable, "dashboard_api.py"],
        cwd=str(ROOT_DIR),
        env=env_dashboard,
    )

    if args.dashboard_only:
        try:
            return dashboard_proc.wait()
        except KeyboardInterrupt:
            _terminate_process(dashboard_proc)
            return 130

    ready = _wait_for_health(dashboard_url, args.health_timeout, dashboard_proc)
    if not ready:
        print("[CFRS] WARN: Dashboard health check timeout, will continue starting camera backend.")
    else:
        print("[CFRS] Dashboard health check OK.")

    env_camera = os.environ.copy()
    env_camera.setdefault("CFRS_BACKEND_INGEST_URL", f"{dashboard_url}/api/result")
    env_camera.setdefault("CFRS_FRAME_RESIZE_SCALE", "0.36")
    env_camera.setdefault("CFRS_PROCESS_EVERY_N_FRAMES", "5")
    env_camera.setdefault("CFRS_FACE_DETECTION_COOLDOWN_FRAMES", "10")
    env_camera.setdefault("CFRS_NO_FACE_INTERVAL_BOOST_FRAMES", "3")
    env_camera.setdefault("CFRS_RUNTIME_NUM_JITTERS", "0")
    env_camera.setdefault("CFRS_BODY_DETECT_EVERY_N_FRAMES", "7")
    env_camera.setdefault("CFRS_BODY_RESIZE_SCALE", "0.35")
    env_camera.setdefault("CFRS_FACE_MISSING_BODY_GRACE_FRAMES", "2")
    env_camera.setdefault("CFRS_PREVIEW_DOWNSCALE", "0.72")
    env_camera.setdefault("CFRS_CV_THREADS", "2")
    env_camera.setdefault("CFRS_CAMERA_WIDTH", "640")
    env_camera.setdefault("CFRS_CAMERA_HEIGHT", "480")
    env_camera.setdefault("CFRS_FRAME_WRITE_INTERVAL_SEC", "0.45")

    print("[CFRS] Starting camera backend (press 'q' on camera window to stop).")
    try:
        backend_exit = subprocess.run(
            [sys.executable, "main.py"],
            cwd=str(ROOT_DIR),
            env=env_camera,
            check=False,
        ).returncode
    except KeyboardInterrupt:
        backend_exit = 130
    finally:
        _terminate_process(dashboard_proc)

    return backend_exit


if __name__ == "__main__":
    raise SystemExit(main())
