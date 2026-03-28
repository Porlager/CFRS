import threading
from datetime import datetime
from pathlib import Path

import pytest
import requests
from werkzeug.serving import make_server

from dashboard_api import create_app


@pytest.fixture
def live_server(tmp_path):
    known_faces_dir = tmp_path / "known_faces"
    app = create_app(
        {
            "TESTING": True,
            "DB_PATH": str(tmp_path / "e2e.db"),
            "DASHBOARD_REFRESH_SEC": 7,
            "KNOWN_FACES_PATH": str(known_faces_dir),
            "CAMERA_FRAME_PATH": str(tmp_path / "latest_frame.jpg"),
        }
    )

    server = make_server("127.0.0.1", 0, app)
    host, port = server.socket.getsockname()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://{host}:{port}"
    try:
        yield base_url
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def sample_payload():
    now = datetime.now().isoformat(timespec="seconds")
    return {
        "timestamp": now,
        "person_count": 4,
        "students": [
            {
                "track_id": 1,
                "name": "SOMCHAI JAIDEE",
                "student_code": "66051281",
                "state": "ตั้งใจเรียน",
                "confirmed": True,
                "confidence": 97.0,
            },
            {
                "track_id": 2,
                "name": "SUDAPORN DEEDEE",
                "student_code": "66051282",
                "state": "หลับ/เหม่อ",
                "confirmed": True,
                "confidence": 91.4,
            },
            {
                "track_id": 3,
                "name": "SOMYING MUNGMEE",
                "student_code": "66051283",
                "state": "ไม่ตั้งใจเรียน",
                "confirmed": False,
                "confidence": 74.3,
            },
            {
                "track_id": 4,
                "name": "คนแปลกหน้า",
                "state": "ไม่ทราบสถานะ",
                "confirmed": False,
                "confidence": 0.0,
            },
        ],
    }


def test_dashboard_page_renders_responsive(live_server):
    response = requests.get(f"{live_server}/", timeout=5)
    assert response.status_code == 200
    html = response.text
    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in html
    assert "แดชบอร์ดติดตามการเรียน" in html
    assert "window-days" in html


def test_ingest_result_then_report_e2e(live_server):
    ingest = requests.post(f"{live_server}/api/result", json=sample_payload(), timeout=5)
    assert ingest.status_code == 200
    body = ingest.json()
    assert body["ok"] is True
    assert body["attendance_saved"] == 2
    assert body["behavior_saved"] == 4

    today = requests.get(f"{live_server}/api/reports/today", timeout=5)
    assert today.status_code == 200
    today_data = today.json()
    assert today_data["ok"] is True
    assert today_data["attendance_total"] >= 2
    assert today_data["drowsy_event_total"] >= 1

    dashboard = requests.get(f"{live_server}/api/reports/dashboard?days=7", timeout=5)
    assert dashboard.status_code == 200
    dashboard_data = dashboard.json()
    assert dashboard_data["ok"] is True
    assert "kpis" in dashboard_data
    assert "trend" in dashboard_data
    assert "runtime" in dashboard_data
    assert "latest_checkin" in dashboard_data
    assert "attendance_today_list" in dashboard_data
    assert len(dashboard_data["trend"]) == 7
    assert isinstance(dashboard_data["attendance_today_list"], list)
    assert len(dashboard_data["attendance_today_list"]) >= 2

    latest = dashboard_data["latest_checkin"]
    assert latest is not None
    assert "full_name" in latest
    assert "first_seen_at" in latest

    kpis = dashboard_data["kpis"]
    assert "total_active_students" in kpis
    assert "attendance_rate_today" in kpis
    assert isinstance(kpis["total_active_students"], int)
    assert isinstance(kpis["attendance_rate_today"], float)

    runtime = dashboard_data["runtime"]
    assert isinstance(runtime["current_students"], list)
    assert len(runtime["current_students"]) >= 4
    assert runtime["attentive_count"] == 1
    assert runtime["inattentive_count"] == 1
    assert runtime["drowsy_count"] == 1
    assert runtime["unknown_count"] == 1
    assert isinstance(runtime["state_counts"], dict)
    assert runtime["state_counts"]["ไม่ตั้งใจเรียน"] == 1
    first_runtime_student = runtime["current_students"][0]
    assert "name" in first_runtime_student
    assert "student_code" in first_runtime_student
    assert "state" in first_runtime_student


def test_auto_save_duplicate_day_e2e(live_server):
    payload = {
        "full_name": "SOMCHAI JAIDEE",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "confidence": 95.0,
        "track_id": 1,
    }

    first = requests.post(f"{live_server}/api/attendance/auto-save", json=payload, timeout=5)
    assert first.status_code == 201
    first_data = first.json()
    assert first_data["ok"] is True
    assert first_data["saved"] is True

    second = requests.post(f"{live_server}/api/attendance/auto-save", json=payload, timeout=5)
    assert second.status_code == 200
    second_data = second.json()
    assert second_data["ok"] is True
    assert second_data["saved"] is False
    assert second_data["reason"] == "already_saved_today"


def test_register_face_endpoint_saves_file(live_server):
    files = {
        "photo": ("sample.jpg", b"fake-jpeg-data", "image/jpeg"),
    }
    data = {"full_name": "SOMCHAI JAIDEE", "student_code": "66051281"}

    resp = requests.post(f"{live_server}/api/register-face", data=data, files=files, timeout=5)
    assert resp.status_code == 201
    payload = resp.json()
    assert payload["ok"] is True
    assert payload["full_name"] == "SOMCHAI JAIDEE"
    assert payload["student_code"] == "66051281"
    assert payload["saved_file"].startswith("SOMCHAI_JAIDEE_")

    saved_path = Path(payload["saved_path"])
    assert saved_path.exists()

    students_resp = requests.get(f"{live_server}/api/students", timeout=5)
    assert students_resp.status_code == 200
    students_payload = students_resp.json()
    assert students_payload["ok"] is True
    match = [s for s in students_payload["students"] if s["full_name"] == "SOMCHAI JAIDEE"]
    assert len(match) == 1
    assert match[0]["student_code"] == "66051281"


def test_register_face_requires_student_code(live_server):
    files = {
        "photo": ("sample.jpg", b"fake-jpeg-data", "image/jpeg"),
    }
    data = {"full_name": "SOMCHAI JAIDEE"}

    resp = requests.post(f"{live_server}/api/register-face", data=data, files=files, timeout=5)
    assert resp.status_code == 400
    payload = resp.json()
    assert payload["ok"] is False
    assert payload["error"] == "student_code_required"
