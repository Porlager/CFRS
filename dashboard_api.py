import atexit
import os
import re
import socket
from datetime import date, datetime

from flask import Flask, Response, jsonify, render_template, request
from werkzeug.exceptions import HTTPException
from backend_db import AttendanceRepository


ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DROWSY_STATES = {"หลับ/เหม่อ", "ฟุบหลับ/หันหลัง", "drowsy", "sleeping"}
INATTENTIVE_STATES = {"ไม่ตั้งใจเรียน", "inattentive", "looking_away"}


def _parse_bounded_int(value: str, default: int, min_value: int, max_value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(min_value, min(parsed, max_value))


def _normalize_person_name(raw: str) -> str:
    return " ".join(str(raw or "").strip().split())


def _classify_state_bucket(state: str) -> str:
    normalized = str(state or "").strip().lower()
    if not normalized:
        return "unknown"
    if any(k in normalized for k in ("หลับ", "ง่วง", "drowsy", "sleep", "ฟุบ", "นอน")):
        return "drowsy"
    if any(k in normalized for k in ("ไม่ตั้งใจ", "inattentive", "away", "looking_away", "หันหลัง")):
        return "inattentive"
    if "ไม่ตั้งใจ" not in normalized and ("ตั้งใจ" in normalized or "attentive" in normalized):
        return "attentive"
    return "unknown"


def _person_name_to_token(name: str) -> str:
    token = re.sub(r"\s+", "_", name.strip().upper())
    token = re.sub(r"[^0-9A-Zก-๙_]", "", token)
    token = token[:60]
    return token or "PERSON"


def _allowed_image_filename(filename: str) -> bool:
    ext = os.path.splitext(filename or "")[1].lower()
    return ext in ALLOWED_IMAGE_EXTENSIONS


def _is_port_available(host: str, port: int) -> bool:
    bind_host = host
    if host in ("0.0.0.0", ""):
        bind_host = "127.0.0.1"

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((bind_host, port))
            return True
        except OSError:
            return False


def _resolve_run_port(host: str, preferred_port: int) -> tuple[int, bool]:
    fallback_range = _parse_bounded_int(os.getenv("CFRS_PORT_FALLBACK_RANGE", "20"), 20, 1, 200)
    if _is_port_available(host, preferred_port):
        return preferred_port, False

    for offset in range(1, fallback_range + 1):
        candidate = preferred_port + offset
        if candidate > 65535:
            break
        if _is_port_available(host, candidate):
            return candidate, True

    raise RuntimeError(
        f"Port {preferred_port} is busy and no free port found in +{fallback_range} range. "
        "Set CFRS_PORT=<free_port> explicitly."
    )


def create_app(test_config=None) -> Flask:
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.getenv("CFRS_SECRET_KEY", "dev-secret-change-me"),
        DB_PATH=os.getenv("CFRS_DB_PATH", "data/cfrs.db"),
        DASHBOARD_REFRESH_SEC=_parse_bounded_int(os.getenv("CFRS_DASH_REFRESH", "12"), 12, 5, 60),
        KNOWN_FACES_PATH=os.getenv("CFRS_KNOWN_FACES_PATH", "known_faces"),
        CAMERA_FRAME_PATH=os.getenv("CFRS_CAMERA_FRAME_PATH", "storage/latest_frame.jpg"),
    )
    if test_config:
        app.config.update(test_config)

    repo = AttendanceRepository(db_path=app.config["DB_PATH"])
    app.extensions["repo"] = repo
    app.extensions["runtime_status"] = {
        "timestamp": None,
        "person_count": 0,
        "attentive_count": 0,
        "inattentive_count": 0,
        "drowsy_count": 0,
        "unknown_count": 0,
        "state_counts": {},
        "current_students": [],
    }

    @atexit.register
    def _shutdown_repo() -> None:
        try:
            repo.close()
        except Exception:
            pass

    def get_repo() -> AttendanceRepository:
        return app.extensions["repo"]

    @app.get("/")
    def dashboard() -> tuple:
        return (
            render_template(
                "dashboard.html",
                refresh_seconds=app.config["DASHBOARD_REFRESH_SEC"],
            ),
            200,
        )

    @app.get("/register")
    def register_face_page() -> tuple:
        return render_template("register.html"), 200

    @app.get("/api/camera/frame")
    def camera_frame() -> tuple:
        frame_path = app.config["CAMERA_FRAME_PATH"]
        if not os.path.exists(frame_path):
            return jsonify({"ok": False, "error": "camera_frame_not_ready"}), 404

        try:
            # Read into memory to close file handle immediately and reduce writer lock contention on Windows.
            with open(frame_path, "rb") as fp:
                frame_bytes = fp.read()
        except OSError:
            return jsonify({"ok": False, "error": "camera_frame_busy"}), 503

        response = Response(frame_bytes, mimetype="image/jpeg")
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    @app.get("/health")
    def health() -> tuple:
        return jsonify({"ok": True, "service": "cfrs-dashboard-api"}), 200

    @app.get("/api/reports/dashboard")
    def report_dashboard() -> tuple:
        days = request.args.get("days", default="7", type=str)
        limit = request.args.get("recent_limit", default="12", type=str)
        data = get_repo().get_dashboard_summary(
            days=_parse_bounded_int(days, 7, 1, 30),
            recent_limit=_parse_bounded_int(limit, 12, 1, 50),
        )
        data["runtime"] = app.extensions["runtime_status"]
        return jsonify({"ok": True, **data}), 200

    @app.get("/api/runtime/status")
    def runtime_status() -> tuple:
        return jsonify({"ok": True, **app.extensions["runtime_status"]}), 200

    @app.post("/api/result")
    def ingest_result() -> tuple:
        payload = request.get_json(silent=True)
        if not payload:
            return jsonify({"ok": False, "error": "invalid_or_missing_json"}), 400

        students = payload.get("students", []) if isinstance(payload, dict) else []
        attentive = 0
        inattentive = 0
        drowsy = 0
        unknown = 0
        state_counts = {}
        current_students = []
        for item in students:
            state = str(item.get("state", "ไม่ทราบสถานะ"))
            state_counts[state] = state_counts.get(state, 0) + 1
            state_bucket = _classify_state_bucket(state)
            if state_bucket == "drowsy":
                drowsy += 1
            elif state_bucket == "inattentive":
                inattentive += 1
            elif state_bucket == "attentive":
                attentive += 1
            else:
                unknown += 1

            raw_name = str(item.get("name", "Unknown")).strip()
            name = raw_name if raw_name else "Unknown"
            student_code = _normalize_person_name(item.get("student_code"))
            if not student_code:
                student_code = None
            confidence_raw = item.get("confidence")
            try:
                confidence = float(confidence_raw) if confidence_raw is not None else None
            except (TypeError, ValueError):
                confidence = None

            current_students.append(
                {
                    "track_id": item.get("track_id"),
                    "name": name,
                    "student_code": student_code,
                    "state": state,
                    "confirmed": bool(item.get("confirmed", False)),
                    "confidence": confidence,
                }
            )

        app.extensions["runtime_status"] = {
            "timestamp": payload.get("timestamp"),
            "person_count": int(payload.get("person_count", len(students))),
            "attentive_count": attentive,
            "inattentive_count": inattentive,
            "drowsy_count": drowsy,
            "unknown_count": unknown,
            "state_counts": state_counts,
            "current_students": current_students,
        }

        result = get_repo().ingest_ai_payload(payload)
        return jsonify(result), 200

    @app.post("/api/attendance/auto-save")
    def auto_save_attendance() -> tuple:
        body = request.get_json(silent=True) or {}
        full_name = body.get("full_name")
        student_code = _normalize_person_name(body.get("student_code"))
        if not full_name and not student_code:
            return jsonify({"ok": False, "error": "full_name_required"}), 400

        result = get_repo().auto_save_attendance(
            full_name=full_name or student_code,
            student_code=student_code,
            timestamp=body.get("timestamp"),
            confidence=body.get("confidence"),
            track_id=body.get("track_id"),
            source="manual-api",
        )

        status = 201 if result.get("saved") else 200
        return jsonify({"ok": True, **result}), status

    @app.post("/api/register-face")
    def register_face() -> tuple:
        full_name = _normalize_person_name(request.form.get("full_name"))
        if not full_name:
            return jsonify({"ok": False, "error": "full_name_required"}), 400

        student_code = _normalize_person_name(request.form.get("student_code"))
        if not student_code:
            return jsonify({"ok": False, "error": "student_code_required"}), 400

        existing_code_owner = get_repo().get_student_by_code(student_code)
        if existing_code_owner and existing_code_owner.get("full_name") != full_name:
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": "student_code_already_exists",
                        "student_code": student_code,
                        "owner": existing_code_owner.get("full_name"),
                    }
                ),
                409,
            )

        photo = request.files.get("photo")
        if photo is None or not photo.filename:
            return jsonify({"ok": False, "error": "photo_required"}), 400

        if not _allowed_image_filename(photo.filename):
            return jsonify(
                {
                    "ok": False,
                    "error": "unsupported_file_type",
                    "allowed": sorted(ALLOWED_IMAGE_EXTENSIONS),
                }
            ), 400

        known_faces_path = app.config["KNOWN_FACES_PATH"]
        os.makedirs(known_faces_path, exist_ok=True)

        ext = os.path.splitext(photo.filename)[1].lower()
        saved_name = f"{_person_name_to_token(full_name)}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}{ext}"
        saved_path = os.path.join(known_faces_path, saved_name)
        photo.save(saved_path)

        student_id = get_repo().upsert_student(full_name=full_name, student_code=student_code)

        return (
            jsonify(
                {
                    "ok": True,
                    "student_id": student_id,
                    "student_code": student_code,
                    "full_name": full_name,
                    "saved_file": saved_name,
                    "saved_path": saved_path,
                    "reload_required": True,
                    "message": "ลงทะเบียนสำเร็จ (ถ้า backend กล้องรันอยู่ ให้รีสตาร์ตเพื่อโหลดใบหน้าใหม่)",
                }
            ),
            201,
        )

    @app.get("/api/students")
    def students() -> tuple:
        limit = request.args.get("limit", default=200, type=int)
        data = get_repo().list_students(limit=max(1, min(limit, 1000)))
        return jsonify({"ok": True, "total": len(data), "students": data}), 200

    @app.get("/api/reports/today")
    def report_today() -> tuple:
        data = get_repo().get_today_report()
        return jsonify({"ok": True, **data}), 200

    @app.get("/api/reports/behavior")
    def report_behavior() -> tuple:
        today = date.today().isoformat()
        start_date = request.args.get("start_date", default=today, type=str)
        end_date = request.args.get("end_date", default=today, type=str)

        data = get_repo().get_behavior_report(start_date=start_date, end_date=end_date)
        return jsonify({"ok": True, **data}), 200

    @app.errorhandler(HTTPException)
    def handle_http_exception(err: HTTPException):
        if request.path.startswith("/api/"):
            return jsonify({"ok": False, "error": err.name.lower().replace(" ", "_")}), err.code
        return render_template("error.html", error_title=err.name, error_message=err.description), err.code

    @app.errorhandler(Exception)
    def handle_exception(err: Exception):
        if request.path.startswith("/api/"):
            return jsonify({"ok": False, "error": "internal_server_error", "message": str(err)}), 500
        return render_template(
            "error.html",
            error_title="Internal Server Error",
            error_message="เกิดข้อผิดพลาดภายในระบบ กรุณาลองใหม่อีกครั้ง",
        ), 500

    return app


app = create_app()


if __name__ == "__main__":
    host = os.getenv("CFRS_HOST", "0.0.0.0")
    preferred_port = _parse_bounded_int(os.getenv("CFRS_PORT", "5000"), 5000, 1, 65535)
    auto_fallback = str(os.getenv("CFRS_PORT_AUTO_FALLBACK", "1")).strip().lower() in ("1", "true", "yes", "on")

    if auto_fallback:
        try:
            port, used_fallback = _resolve_run_port(host, preferred_port)
        except RuntimeError as exc:
            print(f"[{datetime.now()}] ERROR: {exc}")
            raise SystemExit(1)
        if used_fallback:
            print(
                f"[{datetime.now()}] WARN: Port {preferred_port} is busy. "
                f"Using available port {port} (set CFRS_PORT to choose another port)."
            )
    else:
        port = preferred_port

    app.run(host=host, port=port, debug=False)
