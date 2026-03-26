from datetime import date

from flask import Flask, jsonify, request

from backend_db import AttendanceRepository

app = Flask(__name__)
repo = AttendanceRepository(db_path="data/cfrs.db")


@app.get("/health")
def health() -> tuple:
    return jsonify({"ok": True, "service": "cfrs-dashboard-api"}), 200


@app.post("/api/result")
def ingest_result() -> tuple:
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"ok": False, "error": "invalid_or_missing_json"}), 400

    result = repo.ingest_ai_payload(payload)
    return jsonify(result), 200


@app.post("/api/attendance/auto-save")
def auto_save_attendance() -> tuple:
    body = request.get_json(silent=True) or {}
    full_name = body.get("full_name")
    if not full_name:
        return jsonify({"ok": False, "error": "full_name_required"}), 400

    result = repo.auto_save_attendance(
        full_name=full_name,
        timestamp=body.get("timestamp"),
        confidence=body.get("confidence"),
        track_id=body.get("track_id"),
        source="manual-api",
    )

    status = 201 if result.get("saved") else 200
    return jsonify({"ok": True, **result}), status


@app.get("/api/students")
def students() -> tuple:
    limit = request.args.get("limit", default=200, type=int)
    data = repo.list_students(limit=max(1, min(limit, 1000)))
    return jsonify({"ok": True, "total": len(data), "students": data}), 200


@app.get("/api/reports/today")
def report_today() -> tuple:
    data = repo.get_today_report()
    return jsonify({"ok": True, **data}), 200


@app.get("/api/reports/behavior")
def report_behavior() -> tuple:
    today = date.today().isoformat()
    start_date = request.args.get("start_date", default=today, type=str)
    end_date = request.args.get("end_date", default=today, type=str)

    data = repo.get_behavior_report(start_date=start_date, end_date=end_date)
    return jsonify({"ok": True, **data}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
