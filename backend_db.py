import os
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

UNKNOWN_NAMES = {
    "",
    "UNKNOWN",
    "คนแปลกหน้า",
    "ภาพเบลอ",
    "ภาพขยับ/เบลอ",
    "ไม่พบใบหน้า",
}

DROWSY_STATES = {
    "หลับ/เหม่อ",
    "ฟุบหลับ/หันหลัง",
    "drowsy",
    "sleeping",
}


class AttendanceRepository:
    def __init__(self, db_path: str = "data/cfrs.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._init_schema()

    def _init_schema(self) -> None:
        schema_path = os.path.join(os.path.dirname(__file__), "database_schema.sql")
        with open(schema_path, "r", encoding="utf-8") as f:
            sql = f.read()
        with self._lock:
            self._conn.executescript(sql)
            self._conn.commit()

    @staticmethod
    def _normalize_name(full_name: Optional[str]) -> str:
        if not full_name:
            return ""
        return " ".join(full_name.strip().split())

    @staticmethod
    def _split_name(full_name: str) -> Tuple[str, str]:
        parts = full_name.split(" ", 1)
        first_name = parts[0]
        last_name = parts[1] if len(parts) > 1 else ""
        return first_name, last_name

    @staticmethod
    def _safe_iso(ts: Optional[str]) -> str:
        if not ts:
            return datetime.now().isoformat(timespec="seconds")
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).isoformat(timespec="seconds")
        except Exception:
            return datetime.now().isoformat(timespec="seconds")

    @staticmethod
    def _date_of(ts_iso: str) -> str:
        return datetime.fromisoformat(ts_iso).date().isoformat()

    @staticmethod
    def _is_real_name(name: str) -> bool:
        return name and name.upper() not in UNKNOWN_NAMES and name not in UNKNOWN_NAMES

    def upsert_student(self, full_name: str, student_code: Optional[str] = None) -> int:
        normalized = self._normalize_name(full_name)
        if not normalized:
            raise ValueError("full_name is required")

        now_iso = datetime.now().isoformat(timespec="seconds")
        first_name, last_name = self._split_name(normalized)

        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM students WHERE full_name = ?",
                (normalized,),
            ).fetchone()
            if row:
                self._conn.execute(
                    """
                    UPDATE students
                    SET student_code = COALESCE(?, student_code),
                        first_name = ?,
                        last_name = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (student_code, first_name, last_name, now_iso, row["id"]),
                )
                self._conn.commit()
                return int(row["id"])

            cursor = self._conn.execute(
                """
                INSERT INTO students (
                    student_code, first_name, last_name, full_name,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (student_code, first_name, last_name, normalized, now_iso, now_iso),
            )
            self._conn.commit()
            return int(cursor.lastrowid)

    def auto_save_attendance(
        self,
        full_name: str,
        timestamp: Optional[str] = None,
        confidence: Optional[float] = None,
        track_id: Optional[int] = None,
        source: str = "ai-confirmed",
    ) -> Dict[str, Any]:
        normalized = self._normalize_name(full_name)
        if not self._is_real_name(normalized):
            return {
                "saved": False,
                "reason": "ignored_non_student_name",
                "full_name": normalized,
            }

        ts_iso = self._safe_iso(timestamp)
        attendance_date = self._date_of(ts_iso)
        student_id = self.upsert_student(normalized)
        now_iso = datetime.now().isoformat(timespec="seconds")

        with self._lock:
            existing = self._conn.execute(
                """
                SELECT id, first_seen_at FROM attendance_logs
                WHERE student_id = ? AND attendance_date = ?
                """,
                (student_id, attendance_date),
            ).fetchone()

            if existing:
                return {
                    "saved": False,
                    "reason": "already_saved_today",
                    "student_id": student_id,
                    "full_name": normalized,
                    "first_seen_at": existing["first_seen_at"],
                }

            cursor = self._conn.execute(
                """
                INSERT INTO attendance_logs (
                    student_id, attendance_date, first_seen_at,
                    source, confidence, track_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (student_id, attendance_date, ts_iso, source, confidence, track_id, now_iso),
            )
            self._conn.commit()

        return {
            "saved": True,
            "attendance_id": int(cursor.lastrowid),
            "student_id": student_id,
            "full_name": normalized,
            "attendance_date": attendance_date,
            "first_seen_at": ts_iso,
        }

    def save_behavior_status(
        self,
        full_name: Optional[str],
        state: str,
        timestamp: Optional[str] = None,
        confidence: Optional[float] = None,
        track_id: Optional[int] = None,
        source: str = "ai",
    ) -> int:
        normalized = self._normalize_name(full_name)
        ts_iso = self._safe_iso(timestamp)
        now_iso = datetime.now().isoformat(timespec="seconds")
        is_drowsy = 1 if state in DROWSY_STATES else 0

        student_id = None
        if self._is_real_name(normalized):
            student_id = self.upsert_student(normalized)

        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT INTO behavior_status_logs (
                    student_id, track_id, state, is_drowsy,
                    confidence, event_time, source, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (student_id, track_id, state, is_drowsy, confidence, ts_iso, source, now_iso),
            )
            self._conn.commit()
            return int(cursor.lastrowid)

    def ingest_ai_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        ts = payload.get("timestamp")
        students = payload.get("students", [])

        attendance_saved = 0
        attendance_skipped = 0
        behavior_saved = 0

        for item in students:
            name = self._normalize_name(str(item.get("name", "")))
            state = str(item.get("state", "ไม่ทราบสถานะ"))
            confirmed = bool(item.get("confirmed", False))
            track_id = item.get("track_id")
            confidence = item.get("confidence")

            self.save_behavior_status(
                full_name=name if self._is_real_name(name) else None,
                state=state,
                timestamp=ts,
                confidence=confidence,
                track_id=track_id,
                source="ai-payload",
            )
            behavior_saved += 1

            if confirmed and self._is_real_name(name):
                result = self.auto_save_attendance(
                    full_name=name,
                    timestamp=ts,
                    confidence=confidence,
                    track_id=track_id,
                    source="ai-confirmed",
                )
                if result.get("saved"):
                    attendance_saved += 1
                else:
                    attendance_skipped += 1

        return {
            "ok": True,
            "person_count": payload.get("person_count", 0),
            "students_received": len(students),
            "attendance_saved": attendance_saved,
            "attendance_skipped": attendance_skipped,
            "behavior_saved": behavior_saved,
            "timestamp": self._safe_iso(ts),
        }

    def list_students(self, limit: int = 200) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, student_code, first_name, last_name, full_name, is_active,
                       created_at, updated_at
                FROM students
                ORDER BY full_name ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_student_by_code(self, student_code: str) -> Optional[Dict[str, Any]]:
        code = (student_code or "").strip()
        if not code:
            return None
        with self._lock:
            row = self._conn.execute(
                """
                SELECT id, student_code, first_name, last_name, full_name, is_active,
                       created_at, updated_at
                FROM students
                WHERE student_code = ?
                LIMIT 1
                """,
                (code,),
            ).fetchone()
        return dict(row) if row else None

    def get_today_report(self) -> Dict[str, Any]:
        today = datetime.now().date().isoformat()

        with self._lock:
            attendance_count = self._conn.execute(
                "SELECT COUNT(*) AS c FROM attendance_logs WHERE attendance_date = ?",
                (today,),
            ).fetchone()["c"]

            drowsy_events = self._conn.execute(
                """
                SELECT COUNT(*) AS c
                FROM behavior_status_logs
                WHERE date(event_time) = ? AND is_drowsy = 1
                """,
                (today,),
            ).fetchone()["c"]

            drowsy_students = self._conn.execute(
                """
                SELECT COUNT(DISTINCT student_id) AS c
                FROM behavior_status_logs
                WHERE date(event_time) = ? AND is_drowsy = 1 AND student_id IS NOT NULL
                """,
                (today,),
            ).fetchone()["c"]

            sleepy_names_rows = self._conn.execute(
                """
                SELECT s.full_name, MAX(b.event_time) AS last_seen
                FROM behavior_status_logs b
                JOIN students s ON s.id = b.student_id
                WHERE date(b.event_time) = ? AND b.is_drowsy = 1
                GROUP BY s.full_name
                ORDER BY last_seen DESC
                """,
                (today,),
            ).fetchall()

        return {
            "date": today,
            "attendance_total": int(attendance_count),
            "drowsy_event_total": int(drowsy_events),
            "drowsy_student_total": int(drowsy_students),
            "drowsy_students": [dict(r) for r in sleepy_names_rows],
        }

    def get_behavior_report(self, start_date: str, end_date: str) -> Dict[str, Any]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT state, COUNT(*) AS total
                FROM behavior_status_logs
                WHERE date(event_time) BETWEEN ? AND ?
                GROUP BY state
                ORDER BY total DESC
                """,
                (start_date, end_date),
            ).fetchall()

            drowsy_unique = self._conn.execute(
                """
                SELECT COUNT(DISTINCT student_id) AS c
                FROM behavior_status_logs
                WHERE date(event_time) BETWEEN ? AND ?
                  AND is_drowsy = 1
                  AND student_id IS NOT NULL
                """,
                (start_date, end_date),
            ).fetchone()["c"]

        return {
            "start_date": start_date,
            "end_date": end_date,
            "state_counts": [dict(r) for r in rows],
            "drowsy_unique_students": int(drowsy_unique),
        }

    def get_dashboard_summary(self, days: int = 7, recent_limit: int = 12) -> Dict[str, Any]:
        days = max(1, min(int(days), 30))
        recent_limit = max(1, min(int(recent_limit), 50))

        today_date = datetime.now().date()
        today = today_date.isoformat()
        start_date = (today_date - timedelta(days=days - 1)).isoformat()

        with self._lock:
            total_active_students = int(
                self._conn.execute(
                    "SELECT COUNT(*) AS c FROM students WHERE is_active = 1",
                ).fetchone()["c"]
            )

            attendance_today = int(
                self._conn.execute(
                    "SELECT COUNT(*) AS c FROM attendance_logs WHERE attendance_date = ?",
                    (today,),
                ).fetchone()["c"]
            )

            drowsy_students_today = int(
                self._conn.execute(
                    """
                    SELECT COUNT(DISTINCT student_id) AS c
                    FROM behavior_status_logs
                    WHERE date(event_time) = ? AND is_drowsy = 1 AND student_id IS NOT NULL
                    """,
                    (today,),
                ).fetchone()["c"]
            )

            behavior_events_today = int(
                self._conn.execute(
                    "SELECT COUNT(*) AS c FROM behavior_status_logs WHERE date(event_time) = ?",
                    (today,),
                ).fetchone()["c"]
            )

            unknown_today = int(
                self._conn.execute(
                    """
                    SELECT COUNT(*) AS c
                    FROM behavior_status_logs
                    WHERE date(event_time) = ? AND student_id IS NULL
                    """,
                    (today,),
                ).fetchone()["c"]
            )

            recent_attendance = [
                dict(r)
                for r in self._conn.execute(
                    """
                    SELECT s.full_name, a.first_seen_at, a.confidence
                    FROM attendance_logs a
                    JOIN students s ON s.id = a.student_id
                    ORDER BY a.first_seen_at DESC
                    LIMIT ?
                    """,
                    (recent_limit,),
                ).fetchall()
            ]

            attendance_today_list = [
                dict(r)
                for r in self._conn.execute(
                    """
                    SELECT s.full_name, a.first_seen_at, a.confidence
                    FROM attendance_logs a
                    JOIN students s ON s.id = a.student_id
                    WHERE a.attendance_date = ?
                    ORDER BY a.first_seen_at DESC
                    LIMIT ?
                    """,
                    (today, max(recent_limit, 30)),
                ).fetchall()
            ]

            state_distribution = [
                dict(r)
                for r in self._conn.execute(
                    """
                    SELECT state, COUNT(*) AS total
                    FROM behavior_status_logs
                    WHERE date(event_time) = ?
                    GROUP BY state
                    ORDER BY total DESC
                    LIMIT 8
                    """,
                    (today,),
                ).fetchall()
            ]

            drowsy_rank = [
                dict(r)
                for r in self._conn.execute(
                    """
                    SELECT s.full_name, COUNT(*) AS drowsy_count, MAX(b.event_time) AS last_seen
                    FROM behavior_status_logs b
                    JOIN students s ON s.id = b.student_id
                    WHERE date(b.event_time) BETWEEN ? AND ?
                      AND b.is_drowsy = 1
                    GROUP BY s.full_name
                    ORDER BY drowsy_count DESC, last_seen DESC
                    LIMIT 8
                    """,
                    (start_date, today),
                ).fetchall()
            ]

            attendance_rows = self._conn.execute(
                """
                SELECT attendance_date, COUNT(*) AS total
                FROM attendance_logs
                WHERE attendance_date BETWEEN ? AND ?
                GROUP BY attendance_date
                ORDER BY attendance_date ASC
                """,
                (start_date, today),
            ).fetchall()
            drowsy_rows = self._conn.execute(
                """
                SELECT date(event_time) AS day, COUNT(*) AS total
                FROM behavior_status_logs
                WHERE date(event_time) BETWEEN ? AND ? AND is_drowsy = 1
                GROUP BY day
                ORDER BY day ASC
                """,
                (start_date, today),
            ).fetchall()

        attendance_map = {r["attendance_date"]: int(r["total"]) for r in attendance_rows}
        drowsy_map = {r["day"]: int(r["total"]) for r in drowsy_rows}

        latest_checkin = attendance_today_list[0] if attendance_today_list else None
        attendance_rate_today = 0.0
        if total_active_students > 0:
            attendance_rate_today = round((attendance_today / total_active_students) * 100.0, 1)

        trend = []
        for i in range(days):
            day = (today_date - timedelta(days=days - 1 - i)).isoformat()
            trend.append(
                {
                    "day": day,
                    "attendance": attendance_map.get(day, 0),
                    "drowsy_events": drowsy_map.get(day, 0),
                }
            )

        return {
            "date": today,
            "window_days": days,
            "kpis": {
                "attendance_today": attendance_today,
                "total_active_students": total_active_students,
                "attendance_rate_today": attendance_rate_today,
                "drowsy_students_today": drowsy_students_today,
                "behavior_events_today": behavior_events_today,
                "unknown_events_today": unknown_today,
            },
            "latest_checkin": latest_checkin,
            "attendance_today_list": attendance_today_list,
            "state_distribution": state_distribution,
            "recent_attendance": recent_attendance,
            "drowsy_rank": drowsy_rank,
            "trend": trend,
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()
