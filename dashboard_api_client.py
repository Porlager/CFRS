import argparse
import json
from datetime import datetime

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:5000"


def pretty_print(data):
    print(json.dumps(data, ensure_ascii=False, indent=2))


def call_health(base_url: str):
    response = requests.get(f"{base_url}/health", timeout=5)
    pretty_print(response.json())


def call_today_report(base_url: str):
    response = requests.get(f"{base_url}/api/reports/today", timeout=5)
    pretty_print(response.json())


def call_behavior_report(base_url: str, start_date: str, end_date: str):
    response = requests.get(
        f"{base_url}/api/reports/behavior",
        params={"start_date": start_date, "end_date": end_date},
        timeout=5,
    )
    pretty_print(response.json())


def call_auto_save(base_url: str, full_name: str):
    payload = {
        "full_name": full_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "confidence": 95.0,
    }
    response = requests.post(f"{base_url}/api/attendance/auto-save", json=payload, timeout=5)
    pretty_print(response.json())


def call_send_sample_result(base_url: str):
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "person_count": 3,
        "students": [
            {"track_id": 1, "name": "SOMCHAI JAIDEE", "state": "ตั้งใจเรียน", "confirmed": True, "confidence": 97.4},
            {"track_id": 2, "name": "SUDAPORN DEEDEE", "state": "หลับ/เหม่อ", "confirmed": True, "confidence": 90.2},
            {"track_id": 3, "name": "คนแปลกหน้า", "state": "ฟุบหลับ/หันหลัง", "confirmed": False, "confidence": 0.0},
        ],
    }
    response = requests.post(f"{base_url}/api/result", json=payload, timeout=5)
    pretty_print(response.json())


def main():
    parser = argparse.ArgumentParser(description="Dashboard API client for CFRS")
    parser.add_argument("command", choices=["health", "today", "behavior", "auto-save", "sample"], help="API command")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Dashboard API base URL")
    parser.add_argument("--name", default="", help="Full name for auto-save")
    parser.add_argument("--start-date", default=datetime.now().date().isoformat(), help="Behavior report start date YYYY-MM-DD")
    parser.add_argument("--end-date", default=datetime.now().date().isoformat(), help="Behavior report end date YYYY-MM-DD")

    args = parser.parse_args()

    if args.command == "health":
        call_health(args.base_url)
    elif args.command == "today":
        call_today_report(args.base_url)
    elif args.command == "behavior":
        call_behavior_report(args.base_url, args.start_date, args.end_date)
    elif args.command == "auto-save":
        if not args.name.strip():
            raise SystemExit("--name is required for auto-save")
        call_auto_save(args.base_url, args.name.strip())
    elif args.command == "sample":
        call_send_sample_result(args.base_url)


if __name__ == "__main__":
    main()
