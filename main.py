import face_recognition
import cv2
import os
import sys
import math
import re
import sqlite3
import time
import numpy as np
import threading
import requests
from collections import deque
from datetime import datetime
from queue import Empty, Full, Queue


REGISTER_FACE_SUFFIX_PATTERN = re.compile(r"_\d{8}_\d{6}_\d{6}$")
UNKNOWN_FACE_NAMES = {"Unknown", "Moving/Blur", ""}


def _normalize_space(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def _name_lookup_key(text: str) -> str:
    return _normalize_space(text).upper()


def _parse_identity_from_filename(filename: str):
    stem = os.path.splitext(filename)[0].strip()
    stem = REGISTER_FACE_SUFFIX_PATTERN.sub("", stem)
    parts = [p for p in stem.split("_") if p]

    if not parts:
        fallback = "UNKNOWN"
        return fallback, None, fallback

    student_code = None
    if parts[0].isdigit() and 6 <= len(parts[0]) <= 16:
        student_code = parts[0]

    if student_code:
        display_tokens = parts[1:]
        display_name = _normalize_space(" ".join(display_tokens)) if display_tokens else student_code
        identity_key = student_code
    else:
        display_name = _normalize_space(" ".join(parts))
        identity_key = _name_lookup_key(display_name) or "UNKNOWN"

    return identity_key, student_code, (display_name or identity_key)


def classify_state_bucket(state: str) -> str:
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


def _smooth_track_state(track_data, observed_state: str, window_size: int) -> str:
    history = track_data.get("state_history")
    if history is None or history.maxlen != window_size:
        history = deque(maxlen=window_size)
        track_data["state_history"] = history

    if observed_state:
        history.append(observed_state)

    if not history:
        track_data["last_state"] = "ไม่ทราบสถานะ"
        return "ไม่ทราบสถานะ"

    vote_weights = {}
    for idx, state in enumerate(history):
        # Recent frames get slightly higher weight for faster adaptation.
        weight = 1.0 + (idx / max(1, len(history) - 1)) * 0.3
        vote_weights[state] = vote_weights.get(state, 0.0) + weight

    stable_state = max(vote_weights.items(), key=lambda item: item[1])[0]
    track_data["last_state"] = stable_state
    return stable_state


def _classify_body_posture_state(body_bb, lying_ratio: float, slouch_ratio: float, previous_state: str) -> str:
    width = max(1, int(body_bb["Right"]) - int(body_bb["Left"]))
    height = max(1, int(body_bb["Bottom"]) - int(body_bb["Top"]))
    ratio = float(height) / float(width)

    if ratio <= lying_ratio:
        return "ฟุบหลับ/นอน"
    if ratio <= slouch_ratio:
        return "หันหลัง/ไม่ตั้งใจ"

    if classify_state_bucket(previous_state) in {"drowsy", "inattentive"}:
        return previous_state
    return "ไม่ทราบสถานะ"


class StudentDirectory:
    def __init__(self, db_path: str = "data/cfrs.db", refresh_sec: float = 20.0):
        self.db_path = db_path
        self.refresh_sec = max(3.0, float(refresh_sec))
        self._last_refresh = 0.0
        self._lock = threading.Lock()
        self._by_code = {}
        self._by_name = {}
        self.refresh_if_needed(force=True)

    def refresh_if_needed(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_refresh) < self.refresh_sec:
            return

        if not os.path.exists(self.db_path):
            self._last_refresh = now
            return

        rows = []
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT student_code, full_name
                FROM students
                WHERE is_active = 1
                """
            ).fetchall()
        except Exception:
            self._last_refresh = now
            return
        finally:
            if conn is not None:
                conn.close()

        by_code = {}
        by_name = {}
        for row in rows:
            code = str(row["student_code"] or "").strip()
            full_name = _normalize_space(str(row["full_name"] or ""))
            if full_name:
                by_name[_name_lookup_key(full_name)] = {
                    "full_name": full_name,
                    "student_code": code or None,
                }
            if code:
                by_code[code] = {
                    "full_name": full_name or code,
                    "student_code": code,
                }

        with self._lock:
            self._by_code = by_code
            self._by_name = by_name
            self._last_refresh = now

    def resolve_identity(self, display_name: str, student_code=None):
        self.refresh_if_needed()
        resolved_name = _normalize_space(display_name)
        resolved_code = str(student_code or "").strip() or None

        with self._lock:
            if resolved_code and resolved_code in self._by_code:
                by_code = self._by_code[resolved_code]
                return by_code["full_name"], resolved_code

            name_key = _name_lookup_key(resolved_name)
            if name_key and name_key in self._by_name:
                by_name = self._by_name[name_key]
                return by_name["full_name"], (by_name.get("student_code") or resolved_code)

        if resolved_code and not resolved_name:
            return resolved_code, resolved_code
        return (resolved_name or "Unknown"), resolved_code


class AsyncPayloadPoster:
    def __init__(self, backend_url: str, timeout: float = 1.2, max_queue_size: int = 4):
        self.backend_url = backend_url
        self.timeout = max(0.2, float(timeout))
        self.queue = Queue(maxsize=max(1, int(max_queue_size)))
        self.stop_event = threading.Event()
        self._last_error_time = 0.0
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def submit(self, payload: dict) -> None:
        try:
            self.queue.put_nowait(payload)
            return
        except Full:
            pass

        # Drop the oldest payload so fresh state can be sent quickly.
        try:
            self.queue.get_nowait()
            self.queue.task_done()
        except Empty:
            pass

        try:
            self.queue.put_nowait(payload)
        except Full:
            pass

    def _run(self) -> None:
        session = requests.Session()
        while not self.stop_event.is_set():
            try:
                payload = self.queue.get(timeout=0.2)
            except Empty:
                continue

            try:
                session.post(self.backend_url, json=payload, timeout=self.timeout)
            except Exception as exc:
                now = time.time()
                if now - self._last_error_time >= 8:
                    print(f"[{datetime.now()}] WARN: Cannot send payload to backend: {exc}")
                    self._last_error_time = now
            finally:
                self.queue.task_done()

        session.close()

    def close(self) -> None:
        self.stop_event.set()
        self._worker.join(timeout=2.0)


def _write_dashboard_frame(frame_output_path, jpeg_bytes):
    tmp_path = f"{frame_output_path}.tmp"

    for attempt in range(4):
        try:
            with open(tmp_path, "wb") as fp:
                fp.write(jpeg_bytes)
            os.replace(tmp_path, frame_output_path)
            return
        except PermissionError:
            time.sleep(0.02 * (attempt + 1))
        except OSError:
            time.sleep(0.02 * (attempt + 1))
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    # Fallback when replace is blocked by a transient read lock.
    with open(frame_output_path, "wb") as fp:
        fp.write(jpeg_bytes)

class ClassroomFacialRecognitionService:

    def __init__(self, known_faces_path='known_faces'):
        self.path = known_faces_path
        self.known_db = {}
        self.identity_profiles = {}
        self.WEIGHT_BEST = 0.7
        self.WEIGHT_SECOND = 0.3
        self.CONFIDENCE_K = 12
        self.MIN_CONFIDENCE = 60.0
        
        self.BLUR_MIN_THRESH = 40.0
        self.BLUR_MAX_THRESH = 60.0
        self.BLUR_FACE_RATIO = 800.0
        self.EAR_THRESH = float(os.getenv("CFRS_EAR_THRESH", "0.20"))
        self.POSE_INATTENTIVE_PENALTY = float(os.getenv("CFRS_POSE_INATTENTIVE_PENALTY", "0.06"))
        self.body_detector = cv2.HOGDescriptor()
        self.body_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self._load_and_encode_database()

    def _load_and_encode_database(self):
        """Indexing ด้วย num_jitters=5 พร้อม Error Handling"""
        print(f"[{datetime.now()}] INFO: Booting Identity Service...")
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            print(f"[{datetime.now()}] WARN: Directory created. Please add images to '{self.path}'.")
            return

        for filename in os.listdir(self.path):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            identity_key, student_code, display_name = _parse_identity_from_filename(filename)
            img_path = os.path.join(self.path, filename)

            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"[{datetime.now()}] WARN: Cannot read {filename}. Skipping...")
                    continue

                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(rgb_img, num_jitters=5)

                if len(encodings) > 0:
                    if identity_key not in self.known_db:
                        self.known_db[identity_key] = []
                    self.known_db[identity_key].append(encodings[0])

                    self.identity_profiles[identity_key] = {
                        "student_code": student_code,
                        "display_name": display_name,
                    }
                    label = f"{display_name} ({student_code})" if student_code else display_name
                    print(f"  -> SUCCESS: Learned face from {filename} -> {label}")
                else:
                    print(f"  -> WARN: No face detected in {filename}")
            except Exception as e:
                print(f"[{datetime.now()}] ERROR: Failed to process {filename}: {e}")
        
        print(f"[{datetime.now()}] INFO: Ready! Loaded {len(self.known_db)} identities.")

    def get_identity_profile(self, identity_key: str):
        key = str(identity_key or "").strip()
        profile = self.identity_profiles.get(key)
        if profile:
            return dict(profile)
        normalized_key = _normalize_space(key)
        return {
            "student_code": normalized_key if normalized_key.isdigit() else None,
            "display_name": normalized_key or "Unknown",
        }

    def _check_blur_dynamic(self, rgb_image, face_loc):
        top, right, bottom, left = face_loc
        face_width = right - left
        face_crop = rgb_image[top:bottom, left:right]
        if face_crop.size == 0: return False
        
        gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()

        dynamic_blur_thresh = min(self.BLUR_MAX_THRESH, max(self.BLUR_MIN_THRESH, self.BLUR_FACE_RATIO / max(face_width, 1.0)))

        return variance > dynamic_blur_thresh

    def _get_pose_penalty(self, face_landmarks):
        """รับ Landmarks ที่คำนวณแบบ Batch มาแล้ว ไม่ต้องคำนวณใหม่ทีละหน้า"""
        if not face_landmarks: return 0.0
        
        lm = face_landmarks
        if 'left_eye' in lm and 'right_eye' in lm and 'nose_bridge' in lm:
            left_eye_center = np.mean(lm['left_eye'], axis=0)
            right_eye_center = np.mean(lm['right_eye'], axis=0)
            nose_center = np.mean(lm['nose_bridge'], axis=0)
            
            dist_left = np.linalg.norm(left_eye_center - nose_center)
            dist_right = np.linalg.norm(right_eye_center - nose_center)

            ratio = min(dist_left, dist_right) / max(dist_left, dist_right + 1e-6)
            if ratio < 0.4: return 0.08
            if ratio < 0.6: return 0.04
        return 0.0

    def get_dynamic_threshold_and_margin(self, face_width, pose_penalty):
        face_width = max(20, min(100, face_width))
        t = (face_width - 20) / (100 - 20)
        base_thresh = 0.60 - (0.10 * t)
        margin = 0.08 - (0.06 * t)
        pose_penalty = max(0.0, min(0.1, pose_penalty))
        final_thresh = base_thresh - pose_penalty
        final_thresh = max(0.45, min(0.65, final_thresh))

        return round(final_thresh, 3), round(margin, 3)

    def _get_weighted_distance(self, known_encodings, target_encoding):
        dists = face_recognition.face_distance(known_encodings, target_encoding)
        if len(dists) == 1:
            return dists[0]
        
        sorted_dists = np.sort(dists)
        weighted_dist = (sorted_dists[0] * self.WEIGHT_BEST) + (sorted_dists[1] * self.WEIGHT_SECOND)
        return weighted_dist

    def _calculate_confidence(self, distance, threshold):
        confidence = 1 / (1 + math.exp(self.CONFIDENCE_K * (distance - threshold)))
        return round(max(1.0, min(99.9, confidence * 100)), 2)

    def _eye_aspect_ratio(self, eye_points):
        if not eye_points or len(eye_points) < 6:
            return 0.0
        p = np.array(eye_points, dtype=np.float32)
        a = np.linalg.norm(p[1] - p[5])
        b = np.linalg.norm(p[2] - p[4])
        c = np.linalg.norm(p[0] - p[3])
        if c <= 1e-6:
            return 0.0
        return float((a + b) / (2.0 * c))

    def _classify_attention_state(self, face_landmarks, pose_penalty=0.0):
        if not face_landmarks:
            return "ไม่ทราบสถานะ"

        if pose_penalty >= self.POSE_INATTENTIVE_PENALTY:
            return "ไม่ตั้งใจเรียน"

        left_eye = face_landmarks.get("left_eye")
        right_eye = face_landmarks.get("right_eye")
        if not left_eye or not right_eye:
            return "ไม่ทราบสถานะ"

        left_ear = self._eye_aspect_ratio(left_eye)
        right_ear = self._eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        return "หลับ/เหม่อ" if avg_ear < self.EAR_THRESH else "ตั้งใจเรียน"

    def process_frame(self, frame, resize_scale=1):
        small_frame = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations, num_jitters=1)
        all_face_landmarks = face_recognition.face_landmarks(rgb_small, face_locations)

        detections = []

        for idx, (encoding, face_loc) in enumerate(zip(face_encodings, face_locations)):
            top, right, bottom, left = face_loc
            face_width = right - left
            scale_back = 1.0 / resize_scale
            y1, x2, y2, x1 = [int(v * scale_back) for v in face_loc]
            bbox = {"Top": y1, "Right": x2, "Bottom": y2, "Left": x1}
            current_landmarks = all_face_landmarks[idx] if idx < len(all_face_landmarks) else None
            pose_penalty = self._get_pose_penalty(current_landmarks)
            behavior_state = self._classify_attention_state(current_landmarks, pose_penalty=pose_penalty)

            if not self._check_blur_dynamic(rgb_small, face_loc):
                detections.append(
                    {
                        "Name": "Moving/Blur",
                        "Confidence": 0.0,
                        "BoundingBox": bbox,
                        "State": "ไม่ทราบสถานะ",
                    }
                )
                continue

            threshold, required_margin = self.get_dynamic_threshold_and_margin(face_width, pose_penalty)

            person_distances = {}
            for name, known_encodings in self.known_db.items():
                person_distances[name] = self._get_weighted_distance(known_encodings, encoding)

            sorted_persons = sorted(person_distances.items(), key=lambda x: x[1])

            name = "Unknown"
            conf = 0.0

            if len(sorted_persons) > 0:
                best_match_name, best_dist = sorted_persons[0]

                actual_margin = 1.0
                if len(sorted_persons) > 1:
                    actual_margin = sorted_persons[1][1] - best_dist

                conf = self._calculate_confidence(best_dist, threshold)

                if best_dist <= threshold and actual_margin > required_margin and conf >= self.MIN_CONFIDENCE:
                    name = best_match_name
                else:
                    name = "Unknown"

            detections.append(
                {
                    "Name": name,
                    "Confidence": conf,
                    "BoundingBox": bbox,
                    "State": behavior_state,
                }
            )

        return detections

    def detect_bodies(self, frame, resize_scale=0.45):
        scale = max(0.25, min(1.0, resize_scale))
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        rects, _ = self.body_detector.detectMultiScale(
            small_frame,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )

        detections = []
        scale_back = 1.0 / scale
        for x, y, w, h in rects:
            x1 = int(x * scale_back)
            y1 = int(y * scale_back)
            x2 = int((x + w) * scale_back)
            y2 = int((y + h) * scale_back)
            if (x2 - x1) < 60 or (y2 - y1) < 120:
                continue
            detections.append({"Top": y1, "Right": x2, "Bottom": y2, "Left": x1})
        return detections


def state_for_overlay(state: str) -> str:
    bucket = classify_state_bucket(state)
    if bucket == "attentive":
        return "attentive"
    if bucket == "inattentive":
        return "inattentive"
    if bucket == "drowsy":
        return "drowsy"
    return "unknown"


def bbox_center(bb):
    return ((bb["Left"] + bb["Right"]) // 2, (bb["Top"] + bb["Bottom"]) // 2)


def bbox_iou(a, b):
    x_left = max(a["Left"], b["Left"])
    y_top = max(a["Top"], b["Top"])
    x_right = min(a["Right"], b["Right"])
    y_bottom = min(a["Bottom"], b["Bottom"])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    inter_area = float((x_right - x_left) * (y_bottom - y_top))
    area_a = float((a["Right"] - a["Left"]) * (a["Bottom"] - a["Top"]))
    area_b = float((b["Right"] - b["Left"]) * (b["Bottom"] - b["Top"]))
    denom = area_a + area_b - inter_area
    if denom <= 1e-6:
        return 0.0
    return inter_area / denom


def _open_camera_by_index(camera_index: int):
    backend_candidates = [None]
    if os.name == "nt":
        backend_candidates = [cv2.CAP_DSHOW, cv2.CAP_MSMF, None]
    elif sys.platform == "darwin":
        backend_candidates = [cv2.CAP_AVFOUNDATION, None]

    for backend in backend_candidates:
        cap = cv2.VideoCapture(camera_index, backend) if backend is not None else cv2.VideoCapture(camera_index)
        if cap is not None and cap.isOpened():
            return cap
        if cap is not None:
            cap.release()
    return None


def _scan_available_cameras(max_index: int = 6):
    available = []
    for idx in range(max(1, max_index)):
        cap = _open_camera_by_index(idx)
        if cap is not None and cap.isOpened():
            available.append(idx)
            cap.release()
    return available


def _prompt_camera_choice(available_indices):
    if not available_indices:
        raise RuntimeError("No camera device detected.")

    print(f"[{datetime.now()}] INFO: Available cameras:")
    for idx in available_indices:
        print(f"  [{idx}] Camera index {idx}")

    default_index = available_indices[0]

    while True:
        choice = input(f"Select camera index [{default_index}]: ").strip()
        if not choice:
            return default_index
        if choice.lstrip("-").isdigit() and int(choice) in available_indices:
            return int(choice)
        print(f"[{datetime.now()}] WARN: Invalid choice '{choice}'. Please choose one of {available_indices}.")


def _open_camera_from_env():
    camera_source_raw = str(os.getenv("CFRS_CAMERA_SOURCE", "")).strip()
    prompt_choice = str(os.getenv("CFRS_CAMERA_CHOICE", "1")).strip().lower() in ("1", "true", "yes", "on")
    max_probe = int(os.getenv("CFRS_CAMERA_SCAN_MAX", "6"))

    camera_source_used = camera_source_raw
    cap = None

    if not camera_source_raw or camera_source_raw.lower() == "ask":
        available = _scan_available_cameras(max_probe)
        if not available:
            raise RuntimeError(
                f"No camera detected on this machine (scanned indexes 0..{max(1, max_probe) - 1}). "
                "Connect a camera and try again."
            )

        if prompt_choice and sys.stdin.isatty():
            camera_index = _prompt_camera_choice(available)
        else:
            camera_index = available[0]
            print(
                f"[{datetime.now()}] INFO: Auto-selected camera index {camera_index}. "
                "Set CFRS_CAMERA_SOURCE=ask for interactive choice."
            )

        cap = _open_camera_by_index(camera_index)
        camera_source_used = str(camera_index)
    elif camera_source_raw.lstrip("-").isdigit():
        camera_index = int(camera_source_raw)
        cap = _open_camera_by_index(camera_index)
        camera_source_used = str(camera_index)
    else:
        cap = cv2.VideoCapture(camera_source_raw)

        # IP Webcam often serves frames at /video even if user sets only base URL.
        if (
            (cap is None or not cap.isOpened())
            and camera_source_raw.startswith(("http://", "https://"))
            and not camera_source_raw.rstrip("/").lower().endswith("/video")
        ):
            fallback_url = f"{camera_source_raw.rstrip('/')}/video"
            cap = cv2.VideoCapture(fallback_url)
            camera_source_used = fallback_url

    if (cap is None or not cap.isOpened()) and prompt_choice and sys.stdin.isatty():
        available = _scan_available_cameras(max_probe)
        if available:
            print(
                f"[{datetime.now()}] WARN: Cannot open camera source '{camera_source_raw or 'default'}'. "
                "Please choose from available cameras."
            )
            camera_index = _prompt_camera_choice(available)
            cap = _open_camera_by_index(camera_index)
            camera_source_used = str(camera_index)

    if cap is None or not cap.isOpened():
        available = _scan_available_cameras(max_probe)
        available_msg = ", ".join(str(i) for i in available) if available else "none"
        raise RuntimeError(
            f"Cannot open camera source '{camera_source_raw}'. "
            f"Detected camera indexes: {available_msg}. "
            "Set CFRS_CAMERA_SOURCE=ask to choose camera, or set CFRS_CAMERA_SOURCE=<index>, "
            "or use IP Webcam URL (example: http://PHONE_IP:8080/video)."
        )

    return cap, camera_source_used


if __name__ == "__main__":
    service = ClassroomFacialRecognitionService()
    try:
        cap, camera_source_used = _open_camera_from_env()
        print(f"[{datetime.now()}] INFO: Camera source = {camera_source_used}")
    except Exception as exc:
        print(f"[{datetime.now()}] ERROR: {exc}")
        raise SystemExit(1)

    camera_width = int(os.getenv("CFRS_CAMERA_WIDTH", "960"))
    camera_height = int(os.getenv("CFRS_CAMERA_HEIGHT", "540"))
    frame_resize_scale = float(os.getenv("CFRS_FRAME_RESIZE_SCALE", "0.5"))
    frame_resize_scale = max(0.35, min(1.0, frame_resize_scale))
    process_every_n_frames = int(os.getenv("CFRS_PROCESS_EVERY_N_FRAMES", "3"))
    process_every_n_frames = max(1, min(6, process_every_n_frames))
    body_detect_every_n_frames = int(os.getenv("CFRS_BODY_DETECT_EVERY_N_FRAMES", "4"))
    body_detect_every_n_frames = max(1, min(8, body_detect_every_n_frames))
    body_resize_scale = float(os.getenv("CFRS_BODY_RESIZE_SCALE", "0.45"))
    body_resize_scale = max(0.25, min(0.8, body_resize_scale))
    body_match_max_distance = int(os.getenv("CFRS_BODY_MATCH_MAX_DISTANCE", "190"))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

    backend_url = os.getenv("CFRS_BACKEND_INGEST_URL", "http://127.0.0.1:5000/api/result")
    frame_output_path = os.getenv("CFRS_CAMERA_FRAME_PATH", "storage/latest_frame.jpg")
    post_interval_sec = float(os.getenv("CFRS_POST_INTERVAL_SEC", "1.5"))
    frame_write_interval_sec = float(os.getenv("CFRS_FRAME_WRITE_INTERVAL_SEC", "0.45"))
    payload_post_timeout_sec = float(os.getenv("CFRS_POST_TIMEOUT_SEC", "1.2"))
    state_smoothing_window = int(os.getenv("CFRS_STATE_SMOOTHING_WINDOW", "6"))
    state_smoothing_window = max(3, min(12, state_smoothing_window))
    body_lying_ratio = float(os.getenv("CFRS_BODY_LYING_RATIO", "1.08"))
    body_slouch_ratio = float(os.getenv("CFRS_BODY_SLOUCH_RATIO", "1.45"))
    student_dir_refresh_sec = float(os.getenv("CFRS_STUDENT_DIR_REFRESH_SEC", "20"))
    last_post_time = 0.0
    last_frame_write_time = 0.0
    last_frame_error_time = 0.0
    last_fps_time = time.time()
    smoothed_fps = 0.0
    frame_index = 0
    cached_results = []
    cached_body_boxes = []

    frame_output_dir = os.path.dirname(frame_output_path) or "."
    os.makedirs(frame_output_dir, exist_ok=True)

    tracked_faces = {}
    next_track_id = 0
    
    confirmed_student_codes = set()
    confirmed_names_db = set()
    student_directory = StudentDirectory(db_path=os.getenv("CFRS_DB_PATH", "data/cfrs.db"), refresh_sec=student_dir_refresh_sec)
    payload_poster = AsyncPayloadPoster(
        backend_url=backend_url,
        timeout=payload_post_timeout_sec,
        max_queue_size=int(os.getenv("CFRS_POST_QUEUE_SIZE", "4")),
    )

    tracker_lock = threading.Lock()

    CONFIRMATION_TIME = 5.0
    TIMEOUT = 10.0
    MAX_DISTANCE = 50

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            frame_index += 1
            should_run_heavy = (frame_index % process_every_n_frames == 1) or (not cached_results)
            if should_run_heavy:
                cached_results = service.process_frame(frame, resize_scale=frame_resize_scale)
            results = cached_results

            should_run_body = (frame_index % body_detect_every_n_frames == 1) or (not cached_body_boxes)
            if should_run_body:
                cached_body_boxes = service.detect_bodies(frame, resize_scale=body_resize_scale)
            body_boxes = cached_body_boxes

            payload_students = []
            drowsy_count = 0
            attentive_count = 0
            inattentive_count = 0
            unknown_count = 0
            seen_track_ids = set()
            face_boxes = []
            with tracker_lock:
                keys_to_delete = [k for k, v in tracked_faces.items() if current_time - v["last_seen"] > TIMEOUT]
                for k in keys_to_delete:
                    del tracked_faces[k]

                for res in results:
                    bb = res["BoundingBox"]
                    ai_name = res["Name"]
                    conf = res["Confidence"]
                    behavior_state_raw = res.get("State", "ไม่ทราบสถานะ")
                    behavior_display = state_for_overlay(behavior_state_raw)
                    face_boxes.append(bb)
                    
                    cx = (bb["Left"] + bb["Right"]) // 2
                    cy = (bb["Top"] + bb["Bottom"]) // 2

                    best_match_id = None
                    min_dist = MAX_DISTANCE

                    for t_id, t_data in tracked_faces.items():
                        dist = math.hypot(cx - t_data["centroid"][0], cy - t_data["centroid"][1])
                        if dist < min_dist:
                            min_dist = dist
                            best_match_id = t_id

                    is_tracking = False
                    is_confirmed = False
                    display_name = ai_name
                    display_student_code = None
                    elapsed_time = 0.0
                    payload_track_id = None

                    if best_match_id is None:
                        if ai_name not in UNKNOWN_FACE_NAMES:
                            profile = service.get_identity_profile(ai_name)
                            profile_name, profile_code = student_directory.resolve_identity(
                                profile.get("display_name", ai_name),
                                student_code=profile.get("student_code"),
                            )
                            payload_track_id = next_track_id
                            tracked_faces[next_track_id] = {
                                "identity_key": ai_name,
                                "name": profile_name,
                                "student_code": profile_code,
                                "centroid": (cx, cy),
                                "first_seen": current_time,
                                "last_seen": current_time,
                                "confirmed": False,
                                "state_history": deque(maxlen=state_smoothing_window),
                                "last_state": "ไม่ทราบสถานะ",
                            }
                            display_name = profile_name
                            display_student_code = profile_code
                            is_tracking = True
                            next_track_id += 1
                    else:
                        payload_track_id = best_match_id
                        t_data = tracked_faces[best_match_id]
                        t_data["centroid"] = (cx, cy)
                        t_data["last_seen"] = current_time
                        if ai_name not in UNKNOWN_FACE_NAMES:
                            profile = service.get_identity_profile(ai_name)
                            profile_name, profile_code = student_directory.resolve_identity(
                                profile.get("display_name", ai_name),
                                student_code=profile.get("student_code") or t_data.get("student_code"),
                            )
                            t_data["name"] = profile_name
                            if profile_code:
                                t_data["student_code"] = profile_code
                        
                        elapsed_time = current_time - t_data["first_seen"]
                        if elapsed_time >= CONFIRMATION_TIME:
                            t_data["confirmed"] = True
                            display_name = t_data["name"]
                            display_student_code = t_data.get("student_code")
                            is_confirmed = True
                            is_tracking = True
                        else:
                            display_name = t_data["name"]
                            display_student_code = t_data.get("student_code")
                            is_tracking = True

                    stable_state = behavior_state_raw
                    if is_tracking and payload_track_id is not None:
                        stable_state = _smooth_track_state(
                            tracked_faces[payload_track_id],
                            behavior_state_raw,
                            state_smoothing_window,
                        )
                        behavior_display = state_for_overlay(stable_state)

                    if not is_tracking:
                        if display_name == "Moving/Blur":
                            color = (0, 165, 255)
                            text = "Moving/Blur"
                        else:
                            color = (0, 0, 255)
                            text = f"Unknown ({behavior_display})"
                    else:
                        student_label = f"[{display_student_code}] " if display_student_code else ""
                        if is_confirmed:
                            color = (0, 255, 0)
                            text = f"{student_label}{display_name} ({behavior_display}) {conf:.1f}%"
                            if display_student_code:
                                if display_student_code not in confirmed_student_codes:
                                    print(f">>> [API/Database] เช็คชื่อ: {display_name} ({display_student_code}) เวลา: {datetime.now()}")
                                    confirmed_student_codes.add(display_student_code)
                            elif display_name not in confirmed_names_db:
                                print(f">>> [API/Database] เช็คชื่อ: {display_name} เวลา: {datetime.now()}")
                                confirmed_names_db.add(display_name)
                        else:
                            color = (0, 255, 255)
                            countdown = max(0, CONFIRMATION_TIME - elapsed_time)
                            text = f"กำลังยืนยัน {student_label}{display_name} ({behavior_display}) {countdown:.1f}s"

                    payload_state = stable_state if display_name != "Moving/Blur" else "ไม่ทราบสถานะ"
                    payload_name = display_name if display_name else "Unknown"
                    payload_students.append(
                        {
                            "track_id": payload_track_id,
                            "name": payload_name,
                            "student_code": display_student_code,
                            "identity_key": tracked_faces[payload_track_id].get("identity_key") if payload_track_id in tracked_faces else None,
                            "state": payload_state,
                            "confirmed": bool(is_confirmed),
                            "confidence": float(conf),
                        }
                    )
                    if payload_track_id is not None:
                        seen_track_ids.add(payload_track_id)

                    state_bucket = classify_state_bucket(payload_state)
                    if state_bucket == "drowsy":
                        drowsy_count += 1
                    elif state_bucket == "attentive":
                        attentive_count += 1
                    elif state_bucket == "inattentive":
                        inattentive_count += 1
                    else:
                        unknown_count += 1

                    cv2.rectangle(frame, (bb["Left"], bb["Top"]), (bb["Right"], bb["Bottom"]), color, 2)
                    cv2.putText(frame, text, (bb["Left"], bb["Top"] - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Body fallback: continue tracking and behavior when face temporarily disappears.
                for body_bb in body_boxes:
                    overlap_with_face = any(bbox_iou(body_bb, fb) > 0.22 for fb in face_boxes)
                    if overlap_with_face:
                        continue

                    body_cx, body_cy = bbox_center(body_bb)
                    best_track_id = None
                    best_dist = float("inf")

                    for t_id, t_data in tracked_faces.items():
                        if t_id in seen_track_ids:
                            continue
                        dist = math.hypot(body_cx - t_data["centroid"][0], body_cy - t_data["centroid"][1])
                        if dist < best_dist and dist <= body_match_max_distance:
                            best_dist = dist
                            best_track_id = t_id

                    if best_track_id is None:
                        continue

                    t_data = tracked_faces[best_track_id]
                    t_data["centroid"] = (body_cx, body_cy)
                    t_data["last_seen"] = current_time
                    seen_track_ids.add(best_track_id)

                    inherited_name = t_data.get("name", "Unknown") or "Unknown"
                    inherited_code = t_data.get("student_code")
                    if inherited_name in ["Moving/Blur", ""]:
                        inherited_name = "Unknown"

                    body_state = _classify_body_posture_state(
                        body_bb,
                        lying_ratio=body_lying_ratio,
                        slouch_ratio=body_slouch_ratio,
                        previous_state=t_data.get("last_state", "ไม่ทราบสถานะ"),
                    )
                    stable_body_state = _smooth_track_state(t_data, body_state, state_smoothing_window)

                    payload_students.append(
                        {
                            "track_id": best_track_id,
                            "name": inherited_name,
                            "student_code": inherited_code,
                            "identity_key": t_data.get("identity_key"),
                            "state": stable_body_state,
                            "confirmed": False,
                            "confidence": 0.0,
                        }
                    )
                    body_bucket = classify_state_bucket(stable_body_state)
                    if body_bucket == "drowsy":
                        drowsy_count += 1
                    elif body_bucket == "inattentive":
                        inattentive_count += 1
                    elif body_bucket == "attentive":
                        attentive_count += 1
                    else:
                        unknown_count += 1

                    body_text_name = inherited_name if inherited_name != "Unknown" else "Unknown"
                    body_code_label = f"[{inherited_code}] " if inherited_code else ""
                    body_state_overlay = state_for_overlay(stable_body_state)
                    body_color = (255, 160, 0) if body_bucket in {"drowsy", "inattentive"} else (180, 180, 180)
                    body_text = f"{body_code_label}{body_text_name} ({body_state_overlay})"
                    cv2.rectangle(frame, (body_bb["Left"], body_bb["Top"]), (body_bb["Right"], body_bb["Bottom"]), body_color, 2)
                    cv2.putText(frame, body_text, (body_bb["Left"], max(18, body_bb["Top"] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, body_color, 2)

            now = time.time()
            dt = max(1e-6, now - last_fps_time)
            last_fps_time = now
            current_fps = 1.0 / dt
            smoothed_fps = current_fps if smoothed_fps == 0.0 else (smoothed_fps * 0.9 + current_fps * 0.1)

            cv2.putText(frame, f"People: {len(payload_students)}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (60, 220, 60), 2)
            cv2.putText(frame, f"Attentive: {attentive_count}", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 210, 255), 2)
            cv2.putText(frame, f"Inattentive: {inattentive_count}", (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 180, 255), 2)
            cv2.putText(frame, f"Drowsy: {drowsy_count}", (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 60, 255), 2)
            cv2.putText(frame, f"Unknown: {unknown_count}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2)
            cv2.putText(frame, f"FPS: {smoothed_fps:.1f}", (10, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)

            if current_time - last_post_time >= post_interval_sec:
                last_post_time = current_time
                payload = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "person_count": len(payload_students),
                    "students": payload_students,
                }
                payload_poster.submit(payload)

            if current_time - last_frame_write_time >= frame_write_interval_sec:
                last_frame_write_time = current_time
                try:
                    ok, jpeg = cv2.imencode(
                        ".jpg",
                        frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 78],
                    )
                    if ok:
                        _write_dashboard_frame(frame_output_path, jpeg.tobytes())
                except Exception as exc:
                    if current_time - last_frame_error_time >= 8:
                        print(f"[{datetime.now()}] WARN: Cannot write camera frame for dashboard: {exc}")
                        last_frame_error_time = current_time

            cv2.imshow('Ultimate Classroom Identity (Tracker Lock)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        payload_poster.close()
        cap.release()
        cv2.destroyAllWindows()
