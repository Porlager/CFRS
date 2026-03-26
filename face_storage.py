import json
import os
import shutil
from typing import Dict, List, Optional

import cv2
import mediapipe as mp
import numpy as np


class FaceStorageManager:
    def __init__(self, storage_root: str = "storage/faces", target_size=(200, 200)):
        self.storage_root = storage_root
        self.target_size = target_size
        self.original_root = os.path.join(storage_root, "original")
        self.cache_root = os.path.join(storage_root, "cache")
        self.index_path = os.path.join(storage_root, "index.json")

        os.makedirs(self.original_root, exist_ok=True)
        os.makedirs(self.cache_root, exist_ok=True)

        self._index = self._load_index()
        self._dirty = False

    def _load_index(self) -> Dict[str, Dict[str, str]]:
        if not os.path.exists(self.index_path):
            return {}
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def flush(self) -> None:
        if not self._dirty:
            return
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self._index, f, ensure_ascii=False, indent=2)
        self._dirty = False

    def _cache_file_path(self, person_name: str, filename: str) -> str:
        person_dir = os.path.join(self.cache_root, person_name)
        os.makedirs(person_dir, exist_ok=True)
        stem, _ = os.path.splitext(filename)
        return os.path.join(person_dir, f"{stem}.npy")

    def _original_file_path(self, person_name: str, filename: str) -> str:
        person_dir = os.path.join(self.original_root, person_name)
        os.makedirs(person_dir, exist_ok=True)
        return os.path.join(person_dir, filename)

    def _extract_face_gray(self, img_bgr, face_landmarker) -> Optional[np.ndarray]:
        rgb_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        results = face_landmarker.detect(mp_image)

        if not getattr(results, "face_landmarks", None):
            return None

        face_landmarks = results.face_landmarks[0]
        img_h, img_w, _ = img_bgr.shape
        x_coords = [lm.x * img_w for lm in face_landmarks]
        y_coords = [lm.y * img_h for lm in face_landmarks]

        left = max(0, int(min(x_coords)))
        top = max(0, int(min(y_coords)))
        right = min(img_w - 1, int(max(x_coords)))
        bottom = min(img_h - 1, int(max(y_coords)))

        if top >= bottom or left >= right:
            return None

        crop = cv2.cvtColor(img_bgr[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
        if crop.size == 0:
            return None

        return cv2.resize(crop, self.target_size)

    def get_or_create_face_crop(self, img_path: str, person_name: str, face_landmarker) -> Optional[np.ndarray]:
        abs_path = os.path.abspath(img_path)
        if not os.path.exists(abs_path):
            return None

        file_stat = os.stat(abs_path)
        file_key = abs_path.lower()

        existing = self._index.get(file_key)
        if existing:
            same_mtime = float(existing.get("mtime", -1)) == float(file_stat.st_mtime)
            cache_path = existing.get("cache_path", "")
            if same_mtime and cache_path and os.path.exists(cache_path):
                return np.load(cache_path)

        img = cv2.imread(abs_path)
        if img is None:
            return None

        face_crop = self._extract_face_gray(img, face_landmarker)
        if face_crop is None:
            return None

        filename = os.path.basename(abs_path)
        cache_path = self._cache_file_path(person_name, filename)
        original_copy_path = self._original_file_path(person_name, filename)

        np.save(cache_path, face_crop)
        shutil.copy2(abs_path, original_copy_path)

        self._index[file_key] = {
            "name": person_name,
            "source_path": abs_path,
            "original_copy": original_copy_path,
            "cache_path": cache_path,
            "mtime": str(file_stat.st_mtime),
        }
        self._dirty = True

        return face_crop

    def prepare_from_known_faces(self, known_faces_path: str, face_landmarker) -> List[Dict[str, str]]:
        outputs = []

        if not os.path.exists(known_faces_path):
            os.makedirs(known_faces_path)
            return outputs

        for filename in os.listdir(known_faces_path):
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            person_name = os.path.splitext(filename)[0].split("_")[0].upper()
            src_path = os.path.join(known_faces_path, filename)
            face_crop = self.get_or_create_face_crop(src_path, person_name, face_landmarker)
            if face_crop is None:
                continue
            outputs.append({"name": person_name, "source_path": src_path})

        self.flush()
        return outputs
