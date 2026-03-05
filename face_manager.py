"""Face Detection & Recognition Manager

Uses OpenCV YuNet (detection) + SFace (recognition) — no extra pip packages needed.
Models are auto-downloaded on first run from OpenCV Zoo (~37MB total).
"""

import json
import uuid
import time
import threading
import logging
import urllib.request
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _local_now() -> datetime:
    """Return current local time with timezone info."""
    return datetime.now().astimezone()


def _local_iso() -> str:
    """Return current local time as ISO string with timezone offset."""
    return _local_now().isoformat()

FACE_DATA_DIR = Path("face_data")
FACES_FILE = FACE_DATA_DIR / "faces.json"
SIGHTINGS_FILE = FACE_DATA_DIR / "sightings.json"
OUTFITS_FILE = FACE_DATA_DIR / "outfits.json"
CROPS_DIR = FACE_DATA_DIR / "crops"
SIGHTINGS_IMG_DIR = FACE_DATA_DIR / "sightings_img"
MODELS_DIR = FACE_DATA_DIR / "models"

YUNET_MODEL = "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = "face_recognition_sface_2021dec.onnx"
YUNET_URL = f"https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/{YUNET_MODEL}"
SFACE_URL = f"https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/{SFACE_MODEL}"

# YOLOv4-tiny for pet/object detection (COCO: dog=16, cat=15, person=0)
YOLO_WEIGHTS = "yolov4-tiny.weights"
YOLO_CFG = "yolov4-tiny.cfg"
YOLO_WEIGHTS_URL = "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights"
YOLO_CFG_URL = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"

# COCO class IDs we care about
COCO_PERSON = 0
COCO_CAT = 15
COCO_DOG = 16
COCO_PET_IDS = {COCO_CAT, COCO_DOG}
COCO_NAMES = {COCO_PERSON: "person", COCO_CAT: "cat", COCO_DOG: "dog"}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Face:
    id: str
    name: str                    # "" for unknown
    encodings: list[list[float]] # list of 128-d embeddings (multi-sample)
    image_path: str              # relative: "face_data/crops/{id}.jpg"
    first_seen: str              # ISO datetime
    last_seen: str
    camera_id: str
    status: str                  # "unknown" | "known" | "ignored"


@dataclass
class Sighting:
    id: str
    face_id: str
    camera_id: str
    timestamp: str           # ISO datetime
    confidence: float
    image_path: str          # "face_data/sightings_img/{id}.jpg"


# ---------------------------------------------------------------------------
# FaceManager — CRUD + matching
# ---------------------------------------------------------------------------

class FaceManager:
    COSINE_THRESHOLD = 0.363  # SFace default; lower = stricter

    def __init__(self):
        self.faces: dict[str, Face] = {}
        self.sightings: list[Sighting] = []
        self._lock = threading.Lock()
        self._ensure_dirs()
        self._load()

    # -- storage --

    def _ensure_dirs(self):
        for d in [FACE_DATA_DIR, CROPS_DIR, SIGHTINGS_IMG_DIR, MODELS_DIR]:
            d.mkdir(parents=True, exist_ok=True)

    def _load(self):
        if FACES_FILE.exists():
            try:
                for item in json.loads(FACES_FILE.read_text()):
                    # Migrate old single-encoding format to multi-encoding
                    if "encoding" in item and "encodings" not in item:
                        item["encodings"] = [item.pop("encoding")]
                    elif "encoding" in item:
                        item.pop("encoding")
                    f = Face(**item)
                    # Migrate naive timestamps to local tz
                    f.first_seen = self._fix_naive_ts(f.first_seen)
                    f.last_seen = self._fix_naive_ts(f.last_seen)
                    self.faces[f.id] = f
            except Exception as e:
                logger.error(f"Failed to load faces.json: {e}")

        if SIGHTINGS_FILE.exists():
            try:
                raw = json.loads(SIGHTINGS_FILE.read_text())
                for s in raw:
                    s["timestamp"] = self._fix_naive_ts(s.get("timestamp", ""))
                self.sightings = [Sighting(**s) for s in raw]
            except Exception as e:
                logger.error(f"Failed to load sightings.json: {e}")

    @staticmethod
    def _fix_naive_ts(ts: str) -> str:
        """Add local timezone to naive ISO timestamps."""
        if not ts:
            return ts
        try:
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.astimezone()
                return dt.isoformat()
        except Exception:
            pass
        return ts

    def _save_faces(self):
        data = [asdict(f) for f in self.faces.values()]
        FACES_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def _save_sightings(self):
        # Keep last 10 000 entries
        if len(self.sightings) > 10_000:
            self.sightings = self.sightings[-10_000:]
        data = [asdict(s) for s in self.sightings]
        SIGHTINGS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    # -- matching --

    def find_match(self, encoding: list[float]) -> tuple[Optional[Face], float]:
        """Return (matched Face, best confidence) or (None, 0.0).
        Checks against ALL encodings per face — more samples = better accuracy."""
        if not self.faces:
            return None, 0.0

        enc = np.array(encoding, dtype=np.float64)
        best_face = None
        best_score = 0.0

        for face in self.faces.values():
            if face.status == "ignored":
                continue
            for known_enc in face.encodings:
                known = np.array(known_enc, dtype=np.float64)
                score = float(np.dot(enc, known) / (np.linalg.norm(enc) * np.linalg.norm(known) + 1e-10))
                if score > self.COSINE_THRESHOLD and score > best_score:
                    best_score = score
                    best_face = face

        return best_face, best_score

    # -- CRUD --

    def add_face(self, encoding: list[float], crop_bytes: bytes, camera_id: str) -> Face:
        """Register a new unknown face."""
        # Check if this is a duplicate of an existing unknown
        existing, score = self.find_match(encoding)
        if existing:
            # Already known/tracked — update last_seen
            with self._lock:
                existing.last_seen = _local_iso()
                self._save_faces()
            return existing

        face_id = str(uuid.uuid4())[:8]
        now = _local_iso()
        img_path = str(CROPS_DIR / f"{face_id}.jpg")

        face = Face(
            id=face_id,
            name="",
            encodings=[encoding],
            image_path=img_path,
            first_seen=now,
            last_seen=now,
            camera_id=camera_id,
            status="unknown",
        )

        Path(img_path).write_bytes(crop_bytes)

        with self._lock:
            self.faces[face_id] = face
            self._save_faces()

        logger.info(f"New face detected: {face_id} on camera {camera_id}")
        return face

    def label_face(self, face_id: str, name: str) -> Optional[Face]:
        with self._lock:
            face = self.faces.get(face_id)
            if not face:
                return None

            # Check if a known face with this name already exists — merge
            existing = None
            for f in self.faces.values():
                if f.id != face_id and f.name == name and f.status == "known":
                    existing = f
                    break

            if existing:
                # Merge encodings into existing person (max 10 samples)
                existing.encodings.extend(face.encodings)
                if len(existing.encodings) > 10:
                    existing.encodings = existing.encodings[-10:]
                existing.last_seen = max(existing.last_seen, face.last_seen)
                # Remove the duplicate entry
                self.faces.pop(face_id)
                try:
                    Path(face.image_path).unlink(missing_ok=True)
                except Exception:
                    pass
                self._save_faces()
                logger.info(f"Merged face {face_id} into '{name}' ({existing.id}, {len(existing.encodings)} samples)")
                return existing
            else:
                face.name = name
                face.status = "known"
                self._save_faces()
                logger.info(f"Labeled face {face_id} as '{name}'")
                return face

    def ignore_face(self, face_id: str) -> Optional[Face]:
        with self._lock:
            face = self.faces.get(face_id)
            if not face:
                return None
            face.status = "ignored"
            self._save_faces()
        return face

    def delete_face(self, face_id: str) -> bool:
        with self._lock:
            face = self.faces.pop(face_id, None)
            if not face:
                return False
            # Remove crop image
            try:
                Path(face.image_path).unlink(missing_ok=True)
            except Exception:
                pass
            self._save_faces()
        return True

    def log_sighting(self, face_id: str, camera_id: str, confidence: float,
                     crop_bytes: Optional[bytes] = None) -> Sighting:
        sid = str(uuid.uuid4())[:8]
        now = _local_iso()
        img_path = ""
        if crop_bytes:
            img_path = str(SIGHTINGS_IMG_DIR / f"{sid}.jpg")
            Path(img_path).write_bytes(crop_bytes)

        sighting = Sighting(
            id=sid,
            face_id=face_id,
            camera_id=camera_id,
            timestamp=now,
            confidence=round(confidence, 3),
            image_path=img_path,
        )

        with self._lock:
            self.sightings.append(sighting)
            # Update face last_seen
            face = self.faces.get(face_id)
            if face:
                face.last_seen = now
            self._save_sightings()
            self._save_faces()

        logger.info(f"Sighting: face {face_id} on camera {camera_id} (conf={confidence:.2f})")
        return sighting

    # -- queries --

    def get_faces(self, status: Optional[str] = None) -> list[Face]:
        faces = list(self.faces.values())
        if status:
            faces = [f for f in faces if f.status == status]
        faces.sort(key=lambda f: f.last_seen, reverse=True)
        return faces

    def get_face(self, face_id: str) -> Optional[Face]:
        return self.faces.get(face_id)

    def get_sightings(self, face_id: Optional[str] = None,
                      camera_id: Optional[str] = None,
                      limit: int = 100) -> list[Sighting]:
        result = self.sightings
        if face_id:
            result = [s for s in result if s.face_id == face_id]
        if camera_id:
            result = [s for s in result if s.camera_id == camera_id]
        result = sorted(result, key=lambda s: s.timestamp, reverse=True)
        return result[:limit]


# ---------------------------------------------------------------------------
# FaceDetector — background worker
# ---------------------------------------------------------------------------

def _download_model(url: str, dest: Path):
    """Download a file with progress logging."""
    if dest.exists():
        return
    logger.info(f"Downloading model: {dest.name} ...")
    try:
        urllib.request.urlretrieve(url, str(dest))
        logger.info(f"Downloaded {dest.name} ({dest.stat().st_size // 1024} KB)")
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        raise


class FaceDetector:
    DETECT_INTERVAL = 0.5   # seconds between scans
    MIN_FACE_SIZE = 80      # min face width px
    SIGHTING_COOLDOWN = 300  # 5 min per person per camera
    GREETING_THRESHOLD = 0.50  # cosine similarity to show greeting
    GREETING_TTL = 6.0      # seconds to keep greeting visible
    AUTO_IGNORE_SECONDS = 30  # auto-ignore unknown faces after this many seconds

    def __init__(self, stream_mgr, face_mgr: FaceManager):
        self._stream_mgr = stream_mgr
        self._face_mgr = face_mgr
        self._layout_mgr = None  # set via set_layout_manager()
        self._running = False
        self._threads: dict[str, threading.Thread] = {}
        self._detector = None
        self._recognizer = None
        self._hog = None
        self._yolo_net = None
        self._yolo_output_layers = None
        self._model_lock = threading.Lock()
        self._last_sighting: dict[tuple[str, str], float] = {}
        self._active_detections: dict[str, list[dict]] = {}  # cam_id -> detections
        # Pet detection results
        self._active_pets: dict[str, list[dict]] = {}  # cam_id -> [{bbox, type, confidence}]
        # Outfit color tracking — resets daily
        self._outfit_data: dict[str, dict] = {}  # face_id -> {"date", "shirt_color", "shirt_hex", "pants_color", "pants_hex"}
        self._outfit_date: str = ""  # current tracking date
        self._outfit_log: dict[str, list[dict]] = {}  # face_id -> [{date, shirt_*, pants_*}]
        self._load_outfit_log()
        # Body tracking — persons with bboxes
        self._tracked_persons: dict[str, list[dict]] = {}  # cam_id -> [{bbox, name, ...}]

    def set_layout_manager(self, layout_mgr):
        """Link to LayoutManager to read detection toggle/sensitivity settings."""
        self._layout_mgr = layout_mgr

    def _ensure_models(self):
        _download_model(YUNET_URL, MODELS_DIR / YUNET_MODEL)
        _download_model(SFACE_URL, MODELS_DIR / SFACE_MODEL)

    def _init_models(self):
        with self._model_lock:
            if self._detector is not None:
                return
            self._ensure_models()

            yunet_path = str(MODELS_DIR / YUNET_MODEL)
            sface_path = str(MODELS_DIR / SFACE_MODEL)

            self._detector = cv2.FaceDetectorYN.create(
                yunet_path, "", (320, 320),
                score_threshold=0.7,
                nms_threshold=0.3,
                top_k=10,
            )
            self._recognizer = cv2.FaceRecognizerSF.create(sface_path, "")

            # HOG person detector (built-in, no download needed)
            self._hog = cv2.HOGDescriptor()
            self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

            # YOLOv4-tiny for pet detection
            self._init_yolo()

            logger.info("Face + body + pet detection models loaded")

    def _init_yolo(self):
        """Load YOLOv4-tiny model for pet/object detection."""
        weights_path = MODELS_DIR / YOLO_WEIGHTS
        cfg_path = MODELS_DIR / YOLO_CFG
        _download_model(YOLO_WEIGHTS_URL, weights_path)
        _download_model(YOLO_CFG_URL, cfg_path)
        try:
            self._yolo_net = cv2.dnn.readNet(str(weights_path), str(cfg_path))
            layer_names = self._yolo_net.getLayerNames()
            out_indices = self._yolo_net.getUnconnectedOutLayers()
            self._yolo_output_layers = [layer_names[i - 1] for i in out_indices.flatten()]
            logger.info("YOLOv4-tiny loaded for pet detection")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self._yolo_net = None

    def start(self, camera_ids: list[str]):
        self._running = True
        self._init_models()
        for cam_id in camera_ids:
            self.add_camera(cam_id)

    def stop(self):
        self._running = False
        self._threads.clear()

    def add_camera(self, cam_id: str):
        if cam_id in self._threads:
            return
        t = threading.Thread(target=self._detect_loop, args=(cam_id,), daemon=True)
        self._threads[cam_id] = t
        t.start()
        logger.info(f"Face detection started for camera {cam_id}")

    def remove_camera(self, cam_id: str):
        self._threads.pop(cam_id, None)
        self._active_detections.pop(cam_id, None)

    def get_active_detections(self, cam_id: str = None) -> list[dict]:
        """Return recent known-face detections (within GREETING_TTL seconds)."""
        now = time.time()
        result = []
        cams = [cam_id] if cam_id else list(self._active_detections.keys())
        for cid in cams:
            detections = self._active_detections.get(cid, [])
            fresh = [d for d in detections if now - d["time"] < self.GREETING_TTL]
            self._active_detections[cid] = fresh
            result.extend(fresh)
        return result

    def _load_outfit_log(self):
        """Load outfit history from disk."""
        if OUTFITS_FILE.exists():
            try:
                self._outfit_log = json.loads(OUTFITS_FILE.read_text())
            except Exception as e:
                logger.error(f"Failed to load outfits.json: {e}")
                self._outfit_log = {}

    def _save_outfit_log(self):
        OUTFITS_FILE.write_text(json.dumps(self._outfit_log, indent=2, ensure_ascii=False))

    def _record_outfit(self, face_id: str, shirt: Optional[dict], pants: Optional[dict]):
        """Record today's outfit (shirt + pants) for a face."""
        today = _local_now().strftime("%Y-%m-%d")
        entry = {"date": today}
        if shirt:
            entry["shirt_color"] = shirt["color_name"]
            entry["shirt_hex"] = shirt["color_hex"]
        if pants:
            entry["pants_color"] = pants["color_name"]
            entry["pants_hex"] = pants["color_hex"]

        # Update in-memory today cache
        self._outfit_data[face_id] = entry

        # Update persistent log (one entry per face per day)
        if face_id not in self._outfit_log:
            self._outfit_log[face_id] = []
        log = self._outfit_log[face_id]
        # Replace today's entry if exists
        log = [e for e in log if e.get("date") != today]
        log.append(entry)
        # Keep last 90 days
        log = log[-90:]
        self._outfit_log[face_id] = log
        self._save_outfit_log()

    def get_outfit(self, face_id: str) -> Optional[dict]:
        """Return today's outfit for a face, or None."""
        today = _local_now().strftime("%Y-%m-%d")
        if self._outfit_date != today:
            return None
        return self._outfit_data.get(face_id)

    def get_all_outfits(self) -> dict[str, dict]:
        """Return all outfit data for today."""
        today = _local_now().strftime("%Y-%m-%d")
        if self._outfit_date != today:
            self._outfit_data.clear()
            self._outfit_date = today
            return {}
        return dict(self._outfit_data)

    def get_outfit_history(self, face_id: str) -> list[dict]:
        """Return outfit history for a face (last 90 days)."""
        return self._outfit_log.get(face_id, [])

    def _check_daily_reset(self):
        """Clear outfit data if the date has changed."""
        today = _local_now().strftime("%Y-%m-%d")
        if self._outfit_date != today:
            self._outfit_data.clear()
            self._outfit_date = today
            logger.info(f"Outfit tracking reset for new day: {today}")

    def _extract_outfit_color(self, img: np.ndarray, bbox) -> Optional[dict]:
        """Extract dominant outfit color from body region below face."""
        x, y, fw, fh = bbox
        h, w = img.shape[:2]

        # Body region: below face, slightly wider, ~1.5x face height
        body_top = min(h, y + fh)
        body_bot = min(h, y + fh + int(fh * 1.5))
        body_left = max(0, x - int(fw * 0.15))
        body_right = min(w, x + fw + int(fw * 0.15))

        if body_bot <= body_top or body_right <= body_left:
            return None
        if body_bot - body_top < 10 or body_right - body_left < 10:
            return None

        body = img[body_top:body_bot, body_left:body_right]
        if body.size == 0:
            return None

        # Use center 60% to avoid background edges
        bh, bw = body.shape[:2]
        margin_x = int(bw * 0.2)
        margin_y = int(bh * 0.2)
        center = body[margin_y:bh - margin_y, margin_x:bw - margin_x]
        if center.size == 0:
            center = body

        # Resize for speed and convert to HSV
        small = cv2.resize(center, (20, 30))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        # Use median for robustness
        med_h = float(np.median(hsv[:, :, 0]))
        med_s = float(np.median(hsv[:, :, 1]))
        med_v = float(np.median(hsv[:, :, 2]))

        # Low saturation → achromatic
        if med_s < 40:
            if med_v < 70:
                return {"color_hex": "#333333", "color_name": "Black"}
            elif med_v < 170:
                return {"color_hex": "#999999", "color_name": "Gray"}
            else:
                return {"color_hex": "#EEEEEE", "color_name": "White"}

        # Map hue (0-180 in OpenCV) to color name
        if med_h < 10 or med_h >= 170:
            return {"color_hex": "#E53935", "color_name": "Red"}
        elif med_h < 22:
            return {"color_hex": "#FF9800", "color_name": "Orange"}
        elif med_h < 35:
            return {"color_hex": "#FDD835", "color_name": "Yellow"}
        elif med_h < 78:
            return {"color_hex": "#43A047", "color_name": "Green"}
        elif med_h < 130:
            return {"color_hex": "#1E88E5", "color_name": "Blue"}
        elif med_h < 155:
            return {"color_hex": "#8E24AA", "color_name": "Purple"}
        else:
            return {"color_hex": "#EC407A", "color_name": "Pink"}

    def _extract_pants_color(self, img: np.ndarray, bbox) -> Optional[dict]:
        """Extract pants color from lower body region below face."""
        x, y, fw, fh = bbox
        h, w = img.shape[:2]

        # Pants region: ~2.5-4.5x face height below face top
        pants_top = min(h, y + int(fh * 2.5))
        pants_bot = min(h, y + int(fh * 4.5))
        pants_left = max(0, x - int(fw * 0.2))
        pants_right = min(w, x + fw + int(fw * 0.2))

        if pants_bot - pants_top < 10 or pants_right - pants_left < 10:
            return None

        return self._extract_color_from_rect(img, pants_left, pants_top, pants_right, pants_bot)

    def _extract_color_from_rect(self, img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> Optional[dict]:
        """Extract dominant color from an arbitrary image rectangle."""
        ih, iw = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(iw, x2), min(ih, y2)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None
        region = img[y1:y2, x1:x2]
        if region.size == 0:
            return None
        rh, rw = region.shape[:2]
        mx, my = int(rw * 0.2), int(rh * 0.2)
        center = region[my:rh - my, mx:rw - mx]
        if center.size == 0:
            center = region
        small = cv2.resize(center, (20, 30))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        med_h = float(np.median(hsv[:, :, 0]))
        med_s = float(np.median(hsv[:, :, 1]))
        med_v = float(np.median(hsv[:, :, 2]))
        if med_s < 40:
            if med_v < 70:
                return {"color_hex": "#333333", "color_name": "Black"}
            elif med_v < 170:
                return {"color_hex": "#999999", "color_name": "Gray"}
            else:
                return {"color_hex": "#EEEEEE", "color_name": "White"}
        if med_h < 10 or med_h >= 170:
            return {"color_hex": "#E53935", "color_name": "Red"}
        elif med_h < 22:
            return {"color_hex": "#FF9800", "color_name": "Orange"}
        elif med_h < 35:
            return {"color_hex": "#FDD835", "color_name": "Yellow"}
        elif med_h < 78:
            return {"color_hex": "#43A047", "color_name": "Green"}
        elif med_h < 130:
            return {"color_hex": "#1E88E5", "color_name": "Blue"}
        elif med_h < 155:
            return {"color_hex": "#8E24AA", "color_name": "Purple"}
        else:
            return {"color_hex": "#EC407A", "color_name": "Pink"}

    def _face_in_body(self, face_bbox, body_bbox) -> bool:
        """Check if face center is within body bounding box."""
        fx, fy, fw, fh = face_bbox
        bx, by, bw, bh = body_bbox
        fcx = fx + fw / 2
        fcy = fy + fh / 2
        return bx <= fcx <= bx + bw and by <= fcy <= by + bh

    def _match_outfit_to_person(self, color_name: str) -> tuple[Optional[str], Optional[str]]:
        """Match body outfit color to a known person. Returns (name, face_id) or (None, None)."""
        if not color_name:
            return None, None
        candidates = []
        for face_id, outfit in self._outfit_data.items():
            if outfit.get("shirt_color", outfit.get("color_name", "")) == color_name:
                face = self._face_mgr.get_face(face_id)
                if face and face.name and face.status == "known":
                    candidates.append((face.name, face_id))
        if len(candidates) == 1:
            return candidates[0]
        return None, None

    def get_tracked_persons(self, cam_id: str = None) -> list[dict]:
        """Return currently tracked persons with bounding boxes."""
        if cam_id:
            return list(self._tracked_persons.get(cam_id, []))
        result = []
        for persons in self._tracked_persons.values():
            result.extend(persons)
        return result

    def get_active_pets(self, cam_id: str = None) -> list[dict]:
        """Return currently detected pets with bounding boxes."""
        if cam_id:
            return list(self._active_pets.get(cam_id, []))
        result = []
        for pets in self._active_pets.values():
            result.extend(pets)
        return result

    def _detect_pets(self, cam_id: str, img: np.ndarray) -> list[dict]:
        """Run YOLOv4-tiny to detect cats and dogs. Returns list of detections."""
        if self._yolo_net is None:
            return []

        h, w = img.shape[:2]

        # Sensitivity maps to confidence threshold: sens=0 → 0.7, sens=100 → 0.15
        sens = 50
        if self._layout_mgr:
            sens = self._layout_mgr.get_sensitivity("pet")
        conf_threshold = 0.7 - sens * 0.0055  # range: 0.7 (sens=0) to 0.15 (sens=100)
        conf_threshold = max(0.1, conf_threshold)

        # Prepare input blob (416x416)
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self._yolo_net.setInput(blob)
        outputs = self._yolo_net.forward(self._yolo_output_layers)

        pets = []
        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])

                if class_id not in COCO_PET_IDS:
                    continue
                if confidence < conf_threshold:
                    continue

                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                bw = int(detection[2] * w)
                bh = int(detection[3] * h)
                x = center_x - bw // 2
                y = center_y - bh // 2

                boxes.append([x, y, bw, bh])
                confidences.append(confidence)
                class_ids.append(class_id)

        # NMS to remove duplicates
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.4)
            for i in indices.flatten():
                x, y, bw, bh = boxes[i]
                pets.append({
                    "bbox": [round(x / w, 4), round(y / h, 4),
                             round(bw / w, 4), round(bh / h, 4)],
                    "type": COCO_NAMES.get(class_ids[i], "pet"),
                    "confidence": round(confidences[i], 3),
                    "camera_id": cam_id,
                })

        self._active_pets[cam_id] = pets
        return pets

    def _detect_loop(self, cam_id: str):
        # Wait for first frame to be available
        for _ in range(20):
            if self._stream_mgr.get_frame(cam_id):
                break
            time.sleep(0.5)

        while self._running and cam_id in self._threads:
            try:
                frame_bytes = self._stream_mgr.get_frame(cam_id)
                if frame_bytes:
                    self._process_frame(cam_id, frame_bytes)
            except Exception as e:
                logger.error(f"Face detection error on {cam_id}: {e}")
            time.sleep(self.DETECT_INTERVAL)

    def _is_enabled(self, feature: str) -> bool:
        """Check if a detection feature is enabled via layout manager."""
        if self._layout_mgr:
            return self._layout_mgr.is_detection_enabled(feature)
        # Default: face and human on, others off
        return feature in ("face", "human")

    def _process_frame(self, cam_id: str, frame_bytes: bytes):
        # Daily outfit reset check
        self._check_daily_reset()

        # Auto-ignore unknown faces that have been sitting for 30+ seconds
        if self._is_enabled("face"):
            now_dt = _local_now()
            changed = False
            with self._face_mgr._lock:
                for face in list(self._face_mgr.faces.values()):
                    if face.status == "unknown":
                        try:
                            first = datetime.fromisoformat(face.first_seen)
                            if first.tzinfo is None:
                                first = first.astimezone()
                            if (now_dt - first).total_seconds() >= self.AUTO_IGNORE_SECONDS:
                                face.status = "ignored"
                                changed = True
                                logger.info(f"Auto-ignored face {face.id} (unlabeled for {self.AUTO_IGNORE_SECONDS}s)")
                        except Exception:
                            pass
                if changed:
                    self._face_mgr._save_faces()

        # Decode JPEG
        nparr = np.frombuffer(frame_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return

        h, w = img.shape[:2]

        # ── Phase 0: Pet detection (YOLO) ──
        if self._is_enabled("pet"):
            self._detect_pets(cam_id, img)
        else:
            self._active_pets.pop(cam_id, None)

        # ── Phase 1: Face detection ──
        frame_faces = []  # collect face results for body linking

        if not self._is_enabled("face"):
            # Skip face detection entirely
            self._active_detections.pop(cam_id, None)
        else:
            self._detector.setInputSize((w, h))

        faces = None
        if self._is_enabled("face"):
            # Apply sensitivity: map 0-100 to score_threshold 0.9-0.3
            sens = 50
            if self._layout_mgr:
                sens = self._layout_mgr.get_sensitivity("face")
            face_score = 0.9 - sens * 0.006  # range: 0.9 (sens=0) to 0.3 (sens=100)
            self._detector.setScoreThreshold(max(0.2, face_score))
            _, faces = self._detector.detect(img)

        if faces is not None:
            for face_data in faces:
                bbox = face_data[:4].astype(int)
                x, y, fw, fh = bbox

                if fw < self.MIN_FACE_SIZE:
                    continue

                aligned = self._recognizer.alignCrop(img, face_data)
                embedding = self._recognizer.feature(aligned)
                enc_list = embedding.flatten().tolist()

                pad = int(max(fw, fh) * 0.2)
                cx = max(0, x - pad)
                cy = max(0, y - pad)
                cx2 = min(w, x + fw + pad)
                cy2 = min(h, y + fh + pad)
                crop = img[cy:cy2, cx:cx2]
                _, crop_jpg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                crop_bytes = crop_jpg.tobytes()

                shirt = self._extract_outfit_color(img, (x, y, fw, fh))
                pants = self._extract_pants_color(img, (x, y, fw, fh))
                matched, confidence = self._face_mgr.find_match(enc_list)

                if matched:
                    key = (matched.id, cam_id)
                    now = time.time()
                    last = self._last_sighting.get(key, 0)
                    if now - last > self.SIGHTING_COOLDOWN:
                        self._face_mgr.log_sighting(matched.id, cam_id, confidence, crop_bytes)
                        self._last_sighting[key] = now
                    else:
                        matched.last_seen = _local_iso()

                    if shirt or pants:
                        self._record_outfit(matched.id, shirt, pants)

                    if matched.name and confidence >= self.GREETING_THRESHOLD:
                        detection = {
                            "face_id": matched.id, "name": matched.name,
                            "confidence": round(confidence, 3), "camera_id": cam_id,
                            "time": time.time(),
                        }
                        face_outfit = self._outfit_data.get(matched.id)
                        if face_outfit:
                            detection["outfit_color"] = face_outfit.get("shirt_color", "")
                            detection["outfit_hex"] = face_outfit.get("shirt_hex", "")
                        self._active_detections.setdefault(cam_id, []).append(detection)

                    frame_faces.append({
                        "bbox": (x, y, fw, fh), "face_id": matched.id,
                        "name": matched.name, "confidence": confidence,
                    })
                else:
                    new_face = self._face_mgr.add_face(enc_list, crop_bytes, cam_id)
                    if shirt or pants:
                        self._record_outfit(new_face.id, shirt, pants)

                    if new_face.status == "unknown":
                        detection = {
                            "face_id": new_face.id, "name": "New Customer",
                            "confidence": 0.0, "camera_id": cam_id, "time": time.time(),
                        }
                        if shirt:
                            detection["outfit_color"] = shirt["color_name"]
                            detection["outfit_hex"] = shirt["color_hex"]
                        self._active_detections.setdefault(cam_id, []).append(detection)

                    frame_faces.append({
                        "bbox": (x, y, fw, fh), "face_id": new_face.id,
                        "name": new_face.name or "New Customer", "confidence": 0,
                    })

        # ── Phase 2: Body detection (HOG) — conditional on human detection toggle ──
        bodies = None
        weights = []
        if self._is_enabled("human"):
            detect_h = 320
            scale_factor = h / detect_h
            detect_w = int(w * detect_h / h)
            small_frame = cv2.resize(img, (detect_w, detect_h))

            # Apply sensitivity: map 0-100 to hitThreshold 0.8-0.0
            sens = 50
            if self._layout_mgr:
                sens = self._layout_mgr.get_sensitivity("human")
            hit_thresh = 0.8 - sens * 0.008  # range: 0.8 (sens=0) to 0.0 (sens=100)

            bodies, weights = self._hog.detectMultiScale(
                small_frame, winStride=(8, 8), padding=(4, 4), scale=1.05,
                hitThreshold=max(0.0, hit_thresh),
            )
        else:
            detect_h = 320
            scale_factor = h / detect_h

        tracked = []
        used_faces = set()

        if bodies is not None and len(bodies) > 0:
            for i, (bx, by, bw, bh) in enumerate(bodies):
                # Filter low-confidence detections
                conf = float(weights[i]) if i < len(weights) else 0
                if conf < 0.4:
                    continue

                # Scale back to original coords
                bx_o = int(bx * scale_factor)
                by_o = int(by * scale_factor)
                bw_o = int(bw * scale_factor)
                bh_o = int(bh * scale_factor)

                # Filter: body must be taller than wide (aspect ratio > 1.2)
                if bh_o < bw_o * 1.2:
                    continue

                # Filter: body must be minimum size (at least 5% of frame height)
                if bh_o < h * 0.05:
                    continue

                person_name = ""
                person_face_id = ""
                match_method = "body"
                person_conf = conf

                # Try linking to a detected face
                for fi, ff in enumerate(frame_faces):
                    if fi in used_faces:
                        continue
                    if self._face_in_body(ff["bbox"], (bx_o, by_o, bw_o, bh_o)):
                        person_name = ff["name"]
                        person_face_id = ff["face_id"]
                        match_method = "face"
                        person_conf = ff["confidence"]
                        used_faces.add(fi)
                        break

                # If no face, try outfit color matching
                if not person_name or person_name == "New Customer":
                    torso_top = by_o + int(bh_o * 0.2)
                    torso_bot = by_o + int(bh_o * 0.6)
                    torso_left = bx_o + int(bw_o * 0.15)
                    torso_right = bx_o + bw_o - int(bw_o * 0.15)
                    body_color = self._extract_color_from_rect(img, torso_left, torso_top, torso_right, torso_bot)
                    if body_color:
                        name, fid = self._match_outfit_to_person(body_color["color_name"])
                        if name:
                            person_name = name
                            person_face_id = fid
                            match_method = "outfit"

                # Get first_seen for countdown on unknown faces
                first_seen = ""
                if person_face_id:
                    face_obj = self._face_mgr.get_face(person_face_id)
                    if face_obj:
                        first_seen = face_obj.first_seen or ""

                tracked.append({
                    "bbox": [round(bx_o / w, 4), round(by_o / h, 4),
                             round(bw_o / w, 4), round(bh_o / h, 4)],
                    "name": person_name or "",
                    "face_id": person_face_id,
                    "method": match_method,
                    "confidence": round(person_conf, 3),
                    "camera_id": cam_id,
                    "first_seen": first_seen,
                })

        # Also add faces that weren't linked to any body (close-up face only)
        for fi, ff in enumerate(frame_faces):
            if fi not in used_faces:
                fx, fy, fw_f, fh_f = ff["bbox"]
                # Estimate body bbox from face position
                est_bw = int(fw_f * 2.5)
                est_bh = int(fh_f * 5)
                est_bx = fx + fw_f // 2 - est_bw // 2
                est_by = fy
                first_seen = ""
                if ff["face_id"]:
                    face_obj = self._face_mgr.get_face(ff["face_id"])
                    if face_obj:
                        first_seen = face_obj.first_seen or ""

                tracked.append({
                    "bbox": [round(max(0, est_bx) / w, 4), round(max(0, est_by) / h, 4),
                             round(min(est_bw, w) / w, 4), round(min(est_bh, h) / h, 4)],
                    "name": ff["name"] or "",
                    "face_id": ff["face_id"],
                    "method": "face",
                    "confidence": round(ff["confidence"], 3),
                    "camera_id": cam_id,
                    "first_seen": first_seen,
                })

        self._tracked_persons[cam_id] = tracked
