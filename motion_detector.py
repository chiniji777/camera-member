"""Motion Detection via frame pixel difference.

Compares consecutive JPEG frames per camera, calculates the fraction of
changed pixels, and fires callbacks when motion exceeds the sensitivity threshold.
"""
import io
import time
import logging
import threading
from typing import Optional, Callable

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class MotionDetector:
    """Detects motion by comparing consecutive JPEG frames per camera."""

    def __init__(self, sensitivity: int = 50, debounce_sec: float = 2.0):
        """
        sensitivity: 0-100 scale. Internally mapped to a pixel-change threshold.
            0 = least sensitive (needs huge change), 100 = most sensitive (tiny change triggers).
        """
        self.debounce_sec = debounce_sec
        self._sensitivity = sensitivity
        self._prev_frames: dict[str, np.ndarray] = {}  # cam_id -> grayscale array
        self._last_motion: dict[str, float] = {}        # cam_id -> timestamp
        self._callbacks: list[Callable] = []
        self._enabled = False
        self._lock = threading.Lock()

    @property
    def sensitivity(self) -> int:
        return self._sensitivity

    @sensitivity.setter
    def sensitivity(self, value: int):
        self._sensitivity = max(0, min(100, value))

    def _get_threshold(self) -> tuple[int, float]:
        """Convert 0-100 sensitivity to (pixel_diff_threshold, change_fraction_threshold).
        Higher sensitivity = lower thresholds = easier to trigger.
        """
        # pixel_diff: how much a single pixel must change (30 at sens=50)
        pixel_diff = int(60 - self._sensitivity * 0.5)  # range: 60 (sens=0) to 10 (sens=100)
        pixel_diff = max(5, pixel_diff)

        # change_fraction: what % of pixels must change
        change_frac = 0.15 - self._sensitivity * 0.0014  # range: 0.15 (sens=0) to 0.01 (sens=100)
        change_frac = max(0.005, change_frac)

        return pixel_diff, change_frac

    def set_enabled(self, enabled: bool):
        self._enabled = enabled
        if not enabled:
            with self._lock:
                self._prev_frames.clear()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def on_motion(self, callback: Callable):
        """Register callback: callback(camera_id, intensity)"""
        self._callbacks.append(callback)

    def check_frame(self, cam_id: str, jpeg_bytes: bytes):
        """Called with each new JPEG frame. Compares with previous frame."""
        if not self._enabled:
            return

        # Decode JPEG to grayscale
        nparr = np.frombuffer(jpeg_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return

        # Downsample for speed (160px wide)
        h, w = img.shape
        if w > 160:
            scale = 160 / w
            img = cv2.resize(img, (160, int(h * scale)))

        with self._lock:
            prev = self._prev_frames.get(cam_id)
            self._prev_frames[cam_id] = img

        if prev is None:
            return

        # Ensure same shape
        if prev.shape != img.shape:
            return

        # Calculate pixel difference
        pixel_thresh, change_thresh = self._get_threshold()
        diff = cv2.absdiff(prev, img)
        changed = np.count_nonzero(diff > pixel_thresh)
        intensity = changed / diff.size

        if intensity >= change_thresh:
            now = time.time()
            last = self._last_motion.get(cam_id, 0)
            if now - last >= self.debounce_sec:
                self._last_motion[cam_id] = now
                logger.debug(f"Motion on {cam_id}: {intensity:.2%} (thresh={change_thresh:.3f})")
                for cb in self._callbacks:
                    try:
                        cb(cam_id, intensity)
                    except Exception as e:
                        logger.error(f"Motion callback error: {e}")
