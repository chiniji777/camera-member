"""Motion Detection via MOG2 Background Subtraction.

Uses OpenCV's MOG2 (Mixture of Gaussians) for adaptive background modeling.
Learns the background automatically — resistant to lighting changes and shadows.
"""
import time
import logging
import threading
from typing import Callable

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class MotionDetector:
    """Detects motion using MOG2 background subtraction per camera."""

    def __init__(self, sensitivity: int = 50, debounce_sec: float = 2.0):
        """
        sensitivity: 0-100 scale. Internally mapped to MOG2 thresholds.
            0 = least sensitive (needs huge change), 100 = most sensitive (tiny change triggers).
        """
        self.debounce_sec = debounce_sec
        self._sensitivity = sensitivity
        self._bg_subtractors: dict[str, cv2.BackgroundSubtractorMOG2] = {}  # cam_id -> MOG2
        self._last_motion: dict[str, float] = {}
        self._callbacks: list[Callable] = []
        self._enabled = False
        self._lock = threading.Lock()

    @property
    def sensitivity(self) -> int:
        return self._sensitivity

    @sensitivity.setter
    def sensitivity(self, value: int):
        self._sensitivity = max(0, min(100, value))

    def _get_change_threshold(self) -> float:
        """Convert 0-100 sensitivity to change_fraction_threshold.
        Higher sensitivity = lower threshold = easier to trigger.
        """
        change_frac = 0.15 - self._sensitivity * 0.0014  # range: 0.15 (sens=0) to 0.01 (sens=100)
        return max(0.005, change_frac)

    def _get_or_create_subtractor(self, cam_id: str) -> cv2.BackgroundSubtractorMOG2:
        """Get or create a MOG2 background subtractor for a camera."""
        if cam_id not in self._bg_subtractors:
            # varThreshold: higher = less sensitive to small changes
            # detectShadows: True to detect and mark shadows separately
            var_threshold = int(60 - self._sensitivity * 0.4)  # range: 60 (sens=0) to 20 (sens=100)
            var_threshold = max(10, var_threshold)
            subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500,
                varThreshold=var_threshold,
                detectShadows=True,
            )
            self._bg_subtractors[cam_id] = subtractor
        return self._bg_subtractors[cam_id]

    def set_enabled(self, enabled: bool):
        self._enabled = enabled
        if not enabled:
            with self._lock:
                self._bg_subtractors.clear()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def on_motion(self, callback: Callable):
        """Register callback: callback(camera_id, intensity)"""
        self._callbacks.append(callback)

    def check_frame(self, cam_id: str, jpeg_bytes: bytes):
        """Called with each new JPEG frame. Uses MOG2 to detect motion."""
        if not self._enabled:
            return

        # Decode JPEG to color (MOG2 works better with color for shadow detection)
        nparr = np.frombuffer(jpeg_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return

        # Downsample for speed (320px wide)
        h, w = img.shape[:2]
        if w > 320:
            scale = 320 / w
            img = cv2.resize(img, (320, int(h * scale)))

        with self._lock:
            subtractor = self._get_or_create_subtractor(cam_id)

        # Apply MOG2 — returns foreground mask
        # Pixels: 0 = background, 127 = shadow, 255 = foreground
        fg_mask = subtractor.apply(img)

        # Only count definite foreground (255), ignore shadows (127)
        foreground_pixels = np.count_nonzero(fg_mask == 255)
        total_pixels = fg_mask.size
        intensity = foreground_pixels / total_pixels

        change_thresh = self._get_change_threshold()

        if intensity >= change_thresh:
            now = time.time()
            last = self._last_motion.get(cam_id, 0)
            if now - last >= self.debounce_sec:
                self._last_motion[cam_id] = now
                logger.debug(f"Motion on {cam_id}: {intensity:.2%} (thresh={change_thresh:.3f}, MOG2)")
                for cb in self._callbacks:
                    try:
                        cb(cam_id, intensity)
                    except Exception as e:
                        logger.error(f"Motion callback error: {e}")
