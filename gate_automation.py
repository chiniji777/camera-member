"""Gate Automation — Vision-based gate control using camera feed.

Monitors the camera for:
1. Green car in parking zone + headlights ON → Open gate
2. Car leaves parking zone + gate still open → Close gate

Runs as a background thread, analyzing frames from the MJPEG stream.
"""
import cv2
import numpy as np
import threading
import time
import logging
import urllib.request
from typing import Callable, Optional
from enum import Enum

logger = logging.getLogger(__name__)

# --- Configuration ---
# Parking zone (normalized 0-1 coordinates relative to frame)
PARKING_ZONE = {
    "x1": 0.30, "y1": 0.15,
    "x2": 0.82, "y2": 0.90,
}

# HSV range for olive-green car
GREEN_CAR_HSV = {
    "lower": (15, 8, 50),
    "upper": (60, 100, 210),
}

# Minimum contour area to be considered a car (pixels)
MIN_CAR_AREA = 100000

# Car movement detection
# Track centroid of car contour — if it moves significantly, car is moving
CAR_MOVEMENT_THRESHOLD = 30  # pixels centroid must move to count as "moving"
CAR_MOVING_CONFIRM_FRAMES = 3  # must see movement in N consecutive checks

# Timing
CHECK_INTERVAL = 2.0  # seconds between frame checks
CAR_GONE_CONFIRM_SECONDS = 15  # wait N seconds to confirm car actually left
GATE_CLOSE_DELAY = 10  # seconds after car leaves to close gate
HEADLIGHT_CONFIRM_SECONDS = 3  # headlights must be on for N seconds


class GateState(Enum):
    UNKNOWN = "unknown"
    CLOSED = "closed"
    OPEN = "open"


class GateAutomation:
    """Monitor camera and auto-control gate based on car detection."""

    def __init__(
        self,
        on_gate_command: Callable[[int, str], None],
        stream_base_url: str = "http://localhost:8080",
    ):
        self._on_gate_command = on_gate_command
        self._stream_base_url = stream_base_url
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # State tracking
        self._gate_state = GateState.UNKNOWN
        self._car_present = False
        self._car_moving = False
        self._car_gone_since: Optional[float] = None
        self._gate_opened_at: Optional[float] = None
        self._last_gate_action: Optional[float] = None

        # Car movement tracking
        self._last_car_centroid: Optional[tuple] = None
        self._movement_count = 0  # consecutive frames with movement

        # Prevent rapid gate commands (min 10s between commands)
        self._min_command_interval = 10.0

    def start(self, cam_id: str):
        """Start monitoring a camera."""
        if self._running:
            return
        self._running = True
        self._cam_id = cam_id
        self._thread = threading.Thread(
            target=self._monitor_loop, args=(cam_id,), daemon=True
        )
        self._thread.start()
        logger.info(f"[GateAuto] Started monitoring camera {cam_id}")

    def stop(self):
        """Stop monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("[GateAuto] Stopped")

    def _grab_frame(self, cam_id: str) -> Optional[np.ndarray]:
        """Grab a single frame from MJPEG stream."""
        url = f"{self._stream_base_url}/streams/{cam_id}/live"
        try:
            req = urllib.request.Request(url)
            resp = urllib.request.urlopen(req, timeout=5)
            buf = b""
            while len(buf) < 2_000_000:  # max 2MB
                chunk = resp.read(8192)
                if not chunk:
                    break
                buf += chunk
                start = buf.find(b"\xff\xd8")
                end = buf.find(b"\xff\xd9", start + 2) if start != -1 else -1
                if start != -1 and end != -1:
                    frame_bytes = buf[start : end + 2]
                    arr = np.frombuffer(frame_bytes, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    resp.close()
                    return frame
            resp.close()
        except Exception as e:
            logger.debug(f"[GateAuto] Frame grab failed: {e}")
        return None

    def _detect_green_car(self, frame: np.ndarray) -> bool:
        """Detect green car in parking zone."""
        h, w = frame.shape[:2]
        z = PARKING_ZONE
        zone = frame[
            int(h * z["y1"]) : int(h * z["y2"]),
            int(w * z["x1"]) : int(w * z["x2"]),
        ]

        hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
        lower = np.array(GREEN_CAR_HSV["lower"])
        upper = np.array(GREEN_CAR_HSV["upper"])
        mask = cv2.inRange(hsv, lower, upper)

        # Morphological cleanup
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big = [c for c in contours if cv2.contourArea(c) > MIN_CAR_AREA]

        return len(big) > 0

    def _detect_car_position(self, frame: np.ndarray) -> Optional[tuple]:
        """Get centroid of detected green car. Returns (cx, cy) or None."""
        h, w = frame.shape[:2]
        z = PARKING_ZONE
        y1, y2 = int(h * z["y1"]), int(h * z["y2"])
        x1, x2 = int(w * z["x1"]), int(w * z["x2"])
        zone = frame[y1:y2, x1:x2]

        hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
        lower = np.array(GREEN_CAR_HSV["lower"])
        upper = np.array(GREEN_CAR_HSV["upper"])
        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        big = [c for c in contours if cv2.contourArea(c) > MIN_CAR_AREA]

        if not big:
            return None

        # Use the largest contour
        largest = max(big, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"]) + x1
        cy = int(M["m01"] / M["m00"]) + y1
        return (cx, cy)

    def _detect_car_moving(self, frame: np.ndarray) -> bool:
        """Detect if car is moving by tracking centroid shift."""
        pos = self._detect_car_position(frame)
        if pos is None:
            self._last_car_centroid = None
            self._movement_count = 0
            return False

        if self._last_car_centroid is None:
            self._last_car_centroid = pos
            return False

        dx = abs(pos[0] - self._last_car_centroid[0])
        dy = abs(pos[1] - self._last_car_centroid[1])
        distance = (dx * dx + dy * dy) ** 0.5

        self._last_car_centroid = pos

        if distance > CAR_MOVEMENT_THRESHOLD:
            self._movement_count += 1
            logger.debug(f"[GateAuto] Car moved {distance:.0f}px (count: {self._movement_count})")
            return self._movement_count >= CAR_MOVING_CONFIRM_FRAMES
        else:
            self._movement_count = max(0, self._movement_count - 1)
            return False

    def _can_send_command(self) -> bool:
        """Rate-limit gate commands."""
        if self._last_gate_action is None:
            return True
        return (time.time() - self._last_gate_action) > self._min_command_interval

    def _open_gate(self):
        """Send open gate command."""
        if not self._can_send_command():
            return
        logger.info("[GateAuto] >>> OPENING GATE <<<")
        try:
            self._on_gate_command(0, "เปิดประตู")
            self._gate_state = GateState.OPEN
            self._gate_opened_at = time.time()
            self._last_gate_action = time.time()
        except Exception as e:
            logger.error(f"[GateAuto] Failed to open gate: {e}")

    def _close_gate(self):
        """Send close gate command."""
        if not self._can_send_command():
            return
        logger.info("[GateAuto] >>> CLOSING GATE <<<")
        try:
            self._on_gate_command(2, "ปิดประตู")
            self._gate_state = GateState.CLOSED
            self._gate_opened_at = None
            self._last_gate_action = time.time()
        except Exception as e:
            logger.error(f"[GateAuto] Failed to close gate: {e}")

    def _monitor_loop(self, cam_id: str):
        """Main monitoring loop."""
        logger.info("[GateAuto] Monitor loop started")

        while self._running:
            try:
                frame = self._grab_frame(cam_id)
                if frame is None:
                    time.sleep(CHECK_INTERVAL)
                    continue

                now = time.time()
                car_detected = self._detect_green_car(frame)
                car_moving = self._detect_car_moving(frame) if car_detected else False

                # --- State transitions ---

                # Car appeared
                if car_detected and not self._car_present:
                    logger.info("[GateAuto] Car detected in parking zone")
                    self._car_present = True
                    self._car_gone_since = None

                # Car disappeared
                if not car_detected and self._car_present:
                    if self._car_gone_since is None:
                        self._car_gone_since = now
                        logger.info("[GateAuto] Car may have left, confirming...")
                    elif (now - self._car_gone_since) > CAR_GONE_CONFIRM_SECONDS:
                        logger.info("[GateAuto] Car confirmed GONE from parking zone")
                        self._car_present = False
                        self._car_moving = False
                        self._movement_count = 0
                elif car_detected:
                    self._car_gone_since = None

                # Car started moving
                if car_moving and not self._car_moving:
                    logger.info("[GateAuto] Car is MOVING — preparing to leave")
                    self._car_moving = True

                # Car stopped moving
                if not car_moving and self._car_moving and car_detected:
                    # Only reset if car has been still for a while
                    pass  # keep _car_moving true until car leaves

                # --- Actions ---

                # Rule 1: Car moving + gate not open → OPEN
                if (
                    self._car_moving
                    and self._gate_state != GateState.OPEN
                ):
                    self._open_gate()

                # Rule 2: Car gone + gate open → CLOSE (after delay)
                if (
                    not self._car_present
                    and self._gate_state == GateState.OPEN
                    and self._gate_opened_at
                    and (now - self._gate_opened_at) > GATE_CLOSE_DELAY
                ):
                    self._close_gate()

                # Periodic status log (every 10s)
                if int(now) % 10 == 0:
                    logger.info(
                        f"[GateAuto] car={self._car_present} "
                        f"moving={self._car_moving} "
                        f"gate={self._gate_state.value}"
                    )

            except Exception as e:
                logger.error(f"[GateAuto] Error: {e}")

            time.sleep(CHECK_INTERVAL)

    def get_status(self) -> dict:
        """Get current automation status."""
        return {
            "running": self._running,
            "car_present": self._car_present,
            "car_moving": self._car_moving,
            "gate_state": self._gate_state.value,
        }
