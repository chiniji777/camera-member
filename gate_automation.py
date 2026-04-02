"""Gate Automation — Turn signal / hazard detection for gate control.

Monitors the camera for orange blinking (turn signal or hazard):
- Blink detected while gate closed/unknown → Open gate
- Blink detected while gate open → Close gate
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
# Car front zone (normalized 0-1)
CAR_FRONT_ZONE = {
    "x1": 0.20, "y1": 0.10,
    "x2": 0.65, "y2": 0.45,
}

# Orange/amber HSV range for turn signals
ORANGE_HSV_LOWER = (5, 100, 150)
ORANGE_HSV_UPPER = (25, 255, 255)

# Detection
MIN_ORANGE_PIXELS = 5000       # min orange pixels to count as "signal ON" (real signal = 10k+)
BLINK_CONFIRM_COUNT = 3        # need 3 blink cycles to confirm (prevent false trigger)
BLINK_WINDOW_SECONDS = 8.0     # time window to accumulate blinks
COOLDOWN_SECONDS = 10.0        # cooldown after command sent

# Timing
CHECK_INTERVAL = 0.25  # 4 fps

# Rate limiting
MIN_COMMAND_INTERVAL = 10.0


class GateState(Enum):
    UNKNOWN = "unknown"
    CLOSED = "closed"
    OPEN = "open"


class GateAutomation:
    """Monitor camera and toggle gate on any orange blink."""

    def __init__(
        self,
        on_gate_command: Callable[[int, str], None],
        stream_base_url: str = "http://localhost:8080",
    ):
        self._on_gate_command = on_gate_command
        self._stream_base_url = stream_base_url
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # State
        self._gate_state = GateState.UNKNOWN
        self._last_gate_action: Optional[float] = None
        self._last_command_time: float = 0

        # Blink detection
        self._signal_was_on = False
        self._blink_count = 0
        self._last_blink_time: float = 0

    def start(self, cam_id: str):
        if self._running:
            return
        self._running = True
        self._cam_id = cam_id
        self._thread = threading.Thread(
            target=self._monitor_loop, args=(cam_id,), daemon=True
        )
        self._thread.start()
        logger.info(f"[GateAuto] Started signal detection for camera {cam_id}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("[GateAuto] Stopped")

    def _grab_frame(self, cam_id: str) -> Optional[np.ndarray]:
        url = f"{self._stream_base_url}/streams/{cam_id}/live"
        try:
            req = urllib.request.Request(url)
            resp = urllib.request.urlopen(req, timeout=5)
            buf = b""
            while len(buf) < 2_000_000:
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

    def _detect_orange(self, frame: np.ndarray) -> int:
        """Detect orange pixel count in car front zone."""
        h, w = frame.shape[:2]
        z = CAR_FRONT_ZONE
        zone = frame[
            int(h * z["y1"]) : int(h * z["y2"]),
            int(w * z["x1"]) : int(w * z["x2"]),
        ]
        hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(ORANGE_HSV_LOWER), np.array(ORANGE_HSV_UPPER))
        return int(np.sum(mask > 0))

    def _can_send_command(self) -> bool:
        if self._last_gate_action is None:
            return True
        return (time.time() - self._last_gate_action) > MIN_COMMAND_INTERVAL

    def _open_gate(self):
        if not self._can_send_command():
            return
        logger.info("[GateAuto] >>> OPENING GATE <<<")
        try:
            self._on_gate_command(0, "เปิดประตู")
            self._gate_state = GateState.OPEN
            self._last_gate_action = time.time()
            self._last_command_time = time.time()
        except Exception as e:
            logger.error(f"[GateAuto] Failed to open gate: {e}")

    def _close_gate(self):
        if not self._can_send_command():
            return
        logger.info("[GateAuto] >>> CLOSING GATE <<<")
        try:
            self._on_gate_command(2, "ปิดประตู")
            self._gate_state = GateState.CLOSED
            self._last_gate_action = time.time()
            self._last_command_time = time.time()
        except Exception as e:
            logger.error(f"[GateAuto] Failed to close gate: {e}")

    def _process_frame(self, now: float, orange_px: int):
        """Track blink cycles and toggle gate."""
        signal_on = orange_px > MIN_ORANGE_PIXELS

        # In cooldown
        if (now - self._last_command_time) < COOLDOWN_SECONDS:
            self._signal_was_on = signal_on
            return

        # Clean old blinks
        if (now - self._last_blink_time) > BLINK_WINDOW_SECONDS:
            self._blink_count = 0

        # Blink = was ON, now OFF
        if self._signal_was_on and not signal_on:
            self._blink_count += 1
            self._last_blink_time = now
            logger.info(f"[GateAuto] Blink ✓ count={self._blink_count}")

        self._signal_was_on = signal_on

        # Trigger when enough blinks
        if self._blink_count >= BLINK_CONFIRM_COUNT:
            if self._gate_state == GateState.OPEN:
                self._close_gate()
            else:
                self._open_gate()
            self._blink_count = 0

    def _monitor_loop(self, cam_id: str):
        logger.info("[GateAuto] Signal detection loop started")
        log_counter = 0

        while self._running:
            try:
                frame = self._grab_frame(cam_id)
                if frame is None:
                    time.sleep(CHECK_INTERVAL)
                    continue

                now = time.time()
                orange_px = self._detect_orange(frame)
                self._process_frame(now, orange_px)

                # Periodic log (~10s)
                log_counter += 1
                if log_counter >= 40:
                    log_counter = 0
                    logger.info(
                        f"[GateAuto] orange={orange_px} "
                        f"blinks={self._blink_count} gate={self._gate_state.value}"
                    )

            except Exception as e:
                logger.error(f"[GateAuto] Error: {e}")

            time.sleep(CHECK_INTERVAL)

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "gate_state": self._gate_state.value,
            "pending_blinks": self._blink_count,
            "mode": "signal_toggle",
        }
