import json
import uuid
import subprocess
import shutil
import time
import threading
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

logger = logging.getLogger(__name__)

CAMERAS_FILE = Path("cameras.json")
STREAMS_DIR = Path("streams")


@dataclass
class Camera:
    id: str
    name: str
    url: str          # RTSP URL or MJPEG agent URL
    type: str          # "rtsp" or "mjpeg"
    enabled: bool = True


class CameraManager:
    def __init__(self):
        self.cameras: dict[str, Camera] = {}
        self._load()

    def _load(self):
        if CAMERAS_FILE.exists():
            try:
                data = json.loads(CAMERAS_FILE.read_text())
                for item in data:
                    cam = Camera(**item)
                    self.cameras[cam.id] = cam
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"Failed to load cameras.json: {e}")

    def _save(self):
        data = [asdict(cam) for cam in self.cameras.values()]
        CAMERAS_FILE.write_text(json.dumps(data, indent=2))

    def add(self, name: str, url: str, cam_type: str) -> Camera:
        cam_id = str(uuid.uuid4())[:8]
        cam = Camera(id=cam_id, name=name, url=url, type=cam_type)
        self.cameras[cam_id] = cam
        self._save()
        return cam

    def remove(self, cam_id: str) -> bool:
        if cam_id in self.cameras:
            del self.cameras[cam_id]
            self._save()
            return True
        return False

    def list_all(self) -> list[Camera]:
        return list(self.cameras.values())

    def get(self, cam_id: str) -> Optional[Camera]:
        return self.cameras.get(cam_id)


def _find_ffmpeg() -> Optional[str]:
    path = shutil.which("ffmpeg")
    if path:
        return path
    fallback = Path("C:/ffmpeg/ffmpeg-8.0.1-essentials_build/bin/ffmpeg.exe")
    if fallback.exists():
        return str(fallback)
    return None


class StreamManager:
    """Manages FFmpeg processes that decode RTSP to MJPEG frames via pipe."""

    def __init__(self):
        self.processes: dict[str, subprocess.Popen] = {}
        self._restart_counts: dict[str, int] = {}
        self._running = True
        self._frames: dict[str, bytes] = {}       # latest JPEG per camera
        self._frame_locks: dict[str, threading.Lock] = {}
        self._cameras: dict[str, Camera] = {}      # keep ref for restart
        STREAMS_DIR.mkdir(exist_ok=True)

    def start_stream(self, camera: Camera):
        if camera.type != "rtsp":
            return

        ffmpeg_path = _find_ffmpeg()
        if not ffmpeg_path:
            logger.error("FFmpeg not found. Install FFmpeg to use RTSP cameras.")
            return

        self._cameras[camera.id] = camera
        self._frame_locks.setdefault(camera.id, threading.Lock())

        stream_dir = STREAMS_DIR / camera.id
        stream_dir.mkdir(exist_ok=True)
        ffmpeg_log = stream_dir / "ffmpeg.log"

        cmd = [
            ffmpeg_path,
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-rtsp_transport", "tcp",
            "-i", camera.url,
            "-an",
            "-c:v", "mjpeg",
            "-q:v", "5",
            "-f", "image2pipe",
            "-vf", "fps=10",
            "pipe:1",
        ]

        try:
            log_fh = open(ffmpeg_log, "w")
            logger.info(f"FFmpeg cmd: {' '.join(cmd)}")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=log_fh,
                bufsize=0,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
            )
            self.processes[camera.id] = proc
            self._restart_counts[camera.id] = 0

            # Reader thread: parse JPEG frames from pipe
            t = threading.Thread(
                target=self._read_frames, args=(camera.id, proc), daemon=True
            )
            t.start()

            logger.info(f"Started stream for {camera.name} ({camera.id})")
        except Exception as e:
            logger.error(f"Failed to start FFmpeg for {camera.name}: {e}")

    def _read_frames(self, cam_id: str, proc: subprocess.Popen):
        """Read JPEG frames from FFmpeg stdout pipe."""
        buf = b""
        pipe = proc.stdout
        try:
            while self._running and cam_id in self.processes:
                chunk = pipe.read(4096)
                if not chunk:
                    break
                buf += chunk

                # Extract complete JPEG frames (SOI=FFD8, EOI=FFD9)
                while True:
                    start = buf.find(b"\xff\xd8")
                    if start == -1:
                        buf = b""
                        break
                    end = buf.find(b"\xff\xd9", start + 2)
                    if end == -1:
                        # Keep from start marker, wait for more data
                        buf = buf[start:]
                        break
                    # Complete frame
                    frame = buf[start : end + 2]
                    with self._frame_locks[cam_id]:
                        self._frames[cam_id] = frame
                    buf = buf[end + 2 :]
        except Exception as e:
            logger.error(f"Frame reader error for {cam_id}: {e}")
        finally:
            # FFmpeg exited — restart if still running
            if self._running and cam_id in self.processes:
                count = self._restart_counts.get(cam_id, 0)
                delay = min(2 ** count, 30)
                logger.warning(
                    f"FFmpeg for {cam_id} exited (code {proc.poll()}), restarting in {delay}s"
                )
                time.sleep(delay)
                self._restart_counts[cam_id] = count + 1
                if self._running and cam_id in self._cameras:
                    self.processes.pop(cam_id, None)
                    self.start_stream(self._cameras[cam_id])

    def get_frame(self, cam_id: str) -> Optional[bytes]:
        """Get the latest JPEG frame for a camera."""
        lock = self._frame_locks.get(cam_id)
        if lock is None:
            return None
        with lock:
            return self._frames.get(cam_id)

    def stop_stream(self, cam_id: str):
        proc = self.processes.pop(cam_id, None)
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._frames.pop(cam_id, None)
        self._cameras.pop(cam_id, None)
        stream_dir = STREAMS_DIR / cam_id
        if stream_dir.exists():
            shutil.rmtree(stream_dir, ignore_errors=True)

    def stop_all(self):
        self._running = False
        for cam_id in list(self.processes.keys()):
            self.stop_stream(cam_id)
